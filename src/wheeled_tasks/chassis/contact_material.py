"""Explicit wheel/body/ground contact material sampling.

The legacy chassis scene used one material for every robot collision shape and
an implicit ``0.5`` average-combine convention.  This module keeps the new
material contract data-only so the native environment can bind the exact
values and record the resulting effective wheel/ground coefficient.
"""
from __future__ import annotations

import math

import torch


_COMBINE_MODES = ("average", "min", "multiply", "max")


def _range(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
            or value[0] <= 0.0 or value[1] < value[0]):
        raise ValueError(f"{name} must be an ordered positive finite pair")
    return float(value[0]), float(value[1])


def _scalar(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return float(value)


def combine(a, b, mode):
    """Return the PhysX pair coefficient for two material coefficients."""
    if mode == "average":
        return (a + b) / 2.0
    if mode == "min":
        return torch.minimum(a, b)
    if mode == "multiply":
        return a * b
    if mode == "max":
        return torch.maximum(a, b)
    raise ValueError(f"Unknown friction combine mode: {mode}")


class ContactMaterialDomain:
    """Sample one static ground material per environment at startup.

    Wheel and chassis materials are fixed because they are authored on shared
    USD material bindings.  Ground coefficients are per-environment so a
    training batch can cover the requested low-grip or randomized band without
    changing the wheel material identity.
    """

    def __init__(self, count, config, seed, *, device="cpu", profiles=None):
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Contact material count must be positive")
        cfg = dict(config or {})
        allowed = {"enabled", "combine_mode", "ground_static_friction_range",
            "ground_dynamic_friction_range", "wheel_static_friction", "wheel_dynamic_friction",
            "body_static_friction", "body_dynamic_friction", "ground_sampling_bands"}
        if set(cfg) - allowed:
            raise ValueError(f"Unknown contact material settings: {sorted(set(cfg) - allowed)}")
        if cfg.get("enabled") is not True:
            raise ValueError("ContactMaterialDomain requires enabled=true")
        mode = cfg.get("combine_mode", "average")
        if mode not in _COMBINE_MODES:
            raise ValueError("Unknown contact material combine mode")
        self.cfg = cfg
        self.mode = mode
        self.ground_static_range = _range(cfg.get("ground_static_friction_range"),
                                          "ground_static_friction_range")
        self.ground_dynamic_range = _range(cfg.get("ground_dynamic_friction_range",
                                                   self.ground_static_range),
                                           "ground_dynamic_friction_range")
        self.wheel_static = _scalar(cfg.get("wheel_static_friction"), "wheel_static_friction")
        self.wheel_dynamic = _scalar(cfg.get("wheel_dynamic_friction", self.wheel_static),
                                     "wheel_dynamic_friction")
        self.body_static = _scalar(cfg.get("body_static_friction"), "body_static_friction")
        self.body_dynamic = _scalar(cfg.get("body_dynamic_friction", self.body_static),
                                    "body_dynamic_friction")
        bands = cfg.get("ground_sampling_bands")
        self.bands = []
        if bands is not None:
            if not isinstance(bands, list) or not bands:
                raise ValueError("ground_sampling_bands must be a nonempty explicit mixture")
            for band in bands:
                if not isinstance(band, dict) or set(band) != {"range", "fraction"}:
                    raise ValueError("Each ground band requires range and fraction")
                bounds = _range(band["range"], "ground_sampling_bands.range")
                fraction = _scalar(band["fraction"], "ground_sampling_bands.fraction")
                if bounds[0] < self.ground_static_range[0] or bounds[1] > self.ground_static_range[1]:
                    raise ValueError("Ground sampling band is outside the declared range")
                self.bands.append((bounds, fraction))
            if (not math.isclose(sum(weight for _, weight in self.bands), 1., abs_tol=1e-10)
                    or self.ground_dynamic_range != self.ground_static_range):
                raise ValueError("Ground mixture fractions must sum to one and share static/dynamic ranges")
        if self.ground_dynamic_range[0] > self.ground_static_range[0]:
            raise ValueError("Ground dynamic lower bound cannot exceed the static lower bound")
        if self.wheel_dynamic > self.wheel_static or self.body_dynamic > self.body_static:
            raise ValueError("Dynamic friction cannot exceed static friction")
        generator = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0x434F4E54)
        if profiles is not None:
            if len(profiles) != count:
                raise ValueError("One contact material profile is required per environment")
            static = torch.tensor([float(p["ground_static_friction"]) for p in profiles])
            dynamic = torch.tensor([float(p.get("ground_dynamic_friction", p["ground_static_friction"]))
                                    for p in profiles])
        else:
            if self.bands:
                weights = torch.tensor([weight for _, weight in self.bands], dtype=torch.float64)
                selected = torch.multinomial(weights, count, replacement=True, generator=generator)
                static = torch.empty(count)
                for index, (bounds, _) in enumerate(self.bands):
                    mask = selected == index
                    static[mask] = self._sample(int(mask.sum()), bounds, generator)
            else:
                static = self._sample(count, self.ground_static_range, generator)
            # Sampling the same band with an independent draw can accidentally
            # make kinetic friction exceed static friction.  Correlate equal
            # bands (the normal training recipe) and clamp differing bands to
            # the sampled static value.
            dynamic = (static.clone() if self.ground_dynamic_range == self.ground_static_range
                       else self._sample(count, self.ground_dynamic_range, generator).minimum(static))
        lo_s, hi_s = self.ground_static_range
        lo_d, hi_d = self.ground_dynamic_range
        if (not torch.isfinite(static).all() or not torch.isfinite(dynamic).all()
                or not ((static >= lo_s).all() and (static <= hi_s).all())
                or not ((dynamic >= lo_d).all() and (dynamic <= hi_d).all())
                or not (dynamic <= static).all()):
            raise ValueError("Ground contact material samples are outside their declared ranges")
        self.ground_static = static.to(device)
        self.ground_dynamic = dynamic.to(device)

    def reorder(self, indices):
        """Align sampled values with PhysX's returned articulation order."""
        self.ground_static = self.ground_static[indices]
        self.ground_dynamic = self.ground_dynamic[indices]

    def verify_native(self, stage, robot, env_paths):
        """Read USD bindings and PhysX shape coefficients after sim.reset()."""
        import warp as wp
        from pxr import UsdPhysics, UsdShade

        usd_counts = {"wheel": 0, "body": 0, "ground": 0}
        by_path = {path: i for i, path in enumerate(env_paths)}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            root = path.split("/Robot/")[0] if "/Robot/" in path else path.split("/Terrain/")[0]
            if root not in by_path:
                continue
            i = by_path[root]
            if "/Robot/" in path:
                wheel = any(f"/{name}/" in path for name in ("L_link3", "R_link3"))
                kind = "wheel" if wheel else "body"
                expected = ((self.wheel_static, self.wheel_dynamic) if wheel
                            else (self.body_static, self.body_dynamic))
            elif "/Terrain/" in path:
                kind = "ground"
                expected = (float(self.ground_static[i]), float(self.ground_dynamic[i]))
            else:
                continue
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            api = UsdPhysics.MaterialAPI(material.GetPrim())
            actual = (api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get())
            mode = material.GetPrim().GetAttribute("physxMaterial:frictionCombineMode").Get()
            if (any(value is None for value in actual) or mode != self.mode
                    or any(abs(a - b) > 1e-6 for a, b in zip(actual, expected))):
                raise RuntimeError(f"Contact material binding/readback mismatch at {path}: {actual}, {mode}, expected {expected}")
            usd_counts[kind] += 1
        if min(usd_counts.values()) == 0:
            raise RuntimeError(f"Missing separate wheel/body/ground collision materials: {usd_counts}")
        properties = wp.to_torch(robot.root_view.get_material_properties())
        static, dynamic = properties[..., 0], properties[..., 1]
        wheel = torch.isclose(static, static.new_tensor(self.wheel_static), atol=1e-6, rtol=0.)
        body = torch.isclose(static, static.new_tensor(self.body_static), atol=1e-6, rtol=0.)
        expected_dynamic = torch.where(wheel, dynamic.new_tensor(self.wheel_dynamic),
                                       dynamic.new_tensor(self.body_dynamic))
        if (not (wheel | body).all() or not wheel.any() or not body.any()
                or not torch.allclose(dynamic, expected_dynamic, atol=1e-6, rtol=0.)):
            raise RuntimeError("Separate robot shape materials were not imported by PhysX")
        return {"usd_collision_bindings_verified": True, "usd_collision_counts": usd_counts,
            "physx_robot_shape_coefficients_verified": True,
            "physx_first_robot_shape_materials": properties[0].cpu().tolist(),
            "physx_ground_coefficients": "USD static collision material bindings verified after scene import"}

    @staticmethod
    def _sample(count, bounds, generator):
        low, high = bounds
        return low + (high - low) * torch.rand(count, generator=generator)

    def effective(self, ground, wheel=True):
        """Return the pair coefficient using static/dynamic material values."""
        if wheel:
            static = ground.new_tensor(self.wheel_static)
            dynamic = ground.new_tensor(self.wheel_dynamic)
        else:
            static = ground.new_tensor(self.body_static)
            dynamic = ground.new_tensor(self.body_dynamic)
        return combine(ground, static, self.mode), combine(ground, dynamic, self.mode)

    def summary(self):
        wheel_static = self.ground_static.new_tensor(self.wheel_static)
        wheel_dynamic = self.ground_dynamic.new_tensor(self.wheel_dynamic)
        body_static = self.ground_static.new_tensor(self.body_static)
        body_dynamic = self.ground_dynamic.new_tensor(self.body_dynamic)
        wheel_s = combine(self.ground_static, wheel_static, self.mode)
        wheel_d = combine(self.ground_dynamic, wheel_dynamic, self.mode)
        body_s = combine(self.ground_static, body_static, self.mode)
        body_d = combine(self.ground_dynamic, body_dynamic, self.mode)
        return {
            "schema": "separate-wheel-body-ground-material-v1",
            "combine_mode": self.mode,
            "sampling": "one_ground_material_per_environment_at_startup",
            "ground_static_range": list(self.ground_static_range),
            "ground_dynamic_range": list(self.ground_dynamic_range),
            "ground_sampling_bands": self.cfg.get("ground_sampling_bands"),
            "ground_static_friction_per_environment": self.ground_static.cpu().tolist(),
            "ground_dynamic_friction_per_environment": self.ground_dynamic.cpu().tolist(),
            "wheel_static_friction": self.wheel_static,
            "wheel_dynamic_friction": self.wheel_dynamic,
            "body_static_friction": self.body_static,
            "body_dynamic_friction": self.body_dynamic,
            "sampled_ground_static_min_max": [float(self.ground_static.min()), float(self.ground_static.max())],
            "sampled_ground_dynamic_min_max": [float(self.ground_dynamic.min()), float(self.ground_dynamic.max())],
            "effective_wheel_static_min_max": [float(wheel_s.min()), float(wheel_s.max())],
            "effective_wheel_dynamic_min_max": [float(wheel_d.min()), float(wheel_d.max())],
            "effective_body_static_min_max": [float(body_s.min()), float(body_s.max())],
            "effective_body_dynamic_min_max": [float(body_d.min()), float(body_d.max())],
        }

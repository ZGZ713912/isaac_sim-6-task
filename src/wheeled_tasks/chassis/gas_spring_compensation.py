"""Measured elastic-load holding effort, shared with RMCS's frozen C++ table.

The force curve includes the mean effective gas-spring load, not separately
identified Coulomb friction or directional damping. Right force is a declared
left/right symmetry assumption; signed geometry is evaluated per side.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


def load_compensation(settings, root):
    cfg = dict(settings)
    if cfg.get("enabled") is not True:
        raise ValueError("Gas-spring compensation must be explicitly enabled")
    path = Path(cfg["model_file"])
    if not path.is_absolute():
        path = Path(root) / path
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != cfg["model_sha256"]:
        raise ValueError("Gas-spring compensation model hash differs")
    model = json.loads(content)
    if (model.get("schema") != "wheel-leg-gas-spring-compensation-v2"
            or model.get("coordinate") != "active_model_output"
            or model.get("direction_friction_compensated") is not False):
        raise ValueError("Unsupported gas-spring compensation model")
    gain = cfg.get("gain", 1.)
    if not math.isfinite(gain) or not 0. <= gain <= 1.:
        raise ValueError("Invalid gas-spring compensation gain")
    if cfg.get("force_extrapolation") != "positive_c1_endpoint_trend_v1":
        raise ValueError("Declare the unmeasured endpoint-trend extrapolation explicitly")
    if cfg.get("physical_curve_outside_geometry") != "hold_geometry_endpoint":
        raise ValueError("Declare the physical-force prior outside the geometry lookup")
    if cfg.get("relative_soft_limits") != dict(minimum=-.140638855,maximum=1.3889088544,
            margin=.03,projection="outward_relative_torque_v1"):
        raise ValueError("Gas-compensated training requires matching RMCS relative soft limits")
    return cfg, model


class GasSpringCompensation:
    def __init__(self, settings, *, root, device, dtype=torch.float32):
        self.cfg, self.model = load_compensation(settings, root)
        self.gain = self.cfg.get("gain", 1.)
        self.sides = []
        for side in ("left", "right"):
            row = self.model["sides"][side]
            self.sides.append({key: torch.tensor(row[key], device=device, dtype=dtype)
                for key in ("delta_rad", "beta_deg", "compression_m", "length_jacobian_m_per_rad")})
        self.compression = torch.tensor(self.model["compression_knots_m"], device=device, dtype=dtype)
        self.force_n = torch.tensor(self.model["force_knots_n"], device=device, dtype=dtype)
        ends = torch.stack([row["compression_m"][[0,-1]] for row in self.sides])
        self.geometry_compression_min = ends.min()
        self.geometry_compression_max = ends.max()
        for knots in (self.compression, *(row["delta_rad"] for row in self.sides)):
            if not bool(torch.isfinite(knots).all() and (knots[1:] > knots[:-1]).all()):
                raise ValueError("Gas-spring interpolation needs finite increasing knots")

    @staticmethod
    def interpolate(knots, values, x):
        bounded = x.clamp(knots[0], knots[-1])
        index = torch.searchsorted(knots, bounded.contiguous()).clamp(1, len(knots) - 1)
        u = (bounded - knots[index - 1]) / (knots[index] - knots[index - 1])
        if values.ndim > 1:
            u = u.unsqueeze(-1)
        return torch.lerp(values[index - 1], values[index], u)

    def physical_force(self, compression):
        # Preserve every measured segment. Outside, extend the endpoint slope
        # continuously; the low-compression exponential stays strictly positive.
        s = compression.clamp(self.geometry_compression_min,self.geometry_compression_max)
        measured = self.interpolate(self.compression, self.force_n, s)
        extension = self.model["force_extension"]
        low = self.force_n[0] * torch.exp(extension["low_compression_rate_per_m"] * (s-self.compression[0]))
        high = self.force_n[-1] + extension["high_compression_slope_n_per_m"] * (s-self.compression[-1])
        return torch.where(s<self.compression[0],low,torch.where(s>self.compression[-1],high,measured))

    def evaluate(self, q):
        """q[...,4] = (left hip, left aux, right hip, right aux), model radians."""
        if q.shape[-1] != 4:
            raise ValueError("Compensation requires four active model angles")
        deltas = (q[..., 1] - q[..., 0], q[..., 2] - q[..., 3])
        efforts, betas, forces, weights, extrapolated = [], [], [], [], []
        for delta, row in zip(deltas, self.sides):
            knots = row["delta_rad"]
            inside = (delta >= knots[0]) & (delta <= knots[-1])
            beta = self.interpolate(knots, row["beta_deg"], delta)
            compression = self.interpolate(knots, row["compression_m"], delta)
            jacobian = self.interpolate(knots, row["length_jacobian_m_per_rad"], delta)
            force = self.physical_force(compression)
            weight = self.gain * inside.to(q.dtype)
            efforts.append(-jacobian * (force * weight).unsqueeze(-1))
            betas.append(torch.where(inside, beta, float("nan")))
            forces.append(force * inside)
            weights.append(weight)
            extrapolated.append(((compression < self.compression[0]-1e-8)
                | (compression > self.compression[-1]+1e-8)) & inside)
        torque = torch.cat(efforts, dim=-1)
        # Preserve nonfinite input faults instead of hiding them behind a zero
        # out-of-domain mask. The environment's existing finite guard owns stops.
        torque = torch.where(torch.isfinite(q).all(-1, keepdim=True), torque, float("nan"))
        return torque, dict(beta_deg=torch.stack(betas, -1), force_n=torch.stack(forces, -1),
                           weight=torch.stack(weights, -1),extrapolated=torch.stack(extrapolated,-1))

    def summary(self):
        return dict(self.cfg, model_spec_sha256=self.model["model_spec_sha256"],
            measured_beta_domain_deg=self.model["measured_beta_domain_deg"],
            geometry_beta_domain_deg=self.model["geometry_beta_domain_deg"],
            right_leg_symmetry_assumed=self.model["right_leg_symmetry_assumed"],
            full_soft_limit_coverage=True,force_extrapolation="positive_c1_endpoint_trend_v1",
            direction_friction_compensated=False)

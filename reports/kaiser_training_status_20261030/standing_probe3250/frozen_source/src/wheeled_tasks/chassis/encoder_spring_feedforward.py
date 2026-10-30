"""Encoder-only gas-spring holding effort for the authenticated paired107 linkage.

The two driven root axes determine the passive knee on the retained assembly
branch. The measured axial curve is converted by virtual work, not by a fixed
lever arm. No simulator slider, passive-joint, contact or randomized force input
is accepted by this controller.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import torch
from torch import nn

from .standing_geometry107 import EXTENDED_SHA256, TABLE_RELATIVE, table_path
from .training_gas_spring import force_from_knots


def resolve(root, declaration, spring_settings, spec_path):
    expected = dict(enabled=True, geometry_table=TABLE_RELATIVE,
        geometry_sha256=EXTENDED_SHA256, interpolation="monotone_cubic_knee_v1",
        relative_margin_rad=.001, invalid_geometry="zero_feedforward", total_leg_limit_nm=40.)
    if declaration != expected:
        raise ValueError("Gas-spring compensation requires the explicit paired107 contract")
    path = table_path()
    if path != root / declaration["geometry_table"]:
        raise ValueError("Spring feedforward geometry belongs to another source tree")
    table = json.loads(path.read_text())
    spec = json.loads(spec_path.read_text())
    spec_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    if table["model_spec_sha256"] != spec_sha:
        raise ValueError("Spring feedforward geometry does not match the installed mechanism")
    joints = {j["name"]: j for j in spec["joints"]}
    sides = []
    for name, side in (("left", "L"), ("right", "R")):
        knee = joints["L_joint2" if side == "L" else "R_jonit2"]
        lower = next(j for j in spec["joints"] if j["child"] == side * 3 + "_link2")
        upper = next(c for c in spec["constraints"] if c["name"] == side + "_spring_upper_mount")
        binding = spec["spring_binding"][side + "_spring_slide"]
        source = table["sides"][name]
        sides.append(dict(delta_rad=source["delta_rad"],
            beta_rad=[math.radians(v) for v in source["passive_inner_deg"]],
            knee_rotation=[row[:3] for row in knee["origin"][:3]],
            origin_to_upper_m=[knee["origin"][i][3]-upper["local_pos0_m"][i] for i in range(3)],
            lower_in_shank_m=[lower["origin"][i][3] for i in range(3)],
            knee_axis=knee["axis"], knee_sign=1. if side == "L" else -1.,
            extended_length_m=binding["full_extension_pin_distance_m"]))
    return dict(deepcopy(declaration), model_spec_sha256=spec_sha,
        spring_model_sha256=spring_settings["sha256"], sides=sides,
        force_model=deepcopy(spring_settings["model"]),
        outside_identified_domain=spring_settings["outside_identified_domain"],
        trend_extension=deepcopy(spring_settings.get("resolved_trend_extension")),
        encoder_order=["L_joint1", "LL_joint1", "R_joint1", "RR_joint1"],
        force_scope="measured effective mean axial load; not separated pressure or hysteresis",
        formula="clip(PD - J_length(q_encoder).T * F(extended_length-length), -40, 40)")


def monotone_slopes(x, y):
    """C1 shape-preserving Hermite slopes, in radians/radian."""
    width, secant = x.diff(), y.diff()/x.diff()
    if len(x) == 2:
        return secant.repeat(2)
    w1, w2 = 2.*width[1:]+width[:-1], width[1:]+2.*width[:-1]
    middle = (w1+w2)/(w1/secant[:-1]+w2/secant[1:])
    middle = torch.where(secant[:-1]*secant[1:] > 0., middle, 0.)

    def end(h0, h1, d0, d1):
        value = ((2.*h0+h1)*d0-h0*d1)/(h0+h1)
        value = torch.where(value*d0 <= 0., 0., value)
        return torch.where((d0*d1 < 0.) & (value.abs() > 3.*d0.abs()), 3.*d0, value)

    return torch.cat((end(width[0], width[1], secant[0], secant[1]).reshape(1), middle,
        end(width[-1], width[-2], secant[-1], secant[-2]).reshape(1)))


class EncoderGasSpringFeedforward(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = deepcopy(settings)
        for i, side in enumerate(settings["sides"]):
            for key, value in side.items():
                self.register_buffer(f"side{i}_{key}", torch.tensor(value, dtype=torch.float64))
            x, y = self.get(i, "delta_rad"), self.get(i, "beta_rad")
            if (x.ndim != 1 or x.shape != y.shape or len(x) < 3
                    or not bool((x.diff() > 0).all())
                    or not bool((y.diff() > 0).all() or (y.diff() < 0).all())):
                raise ValueError("Spring feedforward needs a single-valued retained knee branch")
            self.register_buffer(f"side{i}_slopes", monotone_slopes(x, y))
        model = settings["force_model"]
        self.register_buffer("force_knots", torch.tensor(model["compression_knots_m"], dtype=torch.float64))
        self.register_buffer("force_values", torch.tensor(model["force_knots_n"], dtype=torch.float64))
        self.register_buffer("force_slopes", self.force_values.diff()/self.force_knots.diff())

    def get(self, side, key):
        return getattr(self, f"side{side}_{key}")

    def forward(self, encoder_position):
        if encoder_position.ndim != 2 or encoder_position.shape[1] != 4:
            raise ValueError("Need the four calibrated driven leg encoder angles")
        if not bool(torch.isfinite(encoder_position).all()):
            raise ValueError("Nonfinite encoder input to spring compensation")
        replay = getattr(self, "tensor_replay", None)
        holding, self.last = (self._evaluate(encoder_position) if replay is None
                              else replay(encoder_position))
        return holding

    def _evaluate(self, encoder_position):
        """Original pure math; validity checks stay in forward for every call."""
        beta_all, length_all, derivative_all, valid_all = [], [], [], []
        for side in range(2):
            x, y, m = (self.get(side, key) for key in ("delta_rad", "beta_rad", "slopes"))
            difference = encoder_position[:, 2*side+1]-encoder_position[:, 2*side]
            delta = torch.atan2(difference.sin(), difference.cos())
            margin = self.settings["relative_margin_rad"]
            valid = (delta >= x[0]-margin) & (delta <= x[-1]+margin)
            bounded = delta.clamp(x[0], x[-1])
            lo = torch.searchsorted(x, bounded.contiguous(), right=True).sub(1).clamp(0, len(x)-2)
            h, t = x[lo+1]-x[lo], (bounded-x[lo])/(x[lo+1]-x[lo])
            # Differentiate the SAME C1 interpolation used to estimate length.
            beta = ((2*t**3-3*t*t+1)*y[lo]+(t**3-2*t*t+t)*h*m[lo]
                    +(-2*t**3+3*t*t)*y[lo+1]+(t**3-t*t)*h*m[lo+1])
            beta_delta = ((6*t*t-6*t)*y[lo]/h+(3*t*t-4*t+1)*m[lo]
                          +(-6*t*t+6*t)*y[lo+1]/h+(3*t*t-2*t)*m[lo+1])
            # The existing encoder branch permits1mrad numerical penetration.
            # A tangent extension keeps holding torque continuous at the stop;
            # larger branch violations get zero feedforward and stay invalid.
            beta = beta+torch.where(delta < x[0], m[0]*(delta-x[0]),
                torch.where(delta > x[-1], m[-1]*(delta-x[-1]), 0.))
            axis, lower = self.get(side, "knee_axis"), self.get(side, "lower_in_shank_m")
            angle = self.get(side, "knee_sign")*(beta-(math.pi-2.3573))
            c, s = angle.cos()[:, None], angle.sin()[:, None]
            rotated = c*lower+s*torch.linalg.cross(axis, lower)+(1.-c)*axis*(axis*lower).sum()
            displacement = rotated @ self.get(side, "knee_rotation").T+self.get(side, "origin_to_upper_m")
            length = displacement.norm(dim=-1)
            dp_beta = (torch.linalg.cross(axis.expand_as(rotated), rotated)
                       @ self.get(side, "knee_rotation").T)*self.get(side, "knee_sign")
            dl_delta = (displacement*dp_beta).sum(-1)/length*beta_delta
            beta_all.append(beta)
            length_all.append(length)
            derivative_all.append(torch.where(valid, dl_delta, 0.))
            valid_all.append(valid)
        length, beta, valid = (torch.stack(v, -1) for v in (length_all, beta_all, valid_all))
        extended = torch.stack([self.get(i, "extended_length_m") for i in range(2)])
        compression = extended-length
        force = force_from_knots(compression, self.force_knots, self.force_values, self.force_slopes,
            self.settings["outside_identified_domain"], self.settings["force_model"]["stroke_m"])
        d = torch.stack(derivative_all, -1)
        jacobian = torch.stack((-d, d), -1).flatten(-2)
        holding = -jacobian*force.repeat_interleave(2, -1)
        last = dict(encoder_position=encoder_position.clone(), knee_inner_rad=beta,
            length_m=length, compression_m=compression, axial_force_n=force,
            length_jacobian_m_rad=jacobian, holding_effort_nm=holding, geometry_valid=valid,
            outside_identified=(compression < self.force_knots[0]) | (compression > self.force_knots[-1]))
        return holding, last

    def motor_efforts(self, pd_effort, encoder_position):
        holding = self(encoder_position)
        requested = pd_effort+holding
        limited = requested.clamp(-40., 40.)
        self.last.update(pd_effort_nm=pd_effort.clone(), requested_effort_nm=requested,
                         limited_effort_nm=limited.clone())
        return limited

    def install(self, env):
        self.to(device=env.device, dtype=torch.float32)
        env.spring_feedforward = self
        env.startup_report["encoder_spring_feedforward"] = dict(
            enabled=True, controller_hz=round(1./env.dt), input=self.settings["encoder_order"],
            input_source="same sampled/delayed low-level encoder feedback as PD, before target winding",
            geometry_sha256=self.settings["geometry_sha256"],
            model_spec_sha256=self.settings["model_spec_sha256"],
            spring_model_sha256=self.settings["spring_model_sha256"],
            formula=self.settings["formula"], interpolation=self.settings["interpolation"],
            relative_margin_rad=self.settings["relative_margin_rad"],
            invalid_geometry="zero feedforward; original protection retained",
            outside_identified_domain=self.settings["outside_identified_domain"],
            trend_extension=deepcopy(self.settings.get("trend_extension")),
            physical_spring_retained=True, total_leg_limit_nm=40.,
            privileged_force_or_slider_input=False, disabled_motor_zero_after_compensation=True,
            force_scope=self.settings["force_scope"], hardware_ready=False)

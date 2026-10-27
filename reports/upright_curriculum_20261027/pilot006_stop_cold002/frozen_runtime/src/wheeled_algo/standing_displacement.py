"""Optional encoder-based standing observation, without motor control outputs.

This planar coordinate is meaningful while upright on a locally flat floor.
It uses continuous wheel-side angles, leg encoders and projected gravity.
It does not provide ground-truth position or remove wheel slip. Existing
actors/environments do not import or enable this module automatically.
"""
import math

import torch
from torch import nn

from .rolling_velocity import frozen_rolling_estimator


def observation_contract():
    """Versioned observation semantics; this is not the old zero-wheel ABI."""
    from .rolling_velocity import TABLE_SHA256, SPEC_SHA256
    return dict(schema="wheel-leg-standing-displacement-v1", policy_hz=200,
                slots=[14, 16], scale_per_m=20., filter_seconds=.05, limit_m=.25,
                source="continuous_model_wheel_angles_delayed_with_leg_encoders_and_gravity",
                capture="once_after_0.2s_active_quiet_gate_ge_.999_and_recovery_gate_le_.05",
                request="abs_vx_lt_.1_and_abs_yaw_lt_.2", reset="episode_or_RL_entry_or_request_end",
                invalid="zero_output_preserve_reference", legacy_network_slots="forced_zero",
                model_spec_sha256=SPEC_SHA256, geometry_table_sha256=TABLE_SHA256)


def validate_observation_contract(settings):
    enabled = settings.get("quiet_velocity_feedback_mode") == "standing_displacement"
    declared = settings.get("standing_position_observation")
    if enabled:
        if not settings.get("quiet_velocity_feedback") or declared != observation_contract():
            raise ValueError("Standing displacement requires the exact versioned observation contract")
    elif declared is not None:
        raise ValueError("Standing displacement contract requires its explicit feedback mode")
    return enabled


class WheelAngleCoordinate(nn.Module):
    """Absolute forward rolling coordinate of each wheel in metres.

Positions are model radians in policy order [four legs, two wheels]. The
wheel angles must be continuous, already corrected for the physical gearbox
and axis signs. Only relative coordinate differences have a useful origin.
"""

    def __init__(self):
        super().__init__()
        self.geometry = frozen_rolling_estimator()

    def forward(self, positions, gravity):
        q, wheels = positions[:, :4], positions[:, 4:6]
        norm = gravity.norm(dim=-1, keepdim=True)
        valid = (torch.isfinite(positions).all(-1, keepdim=True)
                 & torch.isfinite(gravity).all(-1, keepdim=True)
                 & (norm > .5) & (norm < 1.5))
        direction = gravity / norm.clamp_min(1e-9)
        # Exclude near-sideways/inverted geometry rather than interpreting a
        # pitch-angle branch crossing as a real forward displacement.
        valid = valid & (direction[:, 2:3] < -.94)
        pitch = torch.atan2(direction[:, :1], -direction[:, 2:3])
        coordinates = []
        for side in range(2):
            get = lambda key: getattr(self.geometry, f"side{side}_{key}")
            knots, points = get("delta_rad"), get("wheel_center_b_m")
            i = 2 * side
            difference = q[:, i+1] - q[:, i]
            delta = torch.atan2(difference.sin(), difference.cos())
            valid = valid & (delta[:, None] >= knots[0]) & (delta[:, None] <= knots[-1])
            upper = (delta[:, None] >= knots[None]).to(torch.int64).sum(-1).clamp(1, len(knots)-1)
            lower = upper - 1
            blend = ((delta-knots[lower])/(knots[upper]-knots[lower])).clamp(0., 1.)
            point = points[lower] + blend[:, None] * (points[upper]-points[lower])
            axis, origin = get("hip_axis_b"), get("hip_origin_b_m")
            angle = (q[:, i]-get("hip_reference_rad"))[:, None]
            value = point-origin
            center = (angle.cos()*value + angle.sin()*torch.linalg.cross(axis.expand_as(value), value)
                      + (1.-angle.cos())*axis*(axis*value).sum(-1, keepdim=True) + origin)
            center_x = pitch.cos()*center[:, :1] + pitch.sin()*center[:, 2:3]
            beta_knots = get("passive_inner_deg") * (math.pi/180.)
            beta = beta_knots[lower] + blend*(beta_knots[upper]-beta_knots[lower])
            spin = q[:, i:i+1] + (beta[:, None] if side == 0 else -beta[:, None]) + wheels[:, side:side+1]
            coordinate = get("wheel_radius_m") * (axis[1].sign()*spin + pitch) - center_x
            coordinates.append(coordinate)
        result = torch.cat(coordinates, -1)
        valid = valid & torch.isfinite(result).all(-1, keepdim=True)
        return torch.where(valid, result, torch.zeros_like(result)), valid


class StandingDisplacement:
    """Observation reference and low pass; no leg/wheel commands or PD loop.

Capture once on a valid eligible quiet request. Keep the same reference
through noise, brief tilt and invalid readings; they cannot silently erase
displacement. The caller explicitly resets episode/entry references. A
nonzero motion request ends the reference. Invalid samples expose zeros and
preserve the stored reference/filter until valid measurements resume.
"""

    def __init__(self, count, device, dt=.005, filter_seconds=.05, limit_m=.25):
        if not all(math.isfinite(v) and v > 0. for v in (dt, filter_seconds, limit_m)):
            raise ValueError("Standing observation time and displacement bounds must be finite and positive")
        self.alpha = dt/(filter_seconds+dt)
        self.limit_m = limit_m
        self.anchor = torch.zeros(count, 2, device=device)
        self.filtered = torch.zeros_like(self.anchor)
        self.active = torch.zeros(count, dtype=torch.bool, device=device)
        self.last_tick = torch.full((count,), -1, dtype=torch.long, device=device)

    def reset(self, ids):
        self.active[ids] = False
        self.anchor[ids] = 0.
        self.filtered[ids] = 0.
        self.last_tick[ids] = -1

    def update(self, coordinate, valid, requested, eligible, sample_tick=None):
        valid = valid.reshape(-1).bool() & torch.isfinite(coordinate).all(-1)
        requested, eligible = requested.bool(), eligible.bool()
        ended = ~requested
        self.active[ended] = False
        self.filtered[ended] = 0.
        self.last_tick[ended] = -1
        start = requested & eligible & valid & ~self.active
        self.anchor[start] = coordinate[start]
        self.filtered[start] = 0.
        self.active[start] = True
        use = self.active & requested & valid
        fresh = use if sample_tick is None else use & (self.last_tick != sample_tick)
        displacement = (coordinate-self.anchor).clamp(-self.limit_m, self.limit_m)
        # Indexing protects persistent state from NaN/Inf in rejected samples.
        self.filtered[fresh] += self.alpha*(displacement[fresh]-self.filtered[fresh])
        if sample_tick is not None:
            self.last_tick[fresh] = sample_tick
        output = torch.where(use[:, None], self.filtered, torch.zeros_like(self.filtered))
        return output, use

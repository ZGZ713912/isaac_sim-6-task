"""Single-frame rolling velocity from the sealed107 mechanism and existing35D.

This is a model-derived no-slip estimate, not a calibrated sensor. Invalid knee
geometry disables it. It has no simulator state, history or contact-truth input.
"""
import math
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

SPEC_SHA256 = "0ac82045ab584f20091538aa782d93ea545382dc701faca1f8ae92aeb7cd1997"
TABLE_SHA256 = "57d6516792a40f65c84f66c36d72f14b039b4b61a9779a7b210fe42465befb69"
TABLE_RELATIVE = "reports/flat_base_continuation_20261019/low_friction_candidate001/height_lookup.json"


def frozen_table_path():
    path = Path(__file__).resolve().parents[2] / TABLE_RELATIVE
    if hashlib.sha256(path.read_bytes()).hexdigest() != TABLE_SHA256:
        raise ValueError("Frozen107 rolling geometry table changed")
    return path


def frozen_rolling_estimator():
    return RollingBaseVelocity(json.loads(frozen_table_path().read_text()))


class RollingBaseVelocity(nn.Module):
    def __init__(self, table):
        super().__init__()
        if table.get("model_spec_sha256") != SPEC_SHA256:
            raise ValueError("Rolling velocity requires the authenticated107 geometry")
        self.register_buffer("nominal", torch.tensor([-.42, .13742282595395358, .42, -.1374155762580851]))
        for index, name in enumerate(("left", "right")):
            source = table["sides"][name]
            for key in ("delta_rad", "wheel_center_b_m", "hip_origin_b_m", "hip_axis_b",
                        "hip_reference_rad", "wheel_axis_b", "wheel_radius_m", "passive_inner_deg"):
                value = torch.tensor(source[key], dtype=torch.float32)
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite rolling geometry")
                self.register_buffer(f"side{index}_{key}", value)
            knots = getattr(self, f"side{index}_delta_rad")
            if knots.ndim != 1 or len(knots) < 2 or not bool((knots[1:] > knots[:-1]).all()):
                raise ValueError("Rolling geometry needs increasing relative-angle knots")

    def verify_frozen_geometry(self):
        expected = frozen_rolling_estimator().state_dict()
        actual = self.state_dict()
        if actual.keys() != expected.keys() or any(
                not torch.equal(value.detach().cpu(), expected[name]) for name, value in actual.items()):
            raise ValueError("Rolling estimator geometry buffers changed")

    def forward(self, obs):
        q = obs[:, 10:14] + self.nominal
        dq, omega, gravity = obs[:, 16:22] * 10., obs[:, 4:7] * 2., obs[:, 7:10]
        estimates, validity = [], []
        for side in range(2):
            i = 2 * side
            get = lambda key: getattr(self, f"side{side}_{key}")
            knots, points = get("delta_rad"), get("wheel_center_b_m")
            difference = q[:, i + 1] - q[:, i]
            delta = torch.atan2(difference.sin(), difference.cos())
            valid = torch.isfinite(delta) & (delta >= knots[0]) & (delta <= knots[-1])
            # Comparison/count/gather also have standard ONNX equivalents.
            upper = (delta[:, None] >= knots[None, :]).to(torch.int64).sum(-1).clamp(1, len(knots) - 1)
            lower = upper - 1
            width = knots[upper] - knots[lower]
            blend = ((delta - knots[lower]) / width).clamp(0., 1.)
            point = points[lower] + blend[:, None] * (points[upper] - points[lower])
            derivative = (points[upper] - points[lower]) / width[:, None]
            axis = get("hip_axis_b")
            angle = q[:, i] - get("hip_reference_rad")
            cosine, sine = angle.cos()[:, None], angle.sin()[:, None]

            def rotate(value):
                return (cosine * value + sine * torch.linalg.cross(axis.expand_as(value), value)
                        + (1. - cosine) * axis * (axis * value).sum(-1, keepdim=True))

            relative = rotate(point - get("hip_origin_b_m"))
            center = relative + get("hip_origin_b_m")
            delta_velocity = dq[:, i + 1] - dq[:, i]
            center_velocity = (torch.linalg.cross(axis.expand_as(relative), relative) * dq[:, i:i+1]
                               + rotate(derivative) * delta_velocity[:, None])
            beta = get("passive_inner_deg") * (math.pi / 180.)
            beta_velocity = (beta[upper] - beta[lower]) / width * delta_velocity
            knee_velocity = beta_velocity if side == 0 else -beta_velocity
            # The107 source FK proves hip, passive knee and wheel positive
            # axes coincide on each side (right positive axis is reversed).
            relative_spin = axis * (dq[:, i] + knee_velocity + dq[:, 4 + side])[:, None]
            axle = get("wheel_axis_b")
            downward = gravity - axle * (gravity * axle).sum(-1, keepdim=True)
            norm = downward.norm(dim=-1, keepdim=True)
            contact_offset = get("wheel_radius_m") * downward / norm.clamp_min(1e-9)
            velocity = (-torch.linalg.cross(omega, center + contact_offset) - center_velocity
                        - torch.linalg.cross(relative_spin, contact_offset))
            valid = valid & (norm[:, 0] > 1e-6) & torch.isfinite(velocity).all(-1)
            estimates.append(velocity[:, :1])
            validity.append(valid[:, None])
        valid = validity[0] & validity[1]
        estimate = .5 * (estimates[0] + estimates[1])
        return torch.where(valid, estimate, torch.zeros_like(estimate)), valid

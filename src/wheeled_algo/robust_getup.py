"""Unified six-motor recovery, mixed motion and quiet standing, exportable as 35->6.

All authority and smoothing depend only on deployable IMU, encoders, requests
and the previous applied action. No hidden FSM or simulation state is needed.
"""
import torch
from torch import nn

from .unified_getup import UnifiedGetupActor


KIND = "wheel-leg-robust-getup-v2"


def tilt_gate(obs):
    return ((obs[:, 9:10] + .94) / .44).clamp(0., 1.)


def quiet_gate(obs):
    request = torch.maximum(obs[:, :1].abs() / .12, obs[:, 2:3].abs() / .25)
    tilt = ((-obs[:, 9:10] - .94) / .04).clamp(0., 1.)
    # Release damping during a balance disturbance. Encoder speed is not used
    # here: wheel/leg jitter itself must not repeatedly disable quiet control.
    gyro = (obs[:, 4:6].norm(dim=-1, keepdim=True) * 2. - .4).clamp(0., 1.)
    return (1. - request).clamp(0., 1.) * tilt * (1. - gyro)


class RobustGetupActor(UnifiedGetupActor):
    def __init__(self, parent, *, quiet_alpha=(.25, .25, .25, .25, .65, .65)):
        super().__init__(parent)
        self.quiet = nn.Sequential(nn.Linear(35, 64), nn.ELU(), nn.Linear(64, 6))
        nn.init.zeros_(self.quiet[-1].weight)
        nn.init.zeros_(self.quiet[-1].bias)
        self.register_buffer("quiet_alpha", torch.tensor(quiet_alpha))
        self.register_buffer("quiet_bound", torch.tensor([.4] * 4 + [.5] * 2))

    def forward(self, obs):
        features = torch.tanh(obs * self.feature_scale)
        gate = tilt_gate(obs)
        arc_gate = ((obs[:, :1].abs() - .05) / .2).clamp(0., 1.) * ((obs[:, 2:3].abs() - .05) / .2).clamp(0., 1.)
        motion = self.parent(obs) + arc_gate * self.arc_bound * torch.tanh(self.arc(features))
        motion = torch.maximum(torch.minimum(motion, self.motion_bound), -self.motion_bound)
        quiet = quiet_gate(obs)
        smoothed = obs[:, 22:28] + self.quiet_alpha * (motion - obs[:, 22:28])
        motion = motion + quiet * (smoothed - motion + self.quiet_bound * torch.tanh(self.quiet(features)))
        raw = self.recovery(features)
        legs = (obs[:, 22:26] + 1.2 * torch.tanh(raw[:, :4])).clamp(-13., 13.)
        fallen = torch.cat((legs, 9. * torch.tanh(raw[:, 4:6])), -1)
        result = motion + gate * (fallen - motion)
        return torch.maximum(torch.minimum(result, self.action_bound), -self.action_bound)

    def initialize_from_unified(self, state):
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != {
                "quiet.0.weight", "quiet.0.bias", "quiet.2.weight", "quiet.2.bias", "quiet_alpha", "quiet_bound"}:
            raise ValueError("Unexpected unified-to-robust actor migration")


class SensorConditionedMean(nn.Module):
    """RSL Gaussian input: actual mean and an observation-dependent std scale.

The distribution, stored log-probabilities and PPO all use the same std. Quiet
standing is not destroyed by the larger exploration needed while overturned.
"""
    def __init__(self, mean):
        super().__init__()
        self.mean = mean
        self.log_std = nn.Parameter(torch.full((6,), -.6931471805599453))

    def forward(self, obs):
        recovery = tilt_gate(obs)
        scale = .12 + .88 * recovery
        scale = scale * (1. - .65 * quiet_gate(obs))
        log_std = self.log_std.clamp(-3., -.22314355) + scale.log()
        return torch.stack((self.mean(obs), log_std.expand(-1, 6)), -2)

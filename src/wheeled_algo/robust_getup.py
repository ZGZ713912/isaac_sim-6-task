"""Unified six-motor recovery, mixed motion and quiet standing, exportable as 35->6.

All authority and smoothing depend only on deployable IMU, encoders, requests
and the previous applied action. No hidden FSM or simulation state is needed.
"""
import torch
from torch import nn

from .unified_getup import UnifiedGetupActor
from .getup_symmetry import canonical_recovery_logits
from .paired_recovery import paired_offsets
from .recovery_coordinates import smooth_recovery_legs


KIND = "wheel-leg-robust-getup-v2"


def tilt_gate(obs, stance_aware=False, shape_aware=False):
    original = ((obs[:, 9:10] + .94) / .44).clamp(0., 1.)
    if not stance_aware:
        return original
    # Keep the established balance policy during an ordinary lean. A leg far
    # from its standing phase retains the learned getup authority on the way up.
    severe = ((obs[:, 9:10] + .5) / (.5 - .08715574274765817)).clamp(0., 1.)
    phase = ((obs[:, 10:14].abs().amax(-1, keepdim=True) - 1.) / .8).clamp(0., 1.)
    if shape_aware:
        # A common leg rotation can cross the standing phase while the knee
        # is still collapsed/extended. Retain capture authority until that
        # paired geometry also approaches the nominal standing mechanism.
        legs = obs[:, 10:14].unflatten(-1, (2, 2))
        difference = legs[..., 1] - legs[..., 0]
        difference = torch.atan2(difference.sin(), difference.cos())
        shape = ((difference.abs().amax(-1, keepdim=True) - .2) / .2).clamp(0., 1.)
        phase = torch.maximum(phase, shape)
    return torch.maximum(severe, phase * original)


def quiet_gate(obs):
    request = torch.maximum(obs[:, :1].abs() / .12, obs[:, 2:3].abs() / .25)
    tilt = ((-obs[:, 9:10] - .94) / .04).clamp(0., 1.)
    # Release damping during a balance disturbance. Encoder speed is not used
    # here: wheel/leg jitter itself must not repeatedly disable quiet control.
    gyro = (obs[:, 4:6].norm(dim=-1, keepdim=True) * 2. - .4).clamp(0., 1.)
    return (1. - request).clamp(0., 1.) * tilt * (1. - gyro)


class RobustGetupActor(UnifiedGetupActor):
    def __init__(self, parent, *, quiet_alpha=(.25, .25, .25, .25, .65, .65),
                 canonical_recovery=False, motion_residual_fraction=0.,
                 cyclic_leg_targets=False, leg_action_clip=13., recovery_leg_step=1.2,
                 smooth_knee_targets=False, recovery_knee_center=(.924544, .924537),
                 quiet_velocity_feature_scale=None, quiet_wheel_bound=.5, stance_aware_recovery=False,
                 recovery_shape_aware=False, quiet_high_common_alpha=None):
        super().__init__(parent)
        self.canonical_recovery = canonical_recovery
        self.motion_residual_fraction = motion_residual_fraction
        self.cyclic_leg_targets = cyclic_leg_targets
        self.recovery_leg_step = recovery_leg_step
        self.smooth_knee_targets = smooth_knee_targets
        self.recovery_knee_center = recovery_knee_center
        self.quiet_velocity_feature_scale = quiet_velocity_feature_scale
        self.stance_aware_recovery = stance_aware_recovery
        if recovery_shape_aware and not stance_aware_recovery:
            raise ValueError("Knee-shape recovery requires stance-aware authority")
        self.recovery_shape_aware = recovery_shape_aware
        if quiet_high_common_alpha is not None and not 0. < quiet_high_common_alpha <= 1.:
            raise ValueError("High-height common-angle interpolation must be in (0,1]")
        self.quiet_high_common_alpha = quiet_high_common_alpha
        self.action_bound[:4].fill_(leg_action_clip)
        self.quiet = nn.Sequential(nn.Linear(35, 64), nn.ELU(), nn.Linear(64, 6))
        nn.init.zeros_(self.quiet[-1].weight)
        nn.init.zeros_(self.quiet[-1].bias)
        self.register_buffer("quiet_alpha", torch.tensor(quiet_alpha))
        self.register_buffer("quiet_bound", torch.tensor([.4] * 4 + [quiet_wheel_bound] * 2))

    def forward(self, obs):
        features = torch.tanh(obs * self.feature_scale)
        gate = tilt_gate(obs, self.stance_aware_recovery, self.recovery_shape_aware)
        arc_gate = ((obs[:, :1].abs() - .05) / .2).clamp(0., 1.) * ((obs[:, 2:3].abs() - .05) / .2).clamp(0., 1.)
        moving = torch.maximum(obs[:, :1].abs() / .1, obs[:, 2:3].abs() / .2).clamp(0., 1.)
        residual_gate = arc_gate + self.motion_residual_fraction * (1. - arc_gate) * moving
        motion = self.parent(obs) + residual_gate * self.arc_bound * torch.tanh(self.arc(features))
        motion = torch.maximum(torch.minimum(motion, self.motion_bound), -self.motion_bound)
        quiet = quiet_gate(obs)
        smoothed = obs[:, 22:28] + self.quiet_alpha * (motion - obs[:, 22:28])
        if self.quiet_high_common_alpha is not None:
            # Separate paired balance rotation from knee opening. At high
            # stance the extra common-angle filter delay can destabilize the
            # already delayed IMU/encoder feedback. Keep knee/height smoothing
            # and the existing low/mid-height policy exactly unchanged.
            high = ((obs[:, 3:4] / 5. - .35) / .05).clamp(0., 1.)
            common = (motion[:, :4] - obs[:, 22:26]).unflatten(-1, (2, 2)).mean(-1, keepdim=True)
            common = common.expand(-1, -1, 2).flatten(-2)
            extra = high * (self.quiet_high_common_alpha - self.quiet_alpha[:4]) * common
            smoothed = torch.cat((smoothed[:, :4] + extra, smoothed[:, 4:]), -1)
        quiet_features = features
        if self.quiet_velocity_feature_scale is not None:
            quiet_features = torch.cat((features[:, :16], torch.tanh(obs[:, 16:22] * self.quiet_velocity_feature_scale), features[:, 22:]), -1)
        motion = motion + quiet * (smoothed - motion + self.quiet_bound * torch.tanh(self.quiet(quiet_features)))
        raw = (canonical_recovery_logits(self.recovery, obs, self.feature_scale)
               if self.canonical_recovery else self.recovery(features))
        legs = obs[:, 22:26] + self.recovery_leg_step * torch.tanh(raw[:, :4])
        if self.smooth_knee_targets:
            legs = smooth_recovery_legs(raw[:, :4], obs[:, 22:26], obs[:, 10:14],
                                        self.recovery_leg_step, self.recovery_knee_center)
        elif self.cyclic_leg_targets:
            legs = paired_offsets(legs * .25, obs[:, 10:14]) / .25
        else:
            legs = legs.clamp(-13., 13.)
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
        recovery = tilt_gate(obs, self.mean.stance_aware_recovery, self.mean.recovery_shape_aware)
        scale = .12 + .88 * recovery
        scale = scale * (1. - .65 * quiet_gate(obs))
        log_std = self.log_std.clamp(-3., -.22314355) + scale.log()
        return torch.stack((self.mean(obs), log_std.expand(-1, 6)), -2)

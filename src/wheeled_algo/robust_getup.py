"""Unified six-motor recovery, mixed motion and quiet standing, exportable as 35->6.

All authority and smoothing depend only on deployable IMU, encoders, requests
and the previous applied action. No hidden FSM or simulation state is needed.
"""
import torch
from torch import nn
from copy import deepcopy

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


def fast_forward_reference_gate(obs, speed_range=(1.5, 2.), yaw_fade=.25):
    """Blend to authenticated original feedback over the declared speed range.

    Reverse, low commands and quiet standing keep their learned controller.
    A None yaw fade explicitly retains the reference during mixed commands.
    The gate uses only the declared requests.
    """
    start, full = speed_range
    speed = ((obs[:, :1] - start) / (full - start)).clamp(0., 1.)
    if yaw_fade is None:
        return speed
    return speed * (1. - obs[:, 2:3].abs() / yaw_fade).clamp(0., 1.)


class QuietVelocityFeedback(nn.Module):
    """Zero-initialized feedback paths; legacy velocity modes remain bias-free."""

    def __init__(self, mode="all"):
        super().__init__()
        if mode not in ("all", "wheels", "wheel_memory", "coupled_state"):
            raise ValueError("Quiet velocity feedback mode must be all, wheels, wheel_memory or coupled_state")
        self.mode = mode
        size = 4 if mode == "coupled_state" else 6 if mode == "all" else 2
        inputs = 16 if mode == "coupled_state" else 4 if mode == "wheel_memory" else size
        self.weight = nn.Parameter(torch.zeros(size, inputs))
        if mode == "coupled_state":
            self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, velocity):
        if self.mode == "coupled_state":
            # Same-leg common rotations preserve the paired knee difference.
            # This additive bounded correction is applied after the old tanh,
            # so unequal old logits cannot turn it into a knee-opening change.
            coupled = .5 * torch.tanh(torch.nn.functional.linear(velocity, self.weight, self.bias))
            return torch.cat((coupled[:, :1].expand(-1, 2), coupled[:, 1:2].expand(-1, 2), coupled[:, 2:]), -1)
        if self.mode in ("wheels", "wheel_memory"):
            end = 8 if self.mode == "wheel_memory" else 6
            wheels = torch.nn.functional.linear(velocity[:, 4:end], self.weight)
            return torch.cat((torch.zeros_like(velocity[:, :4]), wheels), -1)
        return torch.nn.functional.linear(velocity, self.weight)


class RobustGetupActor(UnifiedGetupActor):
    def __init__(self, parent, *, quiet_alpha=(.25, .25, .25, .25, .65, .65),
                 canonical_recovery=False, motion_residual_fraction=0.,
                 cyclic_leg_targets=False, leg_action_clip=13., recovery_leg_step=1.2,
                 smooth_knee_targets=False, recovery_knee_center=(.924544, .924537),
                 quiet_velocity_feature_scale=None, quiet_wheel_bound=.5, stance_aware_recovery=False,
                 recovery_shape_aware=False, quiet_high_common_alpha=None, quiet_high_leg_prediction_s=0.,
                 fast_forward_reference=False, fast_forward_reference_range=(1.5, 2.),
                 fast_forward_reference_yaw_fade=.25, quiet_height_experts=False,
                 quiet_native_height_anchors=False, quiet_velocity_feature_gain=1.,
                 quiet_velocity_feedback=False, quiet_velocity_feedback_mode="all"):
        super().__init__(parent)
        self.canonical_recovery = canonical_recovery
        self.motion_residual_fraction = motion_residual_fraction
        self.cyclic_leg_targets = cyclic_leg_targets
        self.recovery_leg_step = recovery_leg_step
        self.smooth_knee_targets = smooth_knee_targets
        self.recovery_knee_center = recovery_knee_center
        self.quiet_velocity_feature_scale = quiet_velocity_feature_scale
        if not 1. <= quiet_velocity_feature_gain <= 30.:
            raise ValueError("Quiet velocity feature gain must be within1...30")
        self.quiet_velocity_feature_gain = quiet_velocity_feature_gain
        self.stance_aware_recovery = stance_aware_recovery
        if recovery_shape_aware and not stance_aware_recovery:
            raise ValueError("Knee-shape recovery requires stance-aware authority")
        self.recovery_shape_aware = recovery_shape_aware
        if quiet_high_common_alpha is not None and not 0. < quiet_high_common_alpha <= 1.:
            raise ValueError("High-height common-angle interpolation must be in (0,1]")
        self.quiet_high_common_alpha = quiet_high_common_alpha
        if not 0. <= quiet_high_leg_prediction_s <= .01:
            raise ValueError("High-height leg prediction must be between 0 and 10 ms")
        self.quiet_high_leg_prediction_s = quiet_high_leg_prediction_s
        if fast_forward_reference and not isinstance(getattr(getattr(parent, "command", None), "base", None), nn.Module):
            raise ValueError("Fast-forward reference requires the authenticated original feedback inside the frozen parent")
        self.fast_forward_reference = fast_forward_reference
        if (len(fast_forward_reference_range) != 2
                or not 0. <= fast_forward_reference_range[0] < fast_forward_reference_range[1] <= 3.):
            raise ValueError("Forward reference transition must be ordered within 0..3 m/s")
        self.fast_forward_reference_range = tuple(fast_forward_reference_range)
        if fast_forward_reference_yaw_fade is not None and not 0. < fast_forward_reference_yaw_fade <= 4.:
            raise ValueError("Forward reference yaw fade must be in (0,4] or None")
        self.fast_forward_reference_yaw_fade = fast_forward_reference_yaw_fade
        self.quiet_height_experts = quiet_height_experts
        self.quiet_native_height_anchors = quiet_native_height_anchors
        if quiet_velocity_feedback and not quiet_height_experts:
            raise ValueError("Quiet velocity feedback requires existing three-height experts")
        if quiet_velocity_feedback_mode not in ("all", "wheels", "wheel_memory", "coupled_state"):
            raise ValueError("Quiet velocity feedback mode must be all, wheels, wheel_memory or coupled_state")
        if quiet_velocity_feedback_mode != "all" and not quiet_velocity_feedback:
            raise ValueError("Wheel-only velocity mode requires explicit quiet velocity feedback")
        self.quiet_velocity_feedback = quiet_velocity_feedback
        self.quiet_velocity_feedback_mode = quiet_velocity_feedback_mode
        self.action_bound[:4].fill_(leg_action_clip)
        self.quiet = nn.Sequential(nn.Linear(35, 64), nn.ELU(), nn.Linear(64, 6))
        nn.init.zeros_(self.quiet[-1].weight)
        nn.init.zeros_(self.quiet[-1].bias)
        if quiet_height_experts:
            self.quiet_low = deepcopy(self.quiet)
            self.quiet_high = deepcopy(self.quiet)
        if quiet_velocity_feedback:
            self.quiet_velocity = nn.ModuleDict({name: QuietVelocityFeedback(quiet_velocity_feedback_mode)
                                                for name in ("mid", "low", "high")})
        self.register_buffer("quiet_alpha", torch.tensor(quiet_alpha))
        self.register_buffer("quiet_bound", torch.tensor([.4] * 4 + [quiet_wheel_bound] * 2))

    def freeze_except_quiet_velocity(self):
        if not self.quiet_velocity_feedback:
            raise ValueError("Velocity-only learning requires migrated quiet velocity feedback")
        self.requires_grad_(False)
        self.quiet_velocity.requires_grad_(True)

    def quiet_correction(self, features, obs):
        mid_logits = self.quiet(features)
        coupled_state = self.quiet_velocity_feedback and self.quiet_velocity_feedback_mode == "coupled_state"
        if self.quiet_velocity_feedback:
            # Encoder velocity is scaled by .1 in the observation contract.
            # Legacy velocity modes are bias-free. The coupled-state mode also
            # sees pose and an affine offset, without a position integrator.
            velocity = torch.tanh(obs[:, 16:22] * 10.)
            if coupled_state:
                # Existing35D only: gyro, gravity, leg positions, six velocities.
                # No previous-action column or new recurrent state is read by
                # this path. A state-dependent bias can retain a DC correction.
                velocity = torch.cat((torch.tanh(obs[:, 4:7] * 2.), obs[:, 7:10],
                                      torch.tanh(obs[:, 10:14]), velocity), -1)
            elif self.quiet_velocity_feedback_mode == "wheel_memory":
                # The previous two wheel commands already occupy the last two
                # entries of the deployed previous-action block. They allow
                # learned command memory without a new observer or hidden state.
                velocity = torch.cat((velocity, torch.tanh(obs[:, 26:28] * 10.)), -1)
            if not coupled_state:
                mid_logits = mid_logits + self.quiet_velocity["mid"](velocity)
        mid = torch.tanh(mid_logits)
        if coupled_state:
            mid = mid + self.quiet_velocity["mid"](velocity)
        if not self.quiet_height_experts:
            return mid
        low_logits, high_logits = self.quiet_low(features), self.quiet_high(features)
        if self.quiet_velocity_feedback and not coupled_state:
            low_logits = low_logits + self.quiet_velocity["low"](velocity)
            high_logits = high_logits + self.quiet_velocity["high"](velocity)
        low, high = torch.tanh(low_logits), torch.tanh(high_logits)
        if coupled_state:
            low = low + self.quiet_velocity["low"](velocity)
            high = high + self.quiet_velocity["high"](velocity)
        # The existing height request is scaled by five. At the three anchor
        # heights only its own expert receives actor gradients. Intermediate
        # requests interpolate continuously; no hidden state or height FSM.
        h = obs[:, 3:4]
        if self.quiet_native_height_anchors:
            # Match the native observation's float32 command * 5 operation.
            # A rounded literal1.525 differs from float32(.305)*5 by one ULP.
            anchors = obs.new_tensor([.23, .305, .42]) * 5.
            lo = ((anchors[1] - h) / (anchors[1] - anchors[0])).clamp(0., 1.)
            hi = ((h - anchors[1]) / (anchors[2] - anchors[1])).clamp(0., 1.)
        else:
            lo = ((1.525 - h) / .375).clamp(0., 1.)
            hi = ((h - 1.525) / .575).clamp(0., 1.)
            lo = torch.where(h <= 1.15, torch.ones_like(lo), lo)
            hi = torch.where(h >= 2.1, torch.ones_like(hi), hi)
        result = mid + lo * (low - mid) + hi * (high - mid)
        return torch.where(lo >= 1., low, torch.where(hi >= 1., high, result))

    def forward(self, obs):
        features = torch.tanh(obs * self.feature_scale)
        gate = tilt_gate(obs, self.stance_aware_recovery, self.recovery_shape_aware)
        arc_gate = ((obs[:, :1].abs() - .05) / .2).clamp(0., 1.) * ((obs[:, 2:3].abs() - .05) / .2).clamp(0., 1.)
        moving = torch.maximum(obs[:, :1].abs() / .1, obs[:, 2:3].abs() / .2).clamp(0., 1.)
        residual_gate = arc_gate + self.motion_residual_fraction * (1. - arc_gate) * moving
        motion_obs = obs
        if self.quiet_high_leg_prediction_s:
            # q is radians and dq is scaled by .1 in the frozen observation
            # contract. This bounded, causal first-order estimate adds no state
            # or new sensor. It is confined to an upright high-stance request.
            prediction_gate = quiet_gate(obs) * ((obs[:, 3:4] / 5. - .35) / .05).clamp(0., 1.)
            predicted = obs[:, 10:14] + prediction_gate * (10. * self.quiet_high_leg_prediction_s) * obs[:, 16:20]
            predicted = torch.where(prediction_gate > 0., torch.atan2(predicted.sin(), predicted.cos()), obs[:, 10:14])
            motion_obs = torch.cat((obs[:, :10], predicted, obs[:, 14:]), -1)
        motion = self.parent(motion_obs) + residual_gate * self.arc_bound * torch.tanh(self.arc(features))
        if self.fast_forward_reference:
            reference_gate = fast_forward_reference_gate(obs, self.fast_forward_reference_range,
                                                         self.fast_forward_reference_yaw_fade)
            reference = self.parent.command.base(motion_obs)
            blended = motion + reference_gate * (reference - motion)
            # Preserve both endpoints exactly. In particular, high straight
            # commands use the original feedback function before the same clip.
            motion = torch.where(reference_gate >= 1., reference,
                                 torch.where(reference_gate > 0., blended, motion))
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
        if self.quiet_high_leg_prediction_s:
            quiet_features = torch.tanh(motion_obs * self.feature_scale)
        if self.quiet_velocity_feature_scale is not None:
            quiet_features = torch.cat((quiet_features[:, :16], torch.tanh(obs[:, 16:22] * self.quiet_velocity_feature_scale), quiet_features[:, 22:]), -1)
        if self.quiet_velocity_feature_gain != 1.:
            # An explicit migration inversely rescales the first-layer velocity
            # columns and transforms Adam moments. Initial feedback is retained;
            # only its learning coordinates change, without saturating a new tanh.
            quiet_features = torch.cat((quiet_features[:, :16],
                quiet_features[:, 16:22] * self.quiet_velocity_feature_gain, quiet_features[:, 22:]), -1)
        motion = motion + quiet * (smoothed - motion + self.quiet_bound * self.quiet_correction(quiet_features, obs))
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
        expected = {"quiet.0.weight", "quiet.0.bias", "quiet.2.weight", "quiet.2.bias", "quiet_alpha", "quiet_bound"}
        if self.quiet_height_experts:
            expected.update(key for key in self.state_dict() if key.startswith(("quiet_low.", "quiet_high.")))
        if self.quiet_velocity_feedback:
            expected.update(key for key in self.state_dict() if key.startswith("quiet_velocity."))
        if result.unexpected_keys or set(result.missing_keys) != expected:
            raise ValueError("Unexpected unified-to-robust actor migration")


class SensorConditionedMean(nn.Module):
    """RSL Gaussian input: actual mean and an observation-dependent std scale.

The distribution, stored log-probabilities and PPO all use the same std. Quiet
standing is not destroyed by the larger exploration needed while overturned.
"""
    def __init__(self, mean, quiet_exploration_fraction=.35, quiet_wheel_exploration_fraction=None):
        super().__init__()
        if not 0. < quiet_exploration_fraction <= 1.:
            raise ValueError("Quiet exploration fraction must be finite and in (0,1]")
        self.mean = mean
        self.quiet_exploration_fraction = quiet_exploration_fraction
        if quiet_wheel_exploration_fraction is not None and not 0. < quiet_wheel_exploration_fraction <= 1.:
            raise ValueError("Quiet wheel exploration fraction must be finite and in (0,1]")
        self.quiet_wheel_exploration_fraction = quiet_wheel_exploration_fraction
        self.log_std = nn.Parameter(torch.full((6,), -.6931471805599453))

    def forward(self, obs):
        recovery = tilt_gate(obs, self.mean.stance_aware_recovery, self.mean.recovery_shape_aware)
        scale = .12 + .88 * recovery
        fraction = self.quiet_exploration_fraction
        if self.quiet_wheel_exploration_fraction is not None:
            fraction = obs.new_tensor([fraction] * 4 + [self.quiet_wheel_exploration_fraction] * 2)
        scale = scale * (1. - (1. - fraction) * quiet_gate(obs))
        log_std = self.log_std.clamp(-3., -.22314355) + scale.log()
        return torch.stack((self.mean(obs), log_std.expand(-1, 6)), -2)

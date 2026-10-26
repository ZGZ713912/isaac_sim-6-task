"""Retain quiet feedback during upright capture, outside recovery authority.

The four static forward bodies preserve the operation order of authenticated
ancestors. Keeping those source files unchanged preserves historical checkpoint
verification. Only the policy quiet gate changes; shared observer and sampling
functions retain their original definitions. This actor is a pure 35D->6D map.
"""
from copy import deepcopy
import math

import torch

from .gap_entry_handoff import handoff_entry_gate
from .getup_symmetry import canonical_recovery_logits
from .motion_target_damping import MotionTargetDampedGapActor, damp_motion_targets, motion_damping_gate
from .paired_recovery import paired_offsets, wrap
from .periodic_entry_actor import exact_state_equal, periodic_canonical_entry_logits, periodic_entry_logits
from .recovery_coordinates import smooth_recovery_legs
from .robust_getup import fast_forward_reference_gate, quiet_gate, tilt_gate

VERSION = "quiet-stance-authority-v1"
SCHEMA = "robust-getup-checkpoint-v4"
MODULE = "src/wheeled_algo/quiet_stance_authority.py"


def validate_floor(floor):
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or not 0. <= floor <= 1.:
        raise ValueError("Retained quiet authority must be finite within [0,1]")


def stance_quiet_gate(obs, floor):
    base = quiet_gate(obs)
    if floor == 0.:
        return base
    request = torch.maximum(obs[:, :1].abs()/.12, obs[:, 2:3].abs()/.25)
    tilt = ((-obs[:, 9:10]-.94)/.04).clamp(0., 1.)
    gyro = (obs[:, 4:6].norm(dim=-1, keepdim=True)*2.-.4).clamp(0., 1.)
    recovery = torch.maximum(handoff_entry_gate(obs), tilt_gate(obs, True, True))
    stance = (1.-recovery/.05).clamp(0., 1.)
    added = floor*(1.-request).clamp(0., 1.)*tilt*gyro*stance
    return torch.where(added > 0., base+added, base)


class QuietStanceAuthorityActor(MotionTargetDampedGapActor):
    def quiet_stance_gate(self, obs):
        return stance_quiet_gate(obs, self.quiet_stance_retained_authority)

    def _robust_forward(self, obs):
        standing_position = None
        if self.quiet_velocity_feedback_mode == "standing_displacement":
            standing_position = obs[:, 14:16]
            # Every previously trained network, gate and sensor feature keeps
            # the original zero-wheel-position contract, including recovery.
            obs = torch.cat((obs[:, :14], torch.zeros_like(obs[:, 14:16]), obs[:, 16:]), -1)
        features = torch.tanh(obs * self.feature_scale)
        gate = tilt_gate(obs, self.stance_aware_recovery, self.recovery_shape_aware)
        arc_gate = ((obs[:, :1].abs() - .05) / .2).clamp(0., 1.) * ((obs[:, 2:3].abs() - .05) / .2).clamp(0., 1.)
        moving = torch.maximum(obs[:, :1].abs() / .1, obs[:, 2:3].abs() / .2).clamp(0., 1.)
        residual_gate = arc_gate + self.motion_residual_fraction * (1. - arc_gate) * moving
        motion_obs = obs
        if self.quiet_high_leg_prediction_s:
            # q is radians and dq is scaled by .1 in the frozen observation
            # contract. This bounded, causal first-order estimate adds no state
            # or new sensor. Old checkpoints retain the high-stance scope.
            prediction_gate = self.quiet_stance_gate(obs)
            if not self.quiet_leg_prediction_all_heights:
                prediction_gate = prediction_gate * ((obs[:, 3:4] / 5. - .35) / .05).clamp(0., 1.)
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
        quiet = self.quiet_stance_gate(obs)
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
        motion = motion + quiet * (smoothed - motion + self.quiet_bound * self.quiet_correction(quiet_features, obs, standing_position))
        raw = (canonical_recovery_logits(self.recovery, obs, self.feature_scale)
               if self.canonical_recovery else self.recovery(features))
        legs = obs[:, 22:26] + self.recovery_leg_step * torch.tanh(raw[:, :4])
        if self.smooth_knee_targets:
            legs = smooth_recovery_legs(raw[:, :4], obs[:, 22:26], obs[:, 10:14],
                                        self.recovery_leg_step, self.recovery_knee_center,
                                        knee_step=self.recovery_knee_step,
                                        previous_common_version=self.recovery_previous_common_version)
        elif self.cyclic_leg_targets:
            legs = paired_offsets(legs * .25, obs[:, 10:14]) / .25
        else:
            legs = legs.clamp(-13., 13.)
        fallen = torch.cat((legs, 9. * torch.tanh(raw[:, 4:6])), -1)
        result = motion + gate * (fallen - motion)
        return torch.maximum(torch.minimum(result, self.action_bound), -self.action_bound)

    def _quiet_forward(self, obs):
        action = self._robust_forward(obs)
        if self.quiet_output_wheel_alpha == 1.:
            return action
        quiet = self.quiet_stance_gate(obs)
        wheel = action[:, 4:]
        filtered = obs[:, 26:28] + self.quiet_output_wheel_alpha * (wheel - obs[:, 26:28])
        blended = wheel + quiet * (filtered - wheel)
        blended = torch.where(quiet >= 1., filtered, torch.where(quiet > 0., blended, wheel))
        return torch.cat((action[:, :4], blended.clamp(-9., 9.)), -1)

    def _periodic_forward(self, obs):
        old = self._quiet_forward(obs)
        gate = handoff_entry_gate(obs)
        frame = torch.cat((obs[:, :14], torch.zeros_like(obs[:, 14:16]), obs[:, 16:]), -1)
        raw = (periodic_canonical_entry_logits(self.entry_recovery, frame, self.feature_scale)
               if self.canonical_recovery else periodic_entry_logits(self.entry_recovery, frame, self.feature_scale))
        legs = smooth_recovery_legs(raw[:, :4], obs[:, 22:26], obs[:, 10:14],
                                   self.recovery_leg_step, self.recovery_knee_center,
                                   previous_common_version=self.recovery_previous_common_version)
        delta = wrap(legs*.25-obs[:, 10:14])
        legs = (obs[:, 10:14]+.12*torch.tanh(delta/.12))/.25
        fallen = torch.cat((legs, 9.*torch.tanh(raw[:, 4:6])), -1)
        blended = old+gate*(fallen-old)
        result = torch.where(gate > 0., blended, old)
        result = torch.maximum(torch.minimum(result, self.action_bound), -self.action_bound)
        delta = wrap(result[:, :4]*.25-obs[:, 10:14])
        magnitude = delta.abs()
        shaped = delta.sign()*(.06+.06*torch.tanh((magnitude-.06)/.06))
        upright = ((-obs[:, 9:10]-.5)/.2).clamp(0., 1.)
        blended = delta+upright*(shaped-delta)
        changed = (upright > 0.) & (magnitude > .06)
        legs = torch.where(changed, (obs[:, 10:14]+blended)/.25, result[:, :4])
        legs = torch.maximum(torch.minimum(legs, self.action_bound[:4]), -self.action_bound[:4])
        legs = self.finalize_leg_pairs(legs)
        return torch.cat((legs, result[:, 4:]), -1)

    def forward(self,obs):
        old = self._periodic_forward(obs)
        if self.motion_leg_alpha==1.:
            return old
        result = damp_motion_targets(old,obs,self.motion_leg_alpha,self.action_bound,
                                     self.stance_aware_recovery,self.recovery_shape_aware)
        gate = motion_damping_gate(obs,self.stance_aware_recovery,self.recovery_shape_aware)
        legs = self.finalize_leg_pairs(result[:,:4])
        return torch.cat((torch.where(gate>0.,legs,old[:,:4]),old[:,4:]),-1)


def attach_stance_authority(mean, floor):
    """Reuse authenticated modules and exact namespaces without adding state."""
    validate_floor(floor)
    if (type(mean) is not MotionTargetDampedGapActor or not mean.stance_aware_recovery
            or not mean.recovery_shape_aware):
        raise ValueError("Quiet stance authority requires the authenticated shape-aware motion actor")
    mean.__class__ = QuietStanceAuthorityActor
    mean.quiet_stance_retained_authority = float(floor)
    return mean


def validate_checkpoint(state, source):
    """Zero PPO migration: every payload tensor and unrelated field is exact."""
    info, prior = state['infos'], source['infos']
    floor = info['settings'].get('quiet_stance_retained_authority')
    validate_floor(floor)
    record = info.get('quiet_stance_authority_migration', {})
    if (info.get('schema') != SCHEMA or prior.get('schema') != 'robust-getup-checkpoint-v2'
            or info['settings'].get('quiet_stance_authority_version') != VERSION
            or info['settings'].get('quiet_stance_recovery_threshold') != .05
            or record.get('schema') != VERSION or record.get('new_PPO') != 0
            or record.get('learning_clock') != info['completed_updates']
            or info['completed_updates'] != prior['completed_updates']
            or 'quiet_stance_authority_version' in prior['settings']):
        raise ValueError("Quiet authority requires an explicit zero-update source migration")
    if not exact_state_equal({k:v for k,v in state.items() if k != 'infos'},
                             {k:v for k,v in source.items() if k != 'infos'}):
        raise ValueError("Quiet authority migration changed weights, optimizer or saved state")
    expected = deepcopy(prior)
    expected['schema'] = SCHEMA
    expected['settings'].update(quiet_stance_authority_version=VERSION,
        quiet_stance_retained_authority=floor, quiet_stance_recovery_threshold=.05)
    expected['source_sha256'][MODULE] = info['source_sha256'].get(MODULE)
    expected['quiet_stance_authority_migration'] = record
    expected['hardware_ready'] = False
    if not exact_state_equal(info, expected):
        raise ValueError("Quiet authority migration changed unrelated checkpoint metadata")

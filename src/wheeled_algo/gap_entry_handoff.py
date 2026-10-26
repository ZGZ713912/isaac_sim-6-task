"""Opt-in extension of floor-entry authority through the measured leg orbit."""
import torch

from .gap_robust_actor import GapRobustGetupActor
from .quiet_output_actor import QuietOutputGetupActor
from .robust_getup import SensorConditionedMean, quiet_gate, tilt_gate
from .getup_symmetry import canonical_recovery_logits
from .recovery_coordinates import smooth_recovery_legs
from .paired_recovery import wrap


def handoff_entry_gate(obs):
    pair = obs[:, 10:14].unflatten(-1, (2, 2))
    common = torch.atan2(pair.sin().sum(-1), pair.cos().sum(-1)).abs().amax(-1, keepdim=True)
    phase = ((common-.30)/.35).clamp(0., 1.)
    upright = ((-obs[:, 9:10]-.7)/.2).clamp(0., 1.)
    return phase*upright


class HandoffGapGetupActor(GapRobustGetupActor):
    def forward(self, obs):
        old = QuietOutputGetupActor.forward(self, obs)
        if not self.upright_entry_recovery:
            return old
        gate = handoff_entry_gate(obs)
        frame = torch.cat((obs[:, :14], torch.zeros_like(obs[:, 14:16]), obs[:, 16:]), -1)
        raw = (canonical_recovery_logits(self.recovery, frame, self.feature_scale)
               if self.canonical_recovery else self.recovery(torch.tanh(frame*self.feature_scale)))
        legs = smooth_recovery_legs(raw[:, :4], obs[:, 22:26], obs[:, 10:14],
                                    self.recovery_leg_step, self.recovery_knee_center,
                                    previous_common_version=self.recovery_previous_common_version)
        delta = wrap(legs*.25-obs[:, 10:14])
        legs = (obs[:, 10:14]+.12*torch.tanh(delta/.12))/.25
        fallen = torch.cat((legs, 9.*torch.tanh(raw[:, 4:6])), -1)
        blended = old+gate*(fallen-old)
        result = torch.where(gate > 0., blended, old)
        return torch.maximum(torch.minimum(result, self.action_bound), -self.action_bound)


class HandoffGapConditionedMean(SensorConditionedMean):
    def forward(self, obs):
        entry = handoff_entry_gate(obs) if self.mean.upright_entry_recovery else torch.zeros_like(obs[:, :1])
        recovery = torch.maximum(tilt_gate(obs, self.mean.stance_aware_recovery, self.mean.recovery_shape_aware), entry)
        scale = .12+.88*recovery
        fraction = self.quiet_exploration_fraction
        if self.quiet_wheel_exploration_fraction is not None:
            fraction = obs.new_tensor([fraction]*4+[self.quiet_wheel_exploration_fraction]*2)
        scale = scale*(1.-(1.-fraction)*quiet_gate(obs)*(1.-entry))
        scale = scale*(1.-entry+entry*obs.new_tensor([.2, .2, .2, .2, 1., 1.]))
        log_std = self.log_std.clamp(-3., -.22314355)+scale.log()
        return torch.stack((self.mean(obs), log_std.expand(-1, 6)), -2)

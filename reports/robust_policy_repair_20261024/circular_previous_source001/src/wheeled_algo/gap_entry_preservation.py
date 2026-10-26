"""A frozen RL entry head beside the trainable fallen head; existing gates only."""
from copy import deepcopy

import torch

from .gap_entry_handoff import handoff_entry_gate
from .getup_symmetry import canonical_recovery_logits
from .paired_recovery import wrap
from .quiet_output_actor import QuietOutputGetupActor
from .recovery_coordinates import smooth_recovery_legs
from .upright_target_limit import UprightTargetLimitedGapActor

VERSION = "frozen-v1"


class EntryPreservingGapActor(UprightTargetLimitedGapActor):
    """Keep the measured orbit-v2 entry function while learning fallen actions.

    The copied head is a learned sensor-feedback network. This adds no phase
    state, timer, trajectory, contact truth, or new actuator authority.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.upright_entry_recovery:
            raise ValueError("Entry preservation requires the enabled orbit-v2 entry")
        self.entry_recovery = deepcopy(self.recovery).requires_grad_(False)

    def initialize_from_unified(self, state):
        # The established initializer validates its precise missing-key set.
        # Supply only this frozen addition, then copy the initialized head.
        additional = {k: v for k, v in self.state_dict().items() if k.startswith("entry_recovery.")}
        super().initialize_from_unified(dict(state, **additional))
        self.entry_recovery.load_state_dict(self.recovery.state_dict(), strict=True)

    def forward(self, obs):
        # Keep the old fallen/quiet/motion path and its exact arithmetic.
        old = QuietOutputGetupActor.forward(self, obs)
        gate = handoff_entry_gate(obs)
        frame = torch.cat((obs[:, :14], torch.zeros_like(obs[:, 14:16]), obs[:, 16:]), -1)
        raw = (canonical_recovery_logits(self.entry_recovery, frame, self.feature_scale)
               if self.canonical_recovery else self.entry_recovery(torch.tanh(frame*self.feature_scale)))
        legs = smooth_recovery_legs(raw[:, :4], obs[:, 22:26], obs[:, 10:14],
                                   self.recovery_leg_step, self.recovery_knee_center)
        delta = wrap(legs*.25-obs[:, 10:14])
        legs = (obs[:, 10:14]+.12*torch.tanh(delta/.12))/.25
        fallen = torch.cat((legs, 9.*torch.tanh(raw[:, 4:6])), -1)
        blended = old+gate*(fallen-old)
        result = torch.where(gate > 0., blended, old)
        result = torch.maximum(torch.minimum(result, self.action_bound), -self.action_bound)
        # This is exactly the existing error-v1 final target limiter.
        delta = wrap(result[:, :4]*.25-obs[:, 10:14])
        magnitude = delta.abs()
        shaped = delta.sign()*(.06+.06*torch.tanh((magnitude-.06)/.06))
        upright = ((-obs[:, 9:10]-.5)/.2).clamp(0., 1.)
        blended = delta+upright*(shaped-delta)
        changed = (upright > 0.) & (magnitude > .06)
        legs = torch.where(changed,(obs[:, 10:14]+blended)/.25,result[:, :4])
        legs = torch.maximum(torch.minimum(legs,self.action_bound[:4]),-self.action_bound[:4])
        return torch.cat((legs,result[:, 4:]),-1)

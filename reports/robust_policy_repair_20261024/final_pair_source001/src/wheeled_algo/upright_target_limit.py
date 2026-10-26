"""Limit final upright position error after every learned actor branch blends."""
import torch

from .gap_entry_handoff import HandoffGapGetupActor
from .paired_recovery import wrap


class UprightTargetLimitedGapActor(HandoffGapGetupActor):
    def forward(self, obs):
        action = super().forward(obs)
        delta = wrap(action[:, :4]*.25-obs[:, 10:14])
        magnitude = delta.abs()
        # Preserve small requests exactly and join with unit derivative.
        # The asymptote is .12 rad; no particular posture or path is prescribed.
        shaped = delta.sign()*(.06+.06*torch.tanh((magnitude-.06)/.06))
        upright = ((-obs[:, 9:10]-.5)/.2).clamp(0., 1.)
        blended = delta+upright*(shaped-delta)
        changed = (upright > 0.) & (magnitude > .06)
        legs = torch.where(changed,(obs[:, 10:14]+blended)/.25,action[:, :4])
        legs = torch.maximum(torch.minimum(legs,self.action_bound[:4]),-self.action_bound[:4])
        return torch.cat((legs,action[:, 4:]),-1)

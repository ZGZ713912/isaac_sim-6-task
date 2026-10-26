"""Limit final upright position error after every learned actor branch blends."""
import math
import torch

from .gap_entry_handoff import HandoffGapGetupActor
from .paired_recovery import DELTA_BOUNDS, NOMINAL_DELTA, wrap


def rebase_final_leg_pairs(legs):
    """Represent each circular target pair on one common winding.

    The downstream paired_branch107_v1 projection takes an arithmetic mean.
    Independently wrapped motor targets must therefore be unwrapped together
    before that projection. Non-crossing targets retain their exact bits.
    """
    pair = (.25 * legs).unflatten(-1, (2, 2))
    difference = pair[..., 1] - pair[..., 0]
    short = wrap(difference)
    common = wrap(pair[..., 0] + .5 * short)
    continuous = torch.stack((common - .5 * short, common + .5 * short), -1)
    return torch.where((difference.abs() > torch.pi)[..., None],
                       continuous / .25, legs.unflatten(-1, (2, 2))).flatten(-2)


class UprightTargetLimitedGapActor(HandoffGapGetupActor):
    def __init__(self, *args, final_target_winding_version=None, **kwargs):
        super().__init__(*args, **kwargs)
        if final_target_winding_version not in (None, "paired-v1"):
            raise ValueError("Unknown final target winding version")
        radius = math.pi + .5 * max(abs(sign * bound - nominal)
            for sign, nominal, bounds in zip((1., -1.), NOMINAL_DELTA, DELTA_BOUNDS)
            for bound in bounds)
        if final_target_winding_version and (not self.cyclic_leg_targets
                or float(self.action_bound[:4].min()) < radius / .25):
            raise ValueError("Final target winding requires cyclic targets and bounds covering every feasible paired angle")
        self.final_target_winding_version = final_target_winding_version

    def finalize_leg_pairs(self, legs):
        if self.final_target_winding_version == "paired-v1":
            legs = rebase_final_leg_pairs(legs)
            legs = torch.maximum(torch.minimum(legs, self.action_bound[:4]), -self.action_bound[:4])
        return legs

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
        legs = self.finalize_leg_pairs(legs)
        return torch.cat((legs,action[:, 4:]),-1)

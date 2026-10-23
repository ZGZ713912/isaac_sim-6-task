"""Read-only prototype for the next zero-PPO causal native test.

No constructor, deployment loader or native environment uses this file. The
convex reblend requires the inspected16-rad-raw paired target contract.
"""
import math

import torch
from torch import nn

from wheeled_algo.getup_symmetry import canonical_recovery_logits
from wheeled_algo.recovery_coordinates import smooth_recovery_legs
from wheeled_algo.robust_getup import quiet_gate, tilt_gate


def roll_gate(mean, obs, start_degrees, full_degrees):
    baseline = tilt_gate(obs, mean.stance_aware_recovery, mean.recovery_shape_aware)
    gravity = obs[:, 7:10]
    norm = gravity.norm(dim=-1, keepdim=True)
    valid = (norm >= .5) & (norm <= 1.5) & (obs[:, 9:10] >= -.94)
    lateral = gravity[:, 1:2].abs() / norm.clamp_min(1e-6)
    start, full = math.sin(math.radians(start_degrees)), math.sin(math.radians(full_degrees))
    roll = ((lateral - start) / (full - start)).clamp(0., 1.)
    roll = torch.where(valid, roll, torch.zeros_like(roll))
    return baseline, torch.maximum(baseline, roll)


class EarlyRollPrototype(nn.Module):
    def __init__(self, mean, start_degrees, full_degrees):
        super().__init__()
        if not 20. <= start_degrees < full_degrees <= 60.:
            raise ValueError("Roll authority range must be ordered within20..60 degrees")
        if (not mean.stance_aware_recovery or not mean.recovery_shape_aware
                or not mean.smooth_knee_targets or not mean.cyclic_leg_targets
                or not mean.canonical_recovery or not bool((mean.action_bound[:4] >= 14.2).all())):
            raise ValueError("Prototype requires inspected canonical paired smooth-knee target and>=14.2 raw leg bounds")
        self.mean = mean
        self.start_degrees, self.full_degrees = start_degrees, full_degrees

    def forward(self, obs):
        action = self.mean(obs)
        original, proposed = roll_gate(self.mean, obs, self.start_degrees, self.full_degrees)
        legacy = torch.cat((obs[:, :14], torch.zeros_like(obs[:, 14:16]), obs[:, 16:]), -1)
        raw = canonical_recovery_logits(self.mean.recovery, legacy, self.mean.feature_scale)
        legs = smooth_recovery_legs(raw[:, :4], legacy[:, 22:26], legacy[:, 10:14],
                                    self.mean.recovery_leg_step, self.mean.recovery_knee_center)
        fallen = torch.cat((legs, 9. * torch.tanh(raw[:, 4:6])), -1)
        fraction = (proposed - original) / (1. - original).clamp_min(1e-6)
        blended = action + fraction * (fallen - action)
        blended = torch.where(proposed >= 1., fallen, blended)
        result = torch.where(proposed > original, blended, action)
        return torch.maximum(torch.minimum(result, self.mean.action_bound), -self.mean.action_bound)

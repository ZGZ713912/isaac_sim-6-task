"""CPU-only residual prototype. No native loader or training imports this file."""
from copy import deepcopy

import torch
from torch import nn

from wheeled_algo.getup_symmetry import canonical_recovery_logits
from wheeled_algo.recovery_coordinates import smooth_recovery_legs
from wheeled_algo.robust_getup import tilt_gate


class AsymmetricRecoveryPrototype(nn.Module):
    def __init__(self, mean):
        super().__init__()
        if not (mean.canonical_recovery and mean.smooth_knee_targets and mean.cyclic_leg_targets):
            raise ValueError("Requires the inspected canonical paired recovery decoder")
        self.mean = deepcopy(mean).requires_grad_(False)
        self.residual = nn.Sequential(nn.Linear(35, 32), nn.ELU(), nn.Linear(32, 6))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer("logit_bound", torch.tensor([.25] * 4 + [.03] * 2))

    def forward(self, obs):
        action = self.mean(obs)
        legacy = torch.cat((obs[:, :14], torch.zeros_like(obs[:, 14:16]), obs[:, 16:]), -1)
        recovery = tilt_gate(legacy, self.mean.stance_aware_recovery, self.mean.recovery_shape_aware)
        side = ((-legacy[:, 8:9] - .15) / .35).clamp(0., 1.)
        raw = canonical_recovery_logits(self.mean.recovery, legacy, self.mean.feature_scale)
        delta = side * self.logit_bound * self.residual(torch.tanh(legacy * self.mean.feature_scale)).tanh()

        def fallen(logits):
            legs = smooth_recovery_legs(logits[:, :4], legacy[:, 22:26], legacy[:, 10:14],
                self.mean.recovery_leg_step, self.mean.recovery_knee_center)
            return torch.cat((legs, 9. * torch.tanh(logits[:, 4:6])), -1)

        result = action + recovery * (fallen(raw + delta) - fallen(raw))
        result = torch.where((side > 0.) & (recovery > 0.), result, action)
        return torch.maximum(torch.minimum(result, self.mean.action_bound), -self.mean.action_bound)

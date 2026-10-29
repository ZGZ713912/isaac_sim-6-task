"""Stateless zero-PPO diagnostic of action recurrence at the 200 Hz clock."""
import torch
from torch import nn


VARIANTS = {
    'all': [.25] * 6,
    'control': [1.] * 6,
    'legs': [.25] * 4 + [1.] * 2,
    'wheels': [1.] * 4 + [.25] * 2,
}


class ActionRelaxationProbe(nn.Module):
    def __init__(self, mean, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Require one of the four frozen diagnostic variants')
        self.mean = mean
        self.variant = variant
        self.register_buffer('alpha', torch.tensor(VARIANTS[variant]))

    def forward(self, obs):
        proposed = self.mean(obs)
        # SCUT35 policy-order previous applied actions are entries 22:28.
        # This uses no new sensor, clock, privileged velocity or hidden state.
        previous = obs[:, 22:28]
        mixed = previous + self.alpha * (proposed - previous)
        return torch.where(self.alpha == 1., proposed, mixed)

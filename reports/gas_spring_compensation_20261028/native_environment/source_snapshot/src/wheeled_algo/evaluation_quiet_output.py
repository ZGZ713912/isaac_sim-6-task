"""Explicit evaluation counterfactual; no learned weights or hidden history."""
import math

import torch
from torch import nn

from .robust_getup import quiet_gate


class QuietWheelOutputFilter(nn.Module):
    """Filter the whole standing wheel target using the applied previous action.

    Every fixed point of the original quiet policy is retained. This is not a
    stability proof: additional feedback delay needs a matched native test.
    """

    def __init__(self, mean, alpha):
        super().__init__()
        if isinstance(alpha, bool) or not math.isfinite(alpha) or not .05 <= alpha <= 1.:
            raise ValueError("Quiet wheel output alpha must be finite within[.05,1]")
        self.mean = mean
        self.alpha = float(alpha)

    def forward(self, obs):
        action = self.mean(obs)
        if self.alpha == 1.:
            return action
        quiet = quiet_gate(obs)
        wheel = action[:, 4:]
        filtered = obs[:, 26:28] + self.alpha * (wheel - obs[:, 26:28])
        blended = wheel + quiet * (filtered - wheel)
        blended = torch.where(quiet >= 1., filtered, torch.where(quiet > 0., blended, wheel))
        return torch.cat((action[:, :4], blended.clamp(-9., 9.)), -1)

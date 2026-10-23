"""Opt-in complete quiet wheel output filtering, preserving actor namespaces."""
import math

import torch

from .robust_getup import RobustGetupActor, quiet_gate


class QuietOutputGetupActor(RobustGetupActor):
    def __init__(self, parent, *, quiet_output_wheel_alpha, **kwargs):
        if (isinstance(quiet_output_wheel_alpha, bool)
                or not math.isfinite(quiet_output_wheel_alpha)
                or not .05 <= quiet_output_wheel_alpha <= 1.):
            raise ValueError("Complete quiet wheel alpha must be finite within[.05,1]")
        super().__init__(parent, **kwargs)
        self.quiet_output_wheel_alpha = float(quiet_output_wheel_alpha)

    def forward(self, obs):
        action = super().forward(obs)
        if self.quiet_output_wheel_alpha == 1.:
            return action
        quiet = quiet_gate(obs)
        wheel = action[:, 4:]
        filtered = obs[:, 26:28] + self.quiet_output_wheel_alpha * (wheel - obs[:, 26:28])
        blended = wheel + quiet * (filtered - wheel)
        blended = torch.where(quiet >= 1., filtered, torch.where(quiet > 0., blended, wheel))
        return torch.cat((action[:, :4], blended.clamp(-9., 9.)), -1)

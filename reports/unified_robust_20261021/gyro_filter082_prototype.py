"""CPU prototype only: release extra quiet wheel delay during balance transients."""
import math

import torch
from torch import nn

from wheeled_algo.robust_getup import quiet_gate


class GyroReleasedWheelFilter(nn.Module):
    def __init__(self, mean, start, full):
        super().__init__()
        if not (math.isfinite(start) and math.isfinite(full) and 0. < start < full <= .4):
            raise ValueError("Gyro-release thresholds must satisfy0<start<full<=.4rad/s")
        if getattr(mean, "quiet_output_wheel_alpha", 1.) != 1.:
            raise ValueError("Prototype must not double-filter an existing quiet output")
        self.mean, self.start, self.full = mean, float(start), float(full)

    def forward(self, obs):
        action = self.mean(obs)
        speed = obs[:, 4:6].norm(dim=-1, keepdim=True) * 2.
        alpha = .25 + .75 * ((speed - self.start) / (self.full - self.start)).clamp(0., 1.)
        wheel = action[:, 4:]
        filtered = obs[:, 26:28] + alpha * (wheel - obs[:, 26:28])
        filtered = torch.where(alpha >= 1., wheel, filtered)
        quiet = quiet_gate(obs)
        blended = wheel + quiet * (filtered - wheel)
        result = torch.where(quiet >= 1., filtered, torch.where(quiet > 0., blended, wheel))
        return torch.cat((action[:, :4], result.clamp(-9., 9.)), -1)

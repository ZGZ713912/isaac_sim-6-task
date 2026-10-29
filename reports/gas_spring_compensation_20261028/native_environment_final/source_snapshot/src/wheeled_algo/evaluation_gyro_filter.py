"""Evaluation-only release of extra quiet wheel delay during balance transients."""
import math

import torch
from torch import nn

from .robust_getup import quiet_gate


def gyro_filter_variants(values):
    if (len(values) not in (2, 4)
            or any(isinstance(value, bool) or not math.isfinite(value) for value in values)):
        raise ValueError("Use one or two finite gyro-release bands in rad/s")
    bands = list(zip(values[::2], values[1::2]))
    if len(set(bands)) != len(bands) or any(not 0. < lo < hi <= .4 for lo, hi in bands):
        raise ValueError("Distinct gyro bands must satisfy0<start<full<=.4rad/s")
    return {"quiet_filter025": None, **{f"quiet_gyro_{lo:.17g}_{hi:.17g}": [lo, hi]
                                     for lo, hi in bands}}


class GyroReleasedWheelFilter(nn.Module):
    """Keep alpha.25 at low roll/pitch rates and reach alpha1 at the upper rate.

    Uses existing scaled gyro slots4:6 and previous applied action slots26:28.
    This adds no state and does not establish closed-loop balance or HF benefit.
    """
    def __init__(self, mean, band):
        super().__init__()
        if band is not None:
            if len(band) != 2:
                raise ValueError("A gyro filter takes exactly one release band")
            gyro_filter_variants(band)
        if getattr(mean, "quiet_output_wheel_alpha", 1.) != 1.:
            raise ValueError("Gyro diagnostic requires a mean without extra output filtering")
        self.mean = mean
        self.band = tuple(band) if band is not None else None

    def forward(self, obs):
        action = self.mean(obs)
        wheel = action[:, 4:]
        alpha = .25
        if self.band is not None:
            start, full = self.band
            speed = obs[:, 4:6].norm(dim=-1, keepdim=True) * 2.
            alpha = .25 + .75 * ((speed - start) / (full - start)).clamp(0., 1.)
        filtered = obs[:, 26:28] + alpha * (wheel - obs[:, 26:28])
        if self.band is not None:
            filtered = torch.where(alpha >= 1., wheel, filtered)
        quiet = quiet_gate(obs)
        blended = wheel + quiet * (filtered - wheel)
        result = torch.where(quiet >= 1., filtered, torch.where(quiet > 0., blended, wheel))
        return torch.cat((action[:, :4], result.clamp(-9., 9.)), -1)

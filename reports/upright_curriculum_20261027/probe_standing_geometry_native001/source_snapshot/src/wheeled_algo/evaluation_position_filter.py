"""Evaluation-only release of extra wheel filtering for larger position errors."""
import math

import torch
from torch import nn

from .robust_getup import quiet_gate


def position_filter_variants(values):
    if (len(values) not in (2, 4)
            or any(isinstance(value, bool) or not math.isfinite(value) for value in values)):
        raise ValueError("Use one or two finite position-release bands in metres")
    bands = list(zip(values[::2], values[1::2]))
    if len(set(bands)) != len(bands) or any(not 0. <= lo < hi <= .05 for lo, hi in bands):
        raise ValueError("Distinct release bands must satisfy0<=lo<hi<=.05m")
    return {"quiet_filter025": None, **{f"quiet_release_{lo:.17g}_{hi:.17g}": [lo, hi]
                                     for lo, hi in bands}}


class PositionReleasedWheelFilter(nn.Module):
    """Keep alpha.25 in the small position band and reach alpha1 at its end.

    Only existing encoder-derived observation slots14:16 are read. This adds
    no integrator or clock, and does not establish closed-loop stability.
    """

    def __init__(self, mean, band):
        super().__init__()
        if band is not None:
            position_filter_variants(band)
        if getattr(mean, "quiet_output_wheel_alpha", 1.) != 1.:
            raise ValueError("Position diagnostic requires a mean without extra output filtering")
        if getattr(mean, "quiet_velocity_feedback_mode", None) != "standing_displacement":
            raise ValueError("Position diagnostic requires declared standing-displacement observations")
        self.mean = mean
        self.band = tuple(band) if band is not None else None

    def forward(self, obs):
        action = self.mean(obs)
        wheel = action[:, 4:]
        alpha = .25
        if self.band is not None:
            lo, hi = self.band
            error = obs[:, 14:16].abs().amax(-1, keepdim=True) / 20.
            alpha = .25 + .75 * ((error - lo) / (hi - lo)).clamp(0., 1.)
        filtered = obs[:, 26:28] + alpha * (wheel - obs[:, 26:28])
        if self.band is not None:
            filtered = torch.where(alpha >= 1., wheel, filtered)
        quiet = quiet_gate(obs)
        blended = wheel + quiet * (filtered - wheel)
        result = torch.where(quiet >= 1., filtered, torch.where(quiet > 0., blended, wheel))
        return torch.cat((action[:, :4], result.clamp(-9., 9.)), -1)

"""Explicit zero-PPO test of smaller learned recovery common-angle increments."""
from copy import deepcopy
import math

from torch import nn


def recovery_step_variants(values):
    if (not 1 <= len(values) <= 2 or len(set(values)) != len(values)
            or any(isinstance(value, bool) or not math.isfinite(value)
                   or not .05 <= value < .3 for value in values)):
        raise ValueError("Recovery step grid requires1..2 unique finite alternatives within[.05,.3); original.3 is included")
    return {"robust": .3, **{f"recovery_step_{value:.17g}": value for value in values}}


class RecoveryStepEvaluation(nn.Module):
    def __init__(self, mean, step):
        super().__init__()
        if (isinstance(step, bool) or not math.isfinite(step) or not .05 <= step <= .3):
            raise ValueError("Recovery step must be finite within[.05,.3]")
        if (not getattr(mean, "smooth_knee_targets", False)
                or not getattr(mean, "cyclic_leg_targets", False)
                or not getattr(mean, "canonical_recovery", False)
                or not getattr(mean, "stance_aware_recovery", False)
                or not getattr(mean, "recovery_shape_aware", False)
                or getattr(mean, "recovery_leg_step", None) != .3):
            raise ValueError("Recovery step diagnostic requires the inspected canonical paired smooth-knee actor with original step.3")
        self.mean = deepcopy(mean)
        self.mean.recovery_leg_step = step

    def forward(self, obs):
        return self.mean(obs)

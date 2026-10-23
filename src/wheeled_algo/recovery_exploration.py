"""Recovery-leg sampling scale; the exported deterministic mean is unchanged."""
import math

import torch

from .robust_getup import SensorConditionedMean, tilt_gate


class RecoveryNoiseMean(SensorConditionedMean):
    def __init__(self, mean, quiet_exploration_fraction, quiet_wheel_exploration_fraction,
                 recovery_leg_fraction):
        if (isinstance(recovery_leg_fraction, bool) or not math.isfinite(recovery_leg_fraction)
                or not .1 <= recovery_leg_fraction <= 1.):
            raise ValueError("Recovery leg noise fraction must be finite within[.1,1]")
        super().__init__(mean, quiet_exploration_fraction, quiet_wheel_exploration_fraction)
        self.recovery_leg_fraction = float(recovery_leg_fraction)

    def forward(self, obs):
        output = super().forward(obs)
        if self.recovery_leg_fraction == 1.:
            return output
        gate = tilt_gate(obs, self.mean.stance_aware_recovery, self.mean.recovery_shape_aware)
        factor = 1. - (1. - self.recovery_leg_fraction) * gate
        log_std = torch.cat((output[:, 1, :4] + factor.log(), output[:, 1, 4:]), -1)
        return torch.stack((output[:, 0], log_std), -2)

"""Bounded standing command intervention for causal native evaluation only.

The outer actor still sees the original request. Only its frozen locomotion
parent receives a forward request corrected from the existing wheel encoders.
No optimizer, state estimate, new observation or motor-level controller is added.
"""
import math

import torch
from torch import nn

from .robust_getup import quiet_gate


class StandingCommandFeedbackParent(nn.Module):
    def __init__(self, parent, gain, velocity_estimator=None):
        super().__init__()
        if not math.isfinite(gain) or abs(gain) > 8.:
            raise ValueError("Diagnostic command feedback gain must be finite and within +/-8")
        self.inner = parent
        self.velocity_estimator = velocity_estimator
        self.register_buffer("gain", torch.tensor(float(gain)))

    @property
    def command(self):
        # Preserve the outer actor's authenticated high-speed reference path.
        return self.inner.command

    def command_delta(self, obs):
        # Encoder scale .1, wheel radius .06 m, forward = (left-right)/2.
        rolling_m_s = .3 * (obs[:, 20:21] - obs[:, 21:22])
        valid = torch.ones_like(rolling_m_s, dtype=torch.bool)
        if self.velocity_estimator is not None:
            rolling_m_s, valid = self.velocity_estimator(obs)
        delta = (-self.gain * quiet_gate(obs) * rolling_m_s).clamp(-.15, .15)
        return torch.where(valid, delta, torch.zeros_like(delta))

    def forward(self, obs):
        delta = self.command_delta(obs)
        forward = torch.where(delta == 0., obs[:, :1], obs[:, :1] + delta)
        return self.inner(torch.cat((forward, obs[:, 1:]), -1))

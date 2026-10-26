"""Sensor-only wheel-update scope, evaluated as an excluded zero-PPO counterfactual."""
import torch
from torch import nn
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from fallen_output_partition_probe001 import partition

VERSION = 'fallen-wheel-entry-protected-probe-v1'


class EntryProtectedFallenWheels(nn.Module):
    def __init__(self, reference, learned, strength=1.):
        super().__init__()
        if strength not in (0., 1.):
            raise ValueError('Only neutral or the predeclared full scoped intervention')
        self.reference = reference
        self.learned = learned
        self.strength = strength
        self.partition_receipt = partition(learned, reference, 'wheels')
        self.requires_grad_(False)

    def forward(self, obs):
        old = self.reference(obs)
        if self.strength == 0.:
            return old
        new = self.learned(obs)
        gate = (1. - handoff_entry_gate(obs) / .05).clamp(0., 1.)
        wheel = old[:, 4:] + gate * (new[:, 4:] - old[:, 4:])
        wheel = torch.where(gate >= 1., new[:, 4:], torch.where(gate > 0., wheel, old[:, 4:]))
        return torch.cat((old[:, :4], wheel), -1)

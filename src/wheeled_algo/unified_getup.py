"""One sensor-only six-axis policy, with continuous recovery/motion authority.

The deployed 1346 policy supplies the initial upright behavior. A shared arc
head sees BOTH requests; there is no translation-versus-yaw winner selection in
the new head. Fallen states have full leg orbit authority from their first
enabled tick. No elapsed time, privileged contact/height or scripted trajectory
is an actor input.
"""
from __future__ import annotations

import torch
from torch import nn


KIND = "wheel-leg-unified-getup-v1"


def recovery_gate(obs):
    tilt = (obs[:, 9:10] + .94) / .44
    legs = (obs[:, 10:14].abs().amax(-1, keepdim=True) - 1.) / .8
    return torch.maximum(tilt, legs).clamp(0., 1.)


class UnifiedGetupActor(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent.eval().requires_grad_(False)
        self.recovery = nn.Sequential(nn.Linear(35, 128), nn.ELU(), nn.Linear(128, 128),
                                      nn.ELU(), nn.Linear(128, 6))
        self.arc = nn.Sequential(nn.Linear(35, 64), nn.ELU(), nn.Linear(64, 64),
                                nn.ELU(), nn.Linear(64, 6))
        for head in (self.recovery, self.arc):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        scale = torch.ones(35)
        scale[0], scale[2], scale[3] = 1 / 3, 1 / 4, 1 / 2
        scale[16:22], scale[22:26], scale[26:28] = .3, 1 / 13, 1 / 9
        self.register_buffer("feature_scale", scale)
        self.register_buffer("arc_bound", torch.tensor([.6] * 4 + [2.] * 2))
        self.register_buffer("motion_bound", torch.tensor([3.] * 4 + [9.] * 2))
        self.register_buffer("action_bound", torch.tensor([13.] * 4 + [9.] * 2))

    def forward(self, obs):
        features = torch.tanh(obs * self.feature_scale)
        gate = recovery_gate(obs)
        parent = self.parent(obs)
        arc_gate = ((obs[:, :1].abs() - .05) / .2).clamp(0., 1.) * ((obs[:, 2:3].abs() - .05) / .2).clamp(0., 1.)
        motion = parent + arc_gate * self.arc_bound * torch.tanh(self.arc(features))
        motion = torch.maximum(torch.minimum(motion, self.motion_bound), -self.motion_bound)
        raw = self.recovery(features)
        # Absolute external targets with an internal previous-target recurrence.
        # Passive reset supplies measured joint angles as the previous target.
        legs = (obs[:, 22:26] + 1.2 * torch.tanh(raw[:, :4])).clamp(-13., 13.)
        wheels = 9. * torch.tanh(raw[:, 4:6])
        fallen = torch.cat((legs, wheels), -1)
        result = motion + gate * (fallen - motion)
        return torch.maximum(torch.minimum(result, self.action_bound), -self.action_bound)

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

"""Read-only, paired native comparison of recovery target/gate hypotheses."""
import torch
from torch import nn

VARIANTS = ("previous_target_leg_gate", "measured_target_leg_gate", "previous_target_tilt_gate", "measured_target_tilt_gate")


class RecoveryAblations(nn.Module):
    def __init__(self, policy, variants):
        super().__init__()
        self.policy = policy
        self.register_buffer("variant", torch.tensor([VARIANTS.index(name) for name in variants]))

    def forward(self, obs):
        p = self.policy
        features = torch.tanh(obs * p.feature_scale)
        tilt = ((obs[:, 9:10] + .94) / .44).clamp(0., 1.)
        leg_gate = ((obs[:, 10:14].abs().amax(-1, keepdim=True) - 1.) / .8).clamp(0., 1.)
        gate = torch.where((self.variant >= 2)[:, None], tilt, torch.maximum(tilt, leg_gate))
        arc_gate = ((obs[:, :1].abs() - .05) / .2).clamp(0., 1.) * ((obs[:, 2:3].abs() - .05) / .2).clamp(0., 1.)
        motion = p.parent(obs) + arc_gate * p.arc_bound * torch.tanh(p.arc(features))
        motion = torch.maximum(torch.minimum(motion, p.motion_bound), -p.motion_bound)
        raw = p.recovery(features)
        previous = torch.where((self.variant % 2 == 1)[:, None], obs[:, 10:14] / .25, obs[:, 22:26])
        legs = (previous + 1.2 * torch.tanh(raw[:, :4])).clamp(-13., 13.)
        fallen = torch.cat((legs, 9. * torch.tanh(raw[:, 4:6])), -1)
        result = motion + gate * (fallen - motion)
        return torch.maximum(torch.minimum(result, p.action_bound), -p.action_bound)

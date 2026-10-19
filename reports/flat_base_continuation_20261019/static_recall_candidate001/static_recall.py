"""Diagnostic composition of an unchanged moving mean and a frozen static snapshot."""
from copy import deepcopy
import torch
from torch import nn
from wheeled_algo.low_demand_feedback import LowDemandFeedbackActor
from wheeled_algo.separated_feedback import SeparatedFeedbackActor


class StaticRecallActor(nn.Module):
    """Select the preserved static feedback continuously using visible commands.

    At zero planar demand this equals the recalled actor at every input. At
    |vx|>=.1m/s or |yaw|>=.2rad/s it equals the moving parent. Braking at zero
    demand also changes, so static equality is not a stopping guarantee.
    This diagnostic module creates no trained checkpoint or release status.
    """
    def __init__(self, parent, recalled):
        super().__init__()
        if type(parent) is not LowDemandFeedbackActor or type(recalled) is not SeparatedFeedbackActor:
            raise ValueError('Require the declared v4 moving and v3 static actors')
        current, old = parent.command.state_dict(), recalled.command.state_dict()
        if current.keys() != old.keys() or any(not torch.equal(v, old[n]) for n,v in current.items()):
            raise ValueError('Static recall must preserve the exact same source and command branch')
        for name in ('input_center', 'input_scale', 'output_scale'):
            if not torch.equal(getattr(parent, name), getattr(recalled, name)):
                raise ValueError('Static recall cannot change feedback normalization or bounds')
        self.parent = deepcopy(parent)
        self.static_feedback = deepcopy(recalled.feedback)
        self.requires_grad_(False)

    @staticmethod
    def moving_mix(obs):
        return torch.maximum(obs[:, :1].abs()/.1, obs[:, 2:3].abs()/.2).clamp(0., 1.)

    def feedback_actions(self, obs):
        features = (obs[:, :28]-self.parent.input_center)*self.parent.input_scale
        static = self.parent.output_scale*torch.tanh(self.static_feedback(features))
        moving = self.parent.feedback_actions(obs)
        mix = self.moving_mix(obs)
        blended = static+mix*(moving-static)
        return torch.where(mix <= 0., static, torch.where(mix >= 1., moving, blended))

    def forward(self, obs):
        return self.parent.command(obs)+self.feedback_actions(obs)

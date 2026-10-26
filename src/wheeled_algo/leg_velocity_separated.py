"""Fresh actor architecture separating leg targets from direct leg-speed response.

The four neural position-target means use current posture, gyro, angles,
commands and previous actions. Raw leg-speed features still reach both wheel
outputs, and native1kHz leg PD retains its measured-velocity damping. This is
an actor parameterization, with Gaussian sampling/log probabilities computed
from the final mean; it is not a runtime action filter or target hold.
"""
from copy import deepcopy

import torch
from rsl_rl.utils import unpad_trajectories

from .wheel_velocity_increment import WheelVelocityIncrementModel


def neutral_leg_velocity(frame):
    return torch.cat((frame[..., :16], torch.zeros_like(frame[..., 16:20]), frame[..., 20:]), dim=-1)


class LegVelocitySeparatedModel(WheelVelocityIncrementModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.register_buffer("leg_velocity_separated_abi", torch.tensor([1, 16, 20, 4], dtype=torch.int64))

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        frame = obs["policy"]
        full = self.mlp(self.get_latent(obs, masks, hidden_state))
        neutral = obs.clone()
        neutral["policy"] = neutral_leg_velocity(frame)
        legs = self.mlp(self.get_latent(neutral, masks, hidden_state))
        output = torch.cat((legs[..., :4], full[..., 4:] + frame[..., 20:22]), dim=-1)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def as_jit(self):
        return _SeparatedInference(self)

    def as_onnx(self, verbose=False):
        return _SeparatedInference(self)


class _SeparatedInference(torch.nn.Module):
    is_recurrent = False

    def __init__(self, model):
        super().__init__()
        self.obs_normalizer = deepcopy(model.obs_normalizer)
        self.mlp = deepcopy(model.mlp)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        full = self.mlp(self.obs_normalizer(frame))
        legs = self.mlp(self.obs_normalizer(neutral_leg_velocity(frame)))
        return torch.cat((legs[..., :4], full[..., 4:] + frame[..., 20:22]), dim=-1)

    @torch.jit.export
    def reset(self) -> None:
        pass

    def get_dummy_inputs(self):
        return (torch.zeros(1, 35),)

    @property
    def input_names(self):
        return ["obs"]

    @property
    def output_names(self):
        return ["actions"]

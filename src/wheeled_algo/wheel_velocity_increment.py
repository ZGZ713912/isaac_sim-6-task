"""From-zero wheel action parameterization, using the existing raw35D ABI.

The learned wheel outputs are velocity increments. The final six actions still
contain absolute wheel velocity targets scaled by10 rad/s. No angle/gyro gain,
teacher, old checkpoint, recovery script or additional inference state is used.

Distribution integration follows the installed RSL-RL5.5.1 MLPModel: the shift
is applied before Gaussian.update(), so sampled actions and native PPO log
probabilities describe the same final targets. Inference retains that shift.
Reference: https://github.com/leggedrobotics/rsl_rl/blob/v5.5.1/rsl_rl/models/mlp_model.py
"""
from copy import deepcopy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import unpad_trajectories


class WheelVelocityIncrementModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        if self.obs_groups != ["policy"] or self.obs_dim != 35 or output_dim != 6:
            raise ValueError("Wheel increments require the exact35D actor and6 policy-order outputs")
        if self.distribution is not None and type(self.distribution) is not GaussianDistribution:
            raise ValueError("Wheel increments require the native untransformed Gaussian")
        # Fail closed if a plain MLP loader is accidentally used for this ABI.
        self.register_buffer("wheel_velocity_increment_abi", torch.tensor([1, 20, 21, 10], dtype=torch.int64))

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        frame = obs["policy"]
        output = self.mlp(self.get_latent(obs, masks, hidden_state))
        # The raw encoder features are0.1*dq. They already equal dq/10, the
        # absolute wheel action coordinates. Do not normalize this identity.
        output = output + torch.cat((torch.zeros_like(output[..., :4]), frame[..., 20:22]), dim=-1)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def as_jit(self):
        return _IncrementInference(self)

    def as_onnx(self, verbose=False):
        return _IncrementInference(self)


class _IncrementInference(torch.nn.Module):
    is_recurrent = False

    def __init__(self, model):
        super().__init__()
        self.obs_normalizer = deepcopy(model.obs_normalizer)
        self.mlp = deepcopy(model.mlp)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        output = self.mlp(self.obs_normalizer(frame))
        return output + torch.cat((torch.zeros_like(output[..., :4]), frame[..., 20:22]), dim=-1)

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

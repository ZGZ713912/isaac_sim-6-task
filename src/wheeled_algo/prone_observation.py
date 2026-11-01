"""Keep gated encoder odometry in its declared physical scale during PPO/play."""
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization


class GatedObservationNormalization(EmpiricalNormalization):
    def __init__(self, shape):
        super().__init__(shape)
        mask = torch.zeros(shape, dtype=torch.bool)
        mask[[14, 15, 31]] = True
        self.register_buffer("fixed_scale_mask", mask)

    def forward(self, x):
        # These are bounded displacement features and a binary validity bit.
        # Their meaning must not depend on how rarely another role activates them.
        return torch.where(self.fixed_scale_mask, x.clamp(-1., 1.), super().forward(x))


class ProneMotionModel(MLPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.obs_normalization or self.obs_dim < 35:
            raise ValueError("ProneMotionModel requires normalized policy35 or its critic prefix")
        self.obs_normalizer = GatedObservationNormalization(self.obs_dim)

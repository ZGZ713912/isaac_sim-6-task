"""Read-only neural feedback ablation; never a trained or deployed policy.

Keep all four leg outputs native. Interpolate only wheel outputs between the
native35D input and the same input with its two wheel-speed features neutral.
"""
import math

import torch


def validate_scale(scale):
    if not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 0. <= scale <= 1.:
        raise ValueError("Wheel encoder response scale must be finite within0..1")
    return float(scale)


def response(native_actor, observation, scale):
    scale = validate_scale(scale)
    if native_actor.training or native_actor.is_recurrent:
        raise ValueError("The encoder probe supports a frozen feed-forward inference actor only")
    full = native_actor(observation)
    if scale == 1.:
        return full
    if observation["policy"].shape[-1] != 35 or full.shape[-1] != 6:
        raise ValueError("Encoder probe requires the exact35D to6D action ABI")
    neutral = observation.clone()
    neutral["policy"][..., 20:22] = 0.
    anchor = native_actor(neutral)
    wheels = anchor[..., 4:] + scale * (full[..., 4:] - anchor[..., 4:])
    return torch.cat((full[..., :4], wheels), dim=-1)


class WheelEncoderResponseProbe(torch.nn.Module):
    def __init__(self, native_actor, scale):
        super().__init__()
        self.native_actor, self.scale = native_actor, validate_scale(scale)
        self.eval()

    def forward(self, observation):
        return response(self.native_actor, observation, self.scale)

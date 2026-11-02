"""Fixed-shape masked operations for the CUDA sampling path.

Reading ``value[boolean_mask]`` materializes dynamic-length indices and makes
CUDA wait for their size on the host. Keep lane selection on the device when
the caller only needs a masked state update or a scalar reduction.
"""
import torch


def copy_masked_rows_(target, mask, value):
    condition = mask.reshape((-1,) + (1,) * (target.ndim - 1))
    return torch.where(condition, value, target, out=target)


def masked_mean(value, mask):
    condition = mask.reshape((-1,) + (1,) * (value.ndim - 1))
    width = value.numel() // mask.numel()
    selected = torch.where(condition, value, 0.)
    return selected.sum() / (mask.sum().clamp_min(1) * width)


def masked_max(value, mask):
    condition = mask.reshape((-1,) + (1,) * (value.ndim - 1))
    maximum = torch.where(condition, value, -torch.inf).amax()
    return torch.where(mask.any(), maximum, 0.)

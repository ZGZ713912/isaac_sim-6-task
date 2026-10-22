"""Learned common leg rotation and bounded, nonintegrating knee coordinates."""
import torch

from .paired_recovery import DELTA_BOUNDS, NOMINAL_DELTA, paired_offsets, wrap


def smooth_recovery_legs(raw, previous, measured, step, knee_center, knee_step=.15):
    pair = raw.unflatten(-1, (2, 2))
    sign = raw.new_tensor([1., -1.])
    bounds = raw.new_tensor(DELTA_BOUNDS)
    nominal = raw.new_tensor(NOMINAL_DELTA)
    center = raw.new_tensor(knee_center)
    bias = ((center - bounds[:, 0]) / (bounds[:, 1] - center)).log()
    logits = 2. * sign * (pair[..., 1] - pair[..., 0]) + bias
    desired = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * logits.sigmoid()
    encoders = measured.unflatten(-1, (2, 2))
    actual = (sign * (nominal + wrap(encoders[..., 1] - encoders[..., 0]))).clamp(bounds[:, 0], bounds[:, 1])
    error = desired - actual
    # Unlike a hard clamp on an accumulated target, this leaves a nonzero knee
    # gradient at the stops. Limit each request relative to measured geometry.
    delta = actual + knee_step * error / (knee_step + error.abs())
    offset_delta = sign * delta - nominal
    common = .25 * previous.unflatten(-1, (2, 2)).mean(-1) + step * .25 * pair.mean(-1).tanh()
    offsets = torch.stack((common - .5 * offset_delta, common + .5 * offset_delta), -1).flatten(-2)
    return paired_offsets(offsets, measured) / .25

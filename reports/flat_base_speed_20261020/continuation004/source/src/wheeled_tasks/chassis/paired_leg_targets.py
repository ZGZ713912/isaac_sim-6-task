"""The deployed paired_branch107_v1 target projection in model coordinates.

Reference: RMCS rmcs_rl/src/wheel_leg/leg_target_winding.hpp. The complete
mechanical specification is authenticated separately, before scene creation.
"""
import math

import torch

MODEL_SPEC_SHA256 = "0ac82045ab584f20091538aa782d93ea545382dc701faca1f8ae92aeb7cd1997"
WINDING = "paired_branch107_v1"
SIGNED_DELTA_BOUNDS = ((-.14062820701680934, 1.3692123767724917),
                       (-.1406283965709381, 1.3692125038958158))


def paired_leg_targets(desired, current):
    """Project each hip/aux pair, then choose one shared nearest winding.

    Like the deployed code, calculate in double precision. Returning the
    caller's dtype makes the float32 PhysX boundary explicit.
    """
    if desired.shape != current.shape or desired.shape[-1] != 4:
        raise ValueError("Paired leg targets require matching [...,4] tensors")
    dtype = desired.dtype
    wanted = desired.double().unflatten(-1, (2, 2))
    measured = current.double().unflatten(-1, (2, 2))
    sign = desired.new_tensor([1., -1.], dtype=torch.float64)
    bounds = desired.new_tensor(SIGNED_DELTA_BOUNDS, dtype=torch.float64)
    mean = wanted.mean(-1)
    delta = (sign * (wanted[..., 1] - wanted[..., 0])).clamp(bounds[:, 0], bounds[:, 1])
    turns = torch.floor(((measured[..., 0] * .5 + measured[..., 1] * .5 - mean) / math.tau) + .5)
    hip = mean - sign * delta * .5 + turns * math.tau
    auxiliary = mean + sign * delta * .5 + turns * math.tau
    actual = sign * (auxiliary - hip)
    outside = (actual < bounds[:, 0]) | (actual > bounds[:, 1])
    auxiliary = torch.where(outside, torch.nextafter(hip + sign * delta, hip), auxiliary)
    return torch.stack((hip, auxiliary), -1).flatten(-2).to(dtype=dtype)

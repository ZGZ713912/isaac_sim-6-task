"""Candidate training cost; no policy input or runtime controller change."""
import math
import torch


def recovery_support_cost(clearance_m, gravity_z, recovery_active, weight=0.):
    """Cost per second for unrecovered, near-upright wheels above the floor.

    Scale by15cm, rather than an exponential whose slope vanishes at the
    observed45cm failure. Sideways/inverted motion has no cost. Native wheel
    cylinder clearance is training-only state; it never enters the actor.
    """
    if not math.isfinite(weight) or not 0. <= weight <= 2.:
        raise ValueError("Recovery support weight must be finite within0..2")
    if weight == 0.:
        return torch.zeros_like(gravity_z)
    upright = ((-gravity_z - .7) / .2).clamp(0., 1.)
    return weight * upright * recovery_active * clearance_m.clamp_min(0.).mean(-1) / .15

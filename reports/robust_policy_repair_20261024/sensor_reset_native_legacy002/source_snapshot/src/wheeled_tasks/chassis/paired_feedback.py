"""Paired PD errors invariant to independently wrapped encoder coordinates.

Unlimited PhysX revolute coordinates can cross their reporting seam at
different times. This lifts feedback onto the feasible common-angle branch;
it neither writes simulator state nor changes the requested knee angle.
"""
import torch


def paired_position_error(target, measured):
    wanted = target.unflatten(-1, (2, 2))
    actual = measured.unflatten(-1, (2, 2))
    sign = target.new_tensor([1., -1.])
    difference = actual[..., 1] - actual[..., 0]
    delta = sign * torch.atan2(difference.sin(), difference.cos())
    wanted_delta = sign * (wanted[..., 1] - wanted[..., 0])
    common = actual[..., 0] + .5 * sign * delta
    common_difference = wanted.mean(-1) - common
    common_error = torch.atan2(common_difference.sin(), common_difference.cos())
    differential_error = sign * (wanted_delta - delta)
    return torch.stack((common_error - .5 * differential_error,
                        common_error + .5 * differential_error), -1).flatten(-2)

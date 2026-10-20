"""Diagnostic wheel-error prior using only the deployed 35D observation.

This is a zero-PPO probe, not an adopted policy or deployment controller.
Encoder speed is not body speed when the tire slips.
"""
import torch


def wheel_error(obs):
    turn = obs[:, 2:3] * (.4373 / 2.)
    desired = torch.cat((obs[:, :1] - turn, -obs[:, :1] - turn), -1) / .6
    return desired - obs[:, 20:22]


def high_gate(obs):
    return (torch.maximum(obs[:, :1].abs() / .75, obs[:, 2:3].abs() / 1.5) - 1.).clamp(0., 1.)


def correct(obs, parent, gain):
    offset = (gain * wheel_error(obs)).clamp(-6., 6.) * high_gate(obs)
    correction = torch.cat((torch.zeros_like(obs[:, :4]), offset), -1)
    return torch.where(high_gate(obs) <= 0., parent, parent + correction)


def reachable_reference(obs, margin):
    """Condition on a reachable request while keeping external goals unchanged.

    Encoder speed is a rolling estimate, not privileged body velocity. Yaw uses
    the measured IMU gyro. A zero margin selects the unchanged baseline.
    """
    gate = high_gate(obs) * (margin > 0.).to(obs.dtype)
    speed = (obs[:, 20:21] - obs[:, 21:22]) * .3
    yaw = obs[:, 6:7] * 2.
    vx = speed + (obs[:, :1] - speed).clamp(-margin, margin)
    yaw_margin = margin / (.4373 / 2.)
    wz = yaw + (obs[:, 2:3] - yaw).clamp(-yaw_margin, yaw_margin)
    vx, wz = vx.clamp(-3., 3.), wz.clamp(-4., 4.)
    candidate = torch.cat((vx, obs[:, 1:2], wz, obs[:, 3:]), -1)
    return torch.where(gate <= 0., obs, obs + gate * (candidate - obs))

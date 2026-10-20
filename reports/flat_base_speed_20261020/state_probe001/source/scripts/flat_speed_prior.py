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
    vx = torch.where(obs[:, :1].abs() > .75, vx, obs[:, :1])
    wz = torch.where(obs[:, 2:3].abs() > 1.5, wz, obs[:, 2:3])
    candidate = torch.cat((vx, obs[:, 1:2], wz, obs[:, 3:]), -1)
    return torch.where(gate <= 0., obs, obs + gate * (candidate - obs))


def lean_reference(obs, gain):
    """Small velocity-to-lean reference, capped at 0.05 radians.

    This is an experimental outer velocity loop. The frozen policy still owns
    all joint actions and balance; native survival must validate the sign.
    """
    margin = torch.where(obs[:, :1] < 0., .3, .15)
    margin = torch.where(obs[:, 2:3].abs()/1.5 > obs[:, :1].abs()/.75, .05, margin)
    reference = reachable_reference(obs, margin)
    error = obs[:, :1] - (obs[:, 20:21]-obs[:, 21:22])*.3
    rolling = (obs[:, 20:21]-obs[:, 21:22])*.3
    progress = (rolling*obs[:, :1].sign()).clamp(0.,1.)
    angle = (gain*error).clamp(-.05,.05)*high_gate(obs)*progress
    angle = torch.where(obs[:, :1].abs() > .75, angle, torch.zeros_like(angle))
    gx, gz = reference[:, 7:8], reference[:, 9:10]
    gravity_x = gx*angle.cos()-gz*angle.sin()
    gravity_z = gx*angle.sin()+gz*angle.cos()
    return torch.cat((reference[:, :7], gravity_x, reference[:, 8:9], gravity_z, reference[:, 10:]), -1)


def state_reference(obs, gain):
    """Diagnostic coordinated reference, with measured sign of encoder response.

    At steady high reverse speed the frozen actor increases wheel output when
    its encoder reference moves toward the requested velocity. Its complete
    six-action response is retained. The reference shift fades out at launch.
    Pure turns independently probe the gyro-reference acceleration margin.
    """
    lean = torch.where(obs[:, :1] < 0., -.01, -.04)
    reference = lean_reference(obs, lean)
    rolling = (obs[:, 20:21]-obs[:, 21:22])*.3
    progress = (rolling*obs[:, :1].sign()).clamp(0., 1.)
    offset = (gain*(obs[:, :1]-rolling)/.6).clamp(-2.5, 2.5)*progress
    offset = torch.where(obs[:, :1] < -.75, offset, torch.zeros_like(offset))
    encoders = reference[:, 20:22]+torch.cat((offset, -offset), -1)
    reference = torch.cat((reference[:, :20], encoders, reference[:, 22:]), -1)
    yaw = obs[:, 6:7]*2.
    margin = (.05+gain*.4)/(.4373/2.)
    wz = (yaw+(obs[:, 2:3]-yaw).clamp(-margin, margin)).clamp(-4.,4.)
    wz = torch.where(obs[:, 2:3].abs()>1.5, wz, reference[:, 2:3])
    return torch.cat((reference[:, :2], wz, reference[:, 3:]), -1)

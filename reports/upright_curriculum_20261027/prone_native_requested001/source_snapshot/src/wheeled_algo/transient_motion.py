"""Opt-in sensor-visible gate and bounded leg authority for the next pilot."""
from copy import deepcopy
import math

import torch


ACTOR_SETTINGS = dict(version='leg-assisted-transient-v1',
    vx_deadband_m_s=.03, vx_ramp_m_s=.12,
    yaw_deadband_rad_s=.03, yaw_ramp_rad_s=.07,
    pitch_deadband_rad=math.radians(2.), pitch_ramp_rad=math.radians(6.),
    raw_action_bounds=[.6] * 4 + [.8] * 2)


def validate_settings(settings):
    if settings != ACTOR_SETTINGS:
        raise ValueError('Transient leg settings differ from the declared candidate')
    return deepcopy(settings)


def transient_gate(obs, inherited, settings):
    """Encoder speed is a proxy, not privileged chassis velocity or friction."""
    rolling = (obs[:, 20:21] - obs[:, 21:22]) * .3
    gyro = obs[:, 6:7] * 2.
    vx = ((obs[:, :1] - rolling).abs() - settings['vx_deadband_m_s']) / settings['vx_ramp_m_s']
    yaw = ((obs[:, 2:3] - gyro).abs() - settings['yaw_deadband_rad_s']) / settings['yaw_ramp_rad_s']
    # Projected gravity remains bounded and exports without an angle operator.
    pitch = (obs[:, 7:8].abs() - math.sin(settings['pitch_deadband_rad'])) / math.sin(settings['pitch_ramp_rad'])
    return torch.maximum(inherited, torch.maximum(torch.maximum(vx, yaw), pitch).clamp(0., 1.))


def migrate_actor_buffer(checkpoint, settings):
    """Declare a function change; preserve weights, Adam, std, RNG and clocks."""
    validate_settings(settings)
    result = deepcopy(checkpoint)
    old = result['actor_state_dict']['mlp.motion_bound']
    if not torch.equal(old.cpu(), torch.tensor([.15] * 4 + [.8] * 2)):
        raise ValueError('Transient migration requires the original directional bounds')
    result['actor_state_dict']['mlp.motion_bound'] = old.new_tensor(settings['raw_action_bounds'])
    return result

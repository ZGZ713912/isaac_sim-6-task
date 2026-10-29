"""Damp the frozen leg recurrence while retaining direct learned leg intent."""
from copy import deepcopy

import torch


SETTINGS = dict(version='parent-legs-relaxed-fast-residual-v1', parent_leg_alpha=.25,
                previous_action_slice=[22, 26], learned_correction='direct', wheels='direct')


def validate_settings(settings):
    if settings != SETTINGS:
        raise ValueError('Undeclared parent-leg feedback setting')
    return deepcopy(settings)


def parent_leg_actions(parent, obs, settings):
    previous = obs[:, 22:26]
    legs = previous + settings['parent_leg_alpha'] * (parent[:, :4] - previous)
    return torch.cat((legs, parent[:, 4:]), -1)

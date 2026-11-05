"""Opt-in recovery discovery allocation; cold admission remains independent."""
from copy import deepcopy
import math

import torch


def contract():
    return dict(schema="wheel-leg-recovery-training-focus-v1",
        recovery_fraction=.95, standing_fraction=.05,
        recovery_command=[0., 0., .305],
        completion_stable_seconds=3., completion_end="training_timeout_bootstrap",
        evaluation_early_end=False, motion_training=False,
        admission_gates="unchanged", teacher=False, actor_observation_additions=False)


def validate(config):
    if config != contract():
        raise ValueError("Recovery focus requires the complete versioned training contract")
    return deepcopy(config)


def descriptors(count, generation, config, standing):
    """Focus on recovery, preserving a small literal standing auxiliary pool."""
    c = validate(config)
    if type(count) is not int or count < 2 or type(generation) is not int or generation < 0:
        raise ValueError("Recovery discovery needs two lanes and a nonnegative generation")
    n = min(count-1, max(1, math.floor(count*c["recovery_fraction"]+.5)))
    recovery = dict(curriculum_skill="recovery", role="prone", recovery=True,
        target=list(c["recovery_command"]), birth_height=.305, pitch_deg=0.)
    rows = [deepcopy(recovery) for _ in range(n)]+[deepcopy(standing) for _ in range(count-n)]
    # Keep lane ownership stable across partial resets, as the original course
    # does. Episode generation still advances the independent pose sampler.
    return rows


def completed_timeout(config, quiet_hold, recovery_lane, terminated, *, training, evaluating):
    """A successful training horizon is a timeout, never a failure or cold pass."""
    if config is None or not training or evaluating:
        return torch.zeros_like(recovery_lane)
    return recovery_lane & ~terminated & (quiet_hold >= config["completion_stable_seconds"])

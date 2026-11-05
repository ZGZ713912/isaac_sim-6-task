"""End fixed-case physics episodes after preserving their terminal evidence.

Training already resets terminal lanes. Fixed evaluations keep a sticky finished
mask so a failed case cannot acquire a new attempt or change its original score.
Re-seed those retired lanes at each policy boundary, instead of allowing their
uncontrolled post-terminal mechanisms to diverge while other cases are measured.
"""
from copy import deepcopy
import random

import numpy as np
import torch


CONTRACT = dict(schema="wheel-leg-cold-terminal-lifecycle-v1",
    mode="reset_retired_fixed_cases_at_policy_boundary",
    terminal_evidence="preserved_before_reset",
    finished_mask="sticky_until_next_case",
    active_episode_physics="unchanged",
    training_episode_lifecycle="unchanged",
    live_random_streams_and_reset_generation="preserved",
    nonfinite_protection="unchanged",
    closure_error_is_not_a_fault=True)


def validate(config):
    if config != CONTRACT:
        raise ValueError("Use the complete versioned cold terminal lifecycle contract")
    return deepcopy(config)


def reset_retired_fixed_cases(env):
    """Called after diagnostics/reward/done are captured, before next physics."""
    if env.fixed_cases is None:
        return 0
    retired = env.finished.nonzero().flatten()
    if len(retired) == 0:
        return 0
    generators = [getattr(env, "generator", None), getattr(getattr(env, "domain", None), "rng", None)]
    generators = [(g, g.get_state()) for g in generators if g is not None]
    sampler = getattr(env, "recovery_sampler", None)
    sampler_state = (sampler.rng.getstate(), deepcopy(sampler.class_cursor)) if sampler is not None else None
    from .recovery_start_curriculum import optional_state, restore_optional
    start_state = optional_state(sampler)
    generation = getattr(env, "prone_reset_generation", None)
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
           np.random.get_state(), random.getstate())
    try:
        env.reset(retired)
    finally:
        # Retiring a ghost case must not consume the live cases' next noise
        # sample or change the post-review training population assignment.
        for generator, state in generators:
            generator.set_state(state)
        if sampler_state is not None:
            sampler.rng.setstate(sampler_state[0])
            sampler.class_cursor = sampler_state[1]
        restore_optional(sampler, start_state)
        if generation is not None:
            env.prone_reset_generation = generation
        torch.set_rng_state(rng[0])
        if rng[1]:
            torch.cuda.set_rng_state_all(rng[1])
        np.random.set_state(rng[2])
        random.setstate(rng[3])
    # Reset clears finished for new episodes. These lanes remain retired within
    # this cold batch, including unscored duplicates; they get no second trial.
    env.finished[retired] = True
    count = len(retired)
    env.cold_evaluation_reset_events += count
    return count

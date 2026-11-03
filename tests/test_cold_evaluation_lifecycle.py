"""A failed cold trial stays failed and cannot crash the remaining trials."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.cold_evaluation_lifecycle import CONTRACT, reset_retired_fixed_cases, validate


def fake_env(finished, *, fixed=True):
    env = SimpleNamespace(fixed_cases=[{}]*len(finished) if fixed else None,
        finished=torch.tensor(finished), cold_evaluation_reset_events=0,
        position=torch.arange(len(finished), dtype=torch.float), calls=[])
    def reset(ids):
        env.calls.append(ids.clone())
        env.finished[ids] = False
        env.position[ids] = 0.
    env.reset = reset
    return env


def test_retired_duplicates_and_scored_failures_remain_sticky_on_every_boundary():
    env = fake_env([False, True, False, True])
    terminal_record = env.position.clone()
    assert reset_retired_fixed_cases(env) == 2
    assert env.calls[0].tolist() == [1, 3]
    assert env.finished.tolist() == [False, True, False, True]
    assert env.position.tolist() == [0., 0., 2., 0.]
    assert terminal_record.tolist() == [0., 1., 2., 3.]
    env.position[[1, 3]] = 123.
    assert reset_retired_fixed_cases(env) == 2
    assert env.cold_evaluation_reset_events == 4
    assert env.finished[1] and env.finished[3]
    assert env.position[2] == 2.  # Active episode was never reset.


def test_no_training_reset_or_active_case_reset_is_added():
    for env in (fake_env([False, False]), fake_env([True, False], fixed=False)):
        assert reset_retired_fixed_cases(env) == 0
        assert not env.calls


@pytest.mark.parametrize("fail_reset", [False, True])
def test_retirement_cannot_advance_live_noise_draws_or_population_generation(fail_reset):
    env = fake_env([True, False])
    env.generator = torch.Generator().manual_seed(107)
    env.domain = SimpleNamespace(rng=torch.Generator().manual_seed(83977))
    env.recovery_sampler = SimpleNamespace(rng=random.Random(107), class_cursor={"standard": 2})
    env.prone_reset_generation = 7
    snapshots = [g.get_state().clone() for g in (env.generator, env.domain.rng)]
    global_rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    sampler_rng = env.recovery_sampler.rng.getstate()
    reset = env.reset
    def noisy_reset(ids):
        reset(ids)
        torch.rand(3, generator=env.generator)
        torch.rand(3, generator=env.domain.rng)
        torch.rand(3); np.random.rand(3); random.random()
        env.recovery_sampler.rng.random()
        env.recovery_sampler.class_cursor["standard"] += 1
        env.prone_reset_generation += 1
        if fail_reset:
            raise RuntimeError("native reset failure remains fatal")
    env.reset = noisy_reset
    if fail_reset:
        with pytest.raises(RuntimeError, match="native reset failure"):
            reset_retired_fixed_cases(env)
    else:
        reset_retired_fixed_cases(env)
    assert env.prone_reset_generation == 7
    assert env.recovery_sampler.class_cursor == {"standard": 2}
    assert env.recovery_sampler.rng.getstate() == sampler_rng
    assert all(torch.equal(g.get_state(), state) for g, state in zip((env.generator, env.domain.rng), snapshots))
    assert torch.equal(torch.get_rng_state(), global_rng[0])
    assert np.array_equal(np.random.get_state()[1], global_rng[1][1])
    assert random.getstate() == global_rng[2]


@pytest.mark.parametrize("field", list(CONTRACT))
def test_partial_or_weakened_contract_is_rejected(field):
    broken = deepcopy(CONTRACT)
    broken.pop(field)
    with pytest.raises(ValueError):
        validate(broken)


def test_plan_only_adds_explicit_cold_lifecycle_to_parent_contract():
    root = Path(__file__).resolve().parents[1]
    parent = json.loads((root/'contracts/wheel_leg_prone_motion_v4_terminal_settlement_v1.json').read_text())
    new = json.loads((root/'contracts/wheel_leg_prone_motion_v4_cold_lifecycle_v1.json').read_text())
    assert validate(new.pop('cold_evaluation_lifecycle')) == CONTRACT
    assert new == parent

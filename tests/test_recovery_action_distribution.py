"""Distribution clamps must agree with declared initial action exploration."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from train_prone_motion import learner_config
from wheeled_algo.prone_initialization import initialize_prone_fresh
from wheeled_algo.recovery_action_distribution import contract, std_bounds

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/'contracts/wheel_leg_self_righting_phase_all_v1.json').read_text())


def wide_plan():
    result = deepcopy(PLAN)
    result['exploration_distribution'] = contract()
    result['initial_std'] = result['exploration_distribution']['initial_std'][:]
    return result


def test_large_exploration_requires_a_complete_opt_in():
    wide = wide_plan()
    assert std_bounds(wide) == (.003, 1.) and std_bounds(PLAN) == (.003, .3)
    del wide['exploration_distribution']
    with pytest.raises(ValueError, match='exceeds'): std_bounds(wide)
    wide = wide_plan(); wide['exploration_distribution']['std_range'][1] = 10.
    with pytest.raises(ValueError, match='complete'): std_bounds(wide)


def make_algorithm(plan):
    from rsl_rl.algorithms import PPO
    obs = TensorDict(dict(policy=torch.zeros(32, 35), critic=torch.zeros(32, 87)), batch_size=[32])
    cfg = learner_config(plan, 'cpu'); cfg['multi_gpu'] = None
    return PPO.construct_algorithm(obs, SimpleNamespace(num_envs=32, num_actions=6), cfg, 'cpu'), obs


def test_actual_distribution_preserves_the_requested_std_and_fresh_state():
    plan = wide_plan(); alg, obs = make_algorithm(plan)
    receipt = initialize_prone_fresh(alg, torch.zeros(6), plan)
    alg.actor.distribution.update(alg.actor(obs))
    torch.testing.assert_close(alg.actor.distribution.std[0], torch.tensor(plan['initial_std']))
    assert receipt['std_range'] == [.003, 1.] and not alg.optimizer.state
    assert not receipt['checkpoint_loaded'] and not receipt['teacher_loaded']
    assert all(p.requires_grad for model in (alg.actor, alg.critic) for p in model.parameters())


def test_old_distribution_cannot_silently_clamp_the_new_initialization():
    alg, _ = make_algorithm(PLAN)
    with pytest.raises(ValueError, match='ranges differ'):
        initialize_prone_fresh(alg, torch.zeros(6), wide_plan())

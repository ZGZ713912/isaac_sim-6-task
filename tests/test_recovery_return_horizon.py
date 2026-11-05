"""Physical-time return weights remain distinct from sample and acceptance budgets."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from train_prone_motion import configuration, learner_config, source_hashes
from wheeled_tasks.chassis import prone_motion as task
from wheeled_tasks.chassis.prone_motion_cli import resolve_plan
from wheeled_tasks.chassis.recovery_return_horizon import coefficients, contract, validate

ROOT = Path(__file__).resolve().parents[1]
PLAN_FILE = ROOT/'contracts/wheel_leg_self_righting_return_horizon_v1.json'
PLAN = json.loads(PLAN_FILE.read_text())
BASE = json.loads((ROOT/'contracts/wheel_leg_self_righting_geometric_lift_v1.json').read_text())


@pytest.mark.parametrize('dt', [.005, .01, .02])
@pytest.mark.parametrize('seconds', [.64, 1., 5., 20.])
def test_decay_in_seconds_and_gae_product_are_independent_of_actor_grid(dt, seconds):
    c = coefficients(contract(), dt)
    assert math.isclose(c['gamma']**(seconds/dt), math.exp(-seconds/10.), rel_tol=1e-12)
    assert math.isclose((c['gamma']*c['lam'])**(seconds/dt), math.exp(-seconds), rel_tol=1e-12)
    assert not math.isclose(c['lam']**(seconds/dt), math.exp(-seconds), rel_tol=1e-5)


def test_full_plan_only_changes_the_explicit_return_and_potential_discount():
    assert task.validate(PLAN) == PLAN and task.validate(BASE) == BASE
    restored = deepcopy(PLAN)
    restored.pop('recovery_return_horizon')
    restored['discounts'] = BASE['discounts']
    assert restored == BASE
    assert PLAN['discounts'] == coefficients(contract(), .01)


@pytest.mark.parametrize('filename', ['wheel_leg_prone_motion_v1.json',
    'wheel_leg_prone_motion_v2.json', 'wheel_leg_prone_motion_v4.json',
    'wheel_leg_self_righting_geometric_lift_v1.json'])
def test_legacy_discounts_are_bit_identical_without_the_new_field(filename):
    p = json.loads((ROOT/'contracts'/filename).read_text())
    ratio = task.timing(p)['policy_dt']/task.REFERENCE_DT
    assert task.discounts(p) == dict(gamma=task.REFERENCE_GAMMA**ratio,
                                   lam=task.REFERENCE_LAMBDA**ratio)


@pytest.mark.parametrize('mutate', [lambda c:c.pop('sample_budget'),
    lambda c:c.update(discount_time_constant_s=float('nan')),
    lambda c:c.update(gae_residual_time_constant_s=True),
    lambda c:c.update(rollout_steps=64.), lambda c:c.update(policy_dt_s=.02),
    lambda c:c.update(discount_time_constant_s=20.),
    lambda c:c.update(gae_residual_time_constant_s=10.),
    lambda c:c.update(termination='longer'), lambda c:c.update(extra=True)])
def test_partial_changed_and_nonfinite_contracts_are_rejected(mutate):
    c = contract(); mutate(c)
    with pytest.raises(ValueError): validate(c)


@pytest.mark.parametrize('dt', [0., -1., float('nan'), float('inf'), True, '.01'])
def test_invalid_period_is_rejected(dt):
    with pytest.raises(ValueError): coefficients(contract(), dt)


@pytest.mark.parametrize('change', [lambda p:p.update(schema='wheel-leg-prone-motion-v1'),
    lambda p:p['global_curriculum'].pop('recovery_focus'),
    lambda p:p.update(multi_pose_recovery=None), lambda p:p.update(jump_training={}),
    lambda p:p.update(rollout_steps=128), lambda p:p['timing'].update(policy_dt=.02),
    lambda p:p['discounts'].update(gamma=.99)])
def test_incompatible_task_or_declared_coefficients_fail_validation(change):
    p=deepcopy(PLAN); change(p)
    with pytest.raises(ValueError): task.validate(p)


@pytest.mark.parametrize('options', [dict(num_steps_per_env=128),
    dict(physics_hz=200), dict(physics_hz=200, policy_hz=50), dict(pd_hz=1000)])
def test_runtime_cannot_silently_change_the_horizon_comparison_grid(options):
    with pytest.raises(ValueError): resolve_plan(PLAN, **options)


def test_finite_budget_and_learner_differ_only_in_gamma_and_lambda():
    options=dict(num_envs=64, max_iterations=250, save_interval=50, seed=261105,
        physics_hz=1000, policy_hz=100, pd_hz=200, domain_randomization='off')
    new, old = (resolve_plan(p, **options) for p in (PLAN, BASE))
    assert task.validate(new) == new
    a, b = (learner_config(p, 'cpu') for p in (new, old))
    for name in ('gamma', 'lam'): b['algorithm'][name] = a['algorithm'][name]
    assert a == b
    assert a['max_iterations'] == 250 and a['num_steps_per_env'] == 64
    assert new['num_envs']*new['rollout_steps']*new['max_updates'] == 1024000
    assert a['resume'] is False


def test_native_configuration_and_potential_use_the_same_learner_discount(tmp_path):
    p=resolve_plan(PLAN, num_envs=64, max_iterations=250, domain_randomization='off')
    old=resolve_plan(BASE, num_envs=64, max_iterations=250, domain_randomization='off')
    cfg, settings, _ = configuration(tmp_path, p, device='cpu')
    learner = learner_config(p, 'cpu')
    saved=json.loads((tmp_path/'task_contract.json').read_text())
    assert settings['reward_gamma'] == saved['discounts']['gamma'] == learner['algorithm']['gamma']
    assert saved['discounts']['lam'] == learner['algorithm']['lam']
    assert saved['timing'] == BASE['timing'] and cfg['num_steps_per_env'] == 64
    assert settings['prone_motion_task']['multi_pose_recovery'] == old['multi_pose_recovery']
    assert settings['prone_motion_task']['recovery_impact'] == old['recovery_impact']
    assert settings['prone_motion_task']['evaluation'] == old['evaluation']
    hashes=source_hashes(PLAN_FILE, p)
    assert 'src/wheeled_tasks/chassis/recovery_return_horizon.py' in hashes
    assert 'contracts/wheel_leg_self_righting_return_horizon_v1.json' in hashes


def test_bootstrap_and_dense_potential_are_retained_in_a_short_rollout():
    # Independent backward TD-residual recursion, including a nonzero terminal
    # bootstrap: the configured weights do not create a hard one-second cutoff.
    c=task.discounts(PLAN); values=[.2, .4, .7, 1.1]; rewards=[.01, .03, .02]
    deltas=[rewards[i]+c['gamma']*values[i+1]-values[i] for i in range(3)]
    recursive=0.
    for delta in reversed(deltas): recursive=delta+c['gamma']*c['lam']*recursive
    unrolled=sum(d*math.exp(-i*.01) for i,d in enumerate(deltas))
    assert math.isclose(recursive, unrolled, rel_tol=1e-14)
    assert c['gamma']*c['lam'] > 0. and math.exp(-5.) > 0.

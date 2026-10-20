"""The 200 Hz policy must retain physical horizons and reject partial edits."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.flat_base_continuation import read_plan, write_bundle
from wheeled_tasks.chassis.flat_command_reference import RmcsFlatCommandReference, install_training_reference
from wheeled_tasks.chassis.flat_policy_rate import DEFAULT_PLAN, validate_policy_rate_transition
from wheeled_tasks.chassis.preflight import SubstepControlAudit, design_preflight
from wheeled_tasks.chassis.rewards import reward_terms

ROOT = Path(__file__).resolve().parents[1]


def plans():
    new = read_plan(ROOT, DEFAULT_PLAN)
    old = read_plan(ROOT, new['policy_rate_transition']['parent_plan_file'])
    return old, new


def test_physical_discount_gae_rollout_and_control_horizons_are_preserved():
    old, new = plans()
    receipt = validate_policy_rate_transition(ROOT, old, new, old['control_sha256'])
    assert receipt['policy_hz'] == 200 and receipt['decimation'] == 5
    assert new['num_steps_per_env'] == 384
    assert new['num_steps_per_env'] * new['policy_dt'] == old['num_steps_per_env'] * old['policy_dt'] == 1.92
    for seconds in (.02, .5, 1., 20.):
        for factor in ('gamma', 'lam'):
            assert new[factor]**round(seconds/.005) == pytest.approx(old[factor]**round(seconds/.02), abs=1e-13)
    for key in ('episode_seconds', 'group_episode_seconds', 'train_groups', 'stationary_tracking',
                'learning_rate', 'critic_learning_rate', 'contact_material', 'teacher_anchor'):
        assert new.get(key) == old.get(key)
    assert receipt['new_optimizer_updates'] == 0


@pytest.mark.parametrize('field,value', [('policy_dt', .02), ('gamma', .99), ('lam', .95),
    ('num_steps_per_env', 96), ('learning_rate', .01), ('episode_seconds', 5.)])
def test_partial_frequency_edits_or_other_objective_changes_are_rejected(tmp_path, field, value):
    _, new = plans()
    new[field] = value
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(new))
    with pytest.raises(ValueError):
        read_plan(ROOT, path)


@pytest.mark.parametrize('evaluation,count', [(False, 4096), (True, 32)])
def test_independent_worker_design_and_native_pd_audit_agree(tmp_path, evaluation, count):
    config, seed = write_bundle(ROOT, tmp_path/'config', count, evaluation=evaluation)
    control = load_chassis_control(ROOT/config['control_math_source'])
    assert design_preflight(config, control, ROOT, stage='speed', role='quick' if evaluation else 'train',
        num_envs=count, seed=seed)['status'] == 'static_passed'
    assert config['action_difference_reference_dt'] == .02
    assert config['policy_dt'] == .005
    audit = SubstepControlAudit(.001, .005)
    audit.begin_policy(); audit.targets_decoded()
    for _ in range(5):
        audit.feedback_control(); audit.physics_updated(.001)
    observed = audit.end_policy()
    assert observed['target_decodes'] == 1
    assert observed['feedback_refreshes'] == observed['pd_updates'] == 5
    bad = deepcopy(config); bad['policy_dt'] = .02
    with pytest.raises(ValueError, match='Preflight rejected'):
        design_preflight(bad, control, ROOT, stage='speed', role='quick' if evaluation else 'train',
            num_envs=count, seed=seed)


def test_action_rate_and_second_difference_have_same_density_at_both_policy_rates():
    # Samples from a smooth quadratic trajectory have known velocity and
    # acceleration at t=0; finite-difference scaling must not dilute their cost.
    results = []
    for dt in (.02, .005):
        value = lambda t: torch.full((1, 6), 2*t + 3*t*t, dtype=torch.float64)
        zero = torch.zeros(1,3,dtype=torch.float64)
        result = reward_terms(zero, zero, torch.tensor([[0.,0.,-1.]],dtype=torch.float64),
            torch.tensor([.305],dtype=torch.float64),torch.tensor([[0.,0.,.305]],dtype=torch.float64),
            torch.zeros(1,6),torch.zeros(1,6),torch.zeros(1,6),value(dt),value(0),value(-dt),
            torch.zeros(1,2,3),torch.ones(1),torch.ones(1),torch.zeros(1,dtype=torch.bool),
            action_difference_scale=.02/dt)
        results.append(result)
    for name in ('leg_action_smoothness','wheel_action_smoothness'):
        torch.testing.assert_close(results[0][name], results[1][name], rtol=1e-12, atol=1e-12)
    # A linear action trajectory has exactly equal first-difference density.
    old_delta = (2*.02)*(.02/.02)
    new_delta = (2*.005)*(.02/.005)
    assert new_delta == old_delta


def test_height_reference_preserves_speed_and_advances_once_per_200hz_tick():
    _, plan = plans()
    contract = plan['training_command_reference']['contract']
    assert contract['policy_hz'] == 200 and contract['source_policy_hz'] == 50
    reference = RmcsFlatCommandReference(1, 'cpu', contract)
    request = torch.tensor([[0.,0.,.43]])
    for tick in range(12):
        result = reference.update(request, torch.zeros(1), torch.tensor([tick])).clone()
        assert torch.equal(result, reference.update(request, torch.zeros(1), torch.tensor([tick])))
    assert float(result[0,2]) == pytest.approx(.305 + .06*(2/3), abs=1e-6)
    env = SimpleNamespace(num_envs=1, device='cpu', policy_dt=.005, references=None,
        cfg=dict(flat_base_continuation=True,evaluation_exact_cases=False,physics_dt=.001,command_slew=True),
        commands=torch.tensor([[0.,0.,.305]]),command_target=torch.zeros(1,2),
        episode_length_buf=torch.zeros(1,dtype=torch.long),command_acceleration=torch.zeros(1),
        reset=lambda ids: None,get_observations=lambda: None,
        skills=SimpleNamespace(command_base=request,batches=[(torch.tensor([0]),dict(kind='stand',group_name='stand'))],
            yaw=lambda: torch.zeros(1)))
    report = install_training_reference(env,plan['training_command_reference'])
    assert report['policy_hz'] == 200 and report['physics_hz'] == 1000
    env.get_observations()
    assert float(env.commands[0,2]) == pytest.approx(.305 + (2/3)/200,abs=1e-6)

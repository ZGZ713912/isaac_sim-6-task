"""Strict local planar-only migration, real sampler, and no budget/clock escape."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from wheeled_tasks.chassis import local_scratch as local
from wheeled_tasks.chassis.local_command_revision import REVISION, validate_contracts, validate_parent, validate_plan
from wheeled_tasks.chassis.locomotion_difficulty import (
    difficulty_envelope, difficulty_settings, local_rehearsal_settings, validate_difficulty,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def target():
    with TemporaryDirectory(prefix='command-rehearsal-test-', dir=ROOT / 'runs') as directory:
        path = local.write_bundle(ROOT, Path(directory) / 'config', 512, plan_file=local.REHEARSAL_PLAN_FILE)
        yield json.loads(path.read_text())


def test_rule_only_changes_planar_caps_and_old_rule_remains_strict():
    old, new = difficulty_settings(0), local_rehearsal_settings(0)
    assert validate_difficulty(old) == old and validate_difficulty(new) == new
    restored = deepcopy(new)
    restored['schema'] = old['schema']
    for node, previous in zip(restored['keyframes'], old['keyframes']):
        node.update(vx_max_m_s=previous['vx_max_m_s'], yaw_max_rad_s=previous['yaw_max_rad_s'])
    assert restored == old
    for clock in (0, 60, 120, 240, 360, 2176, 2304, 18912):
        a, b = difficulty_envelope(old, clock), difficulty_envelope(new, clock)
        assert b['vx_max_m_s'] == .5 and b['yaw_max_rad_s'] == 1.
        assert a['motion_height_range_m'] == b['motion_height_range_m'] and a['grade'] == b['grade']
    with pytest.raises(ValueError):
        validate_difficulty(new | {'schema': old['schema']})


@pytest.mark.parametrize('fault', ['origin', 'height', 'cap', 'disabled', 'extra'])
def test_named_rule_cannot_smuggle_changes(fault):
    settings = local_rehearsal_settings(0)
    if fault == 'origin': settings['origin_actor_update'] = 2176
    elif fault == 'height': settings['keyframes'][-1]['motion_height_range_m'] = [.28, .35]
    elif fault == 'cap': settings['keyframes'][0]['vx_max_m_s'] = 1.
    elif fault == 'disabled': settings['enabled'] = False
    else: settings['allow_clip'] = True
    with pytest.raises(ValueError):
        validate_difficulty(settings)


def test_actual_sampler_caps_both_directions_retains_heights_rng_and_exact_bypass(monkeypatch):
    import test_locomotion_difficulty as helper
    monkeypatch.setattr(helper, 'difficulty_settings', local_rehearsal_settings)
    env = helper.environment(actor=2176, origin=0)
    assert env.skills.verify_difficulty_samples()['sample_check'] == 'passed'
    base = env.skills.command_base
    assert base[:, 0].abs().max() <= .500001 and base[:, 1].abs().max() <= 1.000001
    for name, vx in [('forward_5', .5), ('backward_5', -.5), ('forward_05', .5)]:
        actual = base[helper.members(env, name), 0]
        torch.testing.assert_close(actual, torch.full_like(actual, vx))
    for mm in (230, 305, 430):
        torch.testing.assert_close(base[helper.members(env, f'stand_{mm}mm'), 2],
            torch.full_like(base[helper.members(env, f'stand_{mm}mm'), 2], mm / 1000))
    ordinary = helper.environment(actor=2176, origin=0, legacy=True)
    assert torch.equal(env.generator.get_state(), ordinary.generator.get_state())
    exact = helper.environment(actor=2176, origin=0, exact=True)
    assert exact.skills.verify_difficulty_samples()['sample_check'] == 'exact_evaluation_bypass'
    assert (exact.commands[helper.members(exact, 'forward_5'), 0] == 5).all()


def test_independent_design_and_only_allowed_contract_diff(target):
    _, old = local.build_bundle(ROOT, ROOT / 'runs/source-rehearsal-fixture', 512, plan_file=local.BIRTH_RAMP_PLAN_FILE)
    proof = validate_contracts(old['worker'], target, ROOT)
    assert proof['additional_critic_warmup'] == 0
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    receipt = design_preflight(target, load_chassis_control(ROOT / target['control_math_source']), ROOT,
        stage='speed', role='train', seed=290927, num_envs=512, block_index=0)
    assert receipt['budget']['inherited_consumption'] == 1088
    assert receipt['budget']['update_ceiling'] == 18912
    assert receipt['runtime_expected']['locomotion_difficulty']['settings'] == local_rehearsal_settings(0)
    assert old['worker']['nonstandard_start'] == target['nonstandard_start']


@pytest.mark.parametrize('key,value', [('entropy_coef', .005), ('position_iterations', 128),
    ('gamma', .99), ('target_num_envs', 1024), ('initial_noise_std', .5),
    ('curriculum_critic_warmup_updates', 10)])
def test_contract_changes_outside_declared_caps_rejected(target, key, value):
    _, old = local.build_bundle(ROOT, ROOT / 'runs/source-rehearsal-fixture', 512, plan_file=local.BIRTH_RAMP_PLAN_FILE)
    with pytest.raises(ValueError): validate_contracts(old['worker'], target | {key: value}, ROOT)
    with pytest.raises(ValueError): validate_contracts(old['worker'] | {key: value}, target, ROOT)


@pytest.mark.parametrize('key,value', [('update_ceiling', 20000), ('prior_experiment_consumption', 0),
    ('initialization', 'random_actor_critic_fresh_optimizer'), ('curriculum_origins', {'locomotion_difficulty': 2176})])
def test_plan_cannot_refund_or_reset_clocks(key, value):
    plan = deepcopy(local.read_plan(ROOT, local.REHEARSAL_PLAN_FILE))
    with pytest.raises(ValueError): validate_plan(plan | {key: value}, ROOT)


def test_real_named_parent_first128_and_same_contract_only_afterward(target):
    from test_local_scratch import invocation
    assert validate_parent(REVISION['checkpoint_file'], target, ROOT)['total_prior_consumption'] == 3264
    args = invocation()
    with pytest.raises(ValueError, match='full-state continuation'): local.validate_invocation(target, args, ROOT)
    args.resume = Path(REVISION['checkpoint_file'])
    args.resume_local_command_revision = True
    args.updates = 128
    assert local.validate_invocation(target, args, ROOT) == 2176
    args.updates = 129
    with pytest.raises(ValueError): local.validate_invocation(target, args, ROOT)
    args.updates = 128
    args.resume_local_command_revision = False
    with pytest.raises(ValueError): local.validate_invocation(target, args, ROOT)
    args.resume_local_command_revision = True
    args.resume_local_birth_revision = True
    with pytest.raises(ValueError): local.validate_invocation(target, args, ROOT)

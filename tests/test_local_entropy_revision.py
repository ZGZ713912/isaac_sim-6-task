"""Only the named entropy loss and complete fork accounting may change."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from wheeled_tasks.chassis import local_scratch as local
from wheeled_tasks.chassis.local_entropy_revision import REVISION, validate_contracts, validate_parent, validate_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def target():
    with TemporaryDirectory(prefix='entropy-ab-test-', dir=ROOT / 'runs') as directory:
        path = local.write_bundle(ROOT, Path(directory) / 'config', 512, plan_file=local.ENTROPY_AB_PLAN_FILE)
        yield json.loads(path.read_text())


def source():
    return json.loads((Path(REVISION['checkpoint_file']).parent / 'contract.json').read_text())


def test_independent_design_only_entropy_and_budget_bindings(target):
    previous = source()
    receipt = validate_contracts(previous, target, ROOT)
    assert receipt['additional_critic_warmup'] == 0
    assert target['entropy_coef'] == .0005 and previous['entropy_coef'] == 0.
    assert target['nonstandard_start'] == previous['nonstandard_start']
    assert target['locomotion_difficulty'] == previous['locomotion_difficulty']
    assert target['initial_noise_std_policy_order'] == previous['initial_noise_std_policy_order']
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    preflight = design_preflight(target, load_chassis_control(ROOT / target['control_math_source']), ROOT,
        stage='speed', role='train', seed=290927, num_envs=512, block_index=0)
    assert preflight['budget']['inherited_consumption'] == 1216
    assert preflight['budget']['update_ceiling'] == 18784
    assert 2560 + 1216 + 128 == 3904


@pytest.mark.parametrize('field,value', [('entropy_coef', .005), ('position_iterations', 128),
    ('policy_dt', .02), ('physics_dt', .001), ('gamma', .99), ('target_num_envs', 1024),
    ('initial_noise_std_policy_order', [.2] * 6), ('curriculum_critic_warmup_updates', 10),
    ('nonstandard_start', {}), ('locomotion_difficulty', {}), ('termination_event_cost', 4.)])
def test_other_contract_changes_rejected(target, field, value):
    with pytest.raises(ValueError): validate_contracts(source(), target | {field: value}, ROOT)
    with pytest.raises(ValueError): validate_contracts(source() | {field: value}, target, ROOT)


@pytest.mark.parametrize('field,value', [('entropy_coef', 0.), ('update_ceiling', 18912),
    ('prior_experiment_consumption', 1088), ('prior_experiment_receipt_file', 'reports/local_training_20261017/pilot005.json'),
    ('initialization', 'random_actor_critic_fresh_optimizer'), ('curriculum_origins', {'locomotion_difficulty': 2560})])
def test_plan_cannot_reset_or_refund_control(field, value):
    plan = deepcopy(local.read_plan(ROOT, local.ENTROPY_AB_PLAN_FILE))
    with pytest.raises(ValueError): validate_plan(plan | {field: value}, ROOT)


def test_parent_is_exact_common2560_and_not_control_endpoint(target):
    proof = validate_parent(REVISION['checkpoint_file'], target, ROOT)
    assert proof['parent_updates'] == 2560
    assert proof['total_prior_consumption'] == 3776
    with pytest.raises(ValueError): validate_parent(REVISION['control']['checkpoint_file'], target, ROOT)
    with pytest.raises(ValueError): validate_parent(Path(REVISION['checkpoint_file']).parent / 'initial_state/model_final.pt', target, ROOT)


def test_first_block_bounded_full_restore_and_no_fresh_start(target):
    from test_local_scratch import invocation
    args = invocation()
    args.updates = 128
    with pytest.raises(ValueError, match='full-state continuation'): local.validate_invocation(target, args, ROOT)
    args.resume = Path(REVISION['checkpoint_file'])
    args.resume_local_entropy_revision = True
    args.updates = 128
    assert local.validate_invocation(target, args, ROOT) == 2560
    args.updates = 129
    with pytest.raises(ValueError): local.validate_invocation(target, args, ROOT)
    args.updates = 128
    args.resume_local_entropy_revision = False
    with pytest.raises(ValueError): local.validate_invocation(target, args, ROOT)
    args.resume_local_entropy_revision = True
    args.resume_local_command_revision = True
    with pytest.raises(ValueError): local.validate_invocation(target, args, ROOT)


@pytest.mark.parametrize('field,value', [('consumed_updates', 0), ('learning_rate_scale', 1.),
    ('stage_actor_update', 0), ('actor_updates_consumed', 0), ('global_actor_update_offset', 1)])
def test_invocation_rejects_resets_and_external_budget(target, field, value):
    from test_local_scratch import invocation
    args = invocation()
    args.resume = Path(REVISION['checkpoint_file'])
    args.resume_local_entropy_revision = True
    args.updates = 128
    if field == 'actor_updates_consumed': value = 2560
    setattr(args, field, value)
    with pytest.raises(ValueError): local.validate_invocation(target, args, ROOT)


def test_authentication_rejects_corrupted_bound_control_evidence(target, monkeypatch):
    original = local.digest
    bad = REVISION['control']['evaluation_file']
    monkeypatch.setattr(local, 'digest', lambda path: '0' * 64 if str(path) == bad else original(path))
    with pytest.raises(ValueError, match='control evidence changed'):
        validate_parent(REVISION['checkpoint_file'], target, ROOT)

"""The real completed control, same parent and charged budget gate local v9."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from wheeled_tasks.chassis import local_scratch as local
from wheeled_tasks.chassis import local_value_revision as revision

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def target():
    with TemporaryDirectory(prefix='value-clip-test-', dir=ROOT/'runs') as directory:
        path = local.write_bundle(ROOT, Path(directory)/'config', 512, plan_file=local.VALUE_PLAN_FILE)
        yield json.loads(path.read_text())


def source():
    return json.loads((Path(revision.BASE['checkpoint_file']).parent/'contract.json').read_text())


def test_exact_independent_design_and_all_consumed_updates(target):
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    receipt = revision.validate_contracts(source(), target, ROOT)
    assert receipt['additional_critic_warmup'] == 0
    assert receipt['source_use_clipped_value_loss'] is True
    assert receipt['target_use_clipped_value_loss'] is False
    design = design_preflight(target, load_chassis_control(ROOT/target['control_math_source']), ROOT,
        stage='speed', role='train', seed=290927, num_envs=512, block_index=0)
    budget = design['budget']
    assert budget['inherited_consumption'] == 1344
    assert budget['update_ceiling'] == 18656
    assert budget['transition_ceiling'] == 182648832 + (18656-2945)*512*384 == 3271557120
    assert budget['transition_ceiling'] != 18656*512*384
    assert budget['inherited_consumption']+2945+128 == 4417
    assert budget['inherited_consumption']+budget['update_ceiling'] == 20000
    assert (target['num_steps_per_env'],target['lam'],target['gamma'],target['entropy_coef']) == (384,1.,.9974905699336811,.0005)
    for key in ('nonstandard_start','locomotion_difficulty','ordinary_posture_curriculum','v5_control'):
        assert source()[key] == target[key]


@pytest.mark.parametrize('field,value', [
    ('use_clipped_value_loss', True), ('use_clipped_value_loss', 0),
    ('value_loss_coef', 1.), ('max_grad_norm', 10.), ('clip_param', 1.),
    ('learning_rate', .001), ('learning_rate_schedule', 'adaptive'),
    ('num_steps_per_env', 96), ('lam', .9872585449014338), ('gamma', .99),
    ('entropy_coef', 0.), ('policy_dt', .02), ('physics_dt', .001),
    ('position_iterations', 128), ('target_num_envs', 1024),
    ('ppo_minibatch_samples', 12288), ('curriculum_critic_warmup_updates', 10),
    ('initial_noise_std_policy_order', [.2]*6), ('termination_event_cost', 4.),
    ('nonstandard_start', {}), ('locomotion_difficulty', {}),
])
def test_undeclared_algorithm_plant_reward_or_source_changes_rejected(target,field,value):
    with pytest.raises(ValueError):
        revision.validate_contracts(source(), target|{field:value}, ROOT)
    with pytest.raises(ValueError):
        revision.validate_contracts(source()|{field:value}, target, ROOT)


@pytest.mark.parametrize('field,value', [
    ('update_ceiling', 18784), ('prior_experiment_consumption', 1216),
    ('initialization', 'random_actor_critic_fresh_optimizer'),
    ('use_clipped_value_loss', 0), ('use_clipped_value_loss', True),
    ('entropy_coef', 0.), ('curriculum_origins', {'locomotion_difficulty':2945}),
    ('ppo_timing', {'num_steps_per_env':384,'lam':.99}),
])
def test_plan_cannot_refund_control_or_change_other_semantics(field,value):
    plan = deepcopy(local.read_plan(ROOT,local.VALUE_PLAN_FILE))
    with pytest.raises(ValueError): revision.validate_plan(plan|{field:value},ROOT)


def test_named_parent_is_common2945_never_control_endpoint(target):
    result = revision.validate_parent(revision.BASE['checkpoint_file'],target,ROOT)
    assert result['parent_updates'] == 2945 and result['total_prior_consumption'] == 4289
    assert result['parent_training_transitions'] == 182648832
    parent = Path(revision.BASE['checkpoint_file']).parent
    for wrong in (parent/'initial_state/model_final.pt', parent.parent/'pilot019/model_final.pt',
                  parent.parent/'pilot017/model_final.pt'):
        with pytest.raises(ValueError): revision.validate_parent(wrong,target,ROOT)


def test_first_named_block_requires_full_state_and128_ceiling(target):
    from test_local_scratch import invocation
    args = invocation(); args.updates=128
    with pytest.raises(ValueError, match='full-state continuation'):
        local.validate_invocation(target,args,ROOT)
    args.resume=Path(revision.BASE['checkpoint_file']);args.resume_local_value_revision=True
    assert local.validate_invocation(target,args,ROOT)==2945
    args.updates=129
    with pytest.raises(ValueError):local.validate_invocation(target,args,ROOT)
    args.updates=128;args.resume_local_value_revision=False
    with pytest.raises(ValueError):local.validate_invocation(target,args,ROOT)
    args.resume_local_value_revision=args.resume_local_return_revision=True
    with pytest.raises(ValueError,match='exactly one'):local.validate_invocation(target,args,ROOT)


@pytest.mark.parametrize('field,value', [
    ('consumed_updates',0), ('learning_rate_scale',1.), ('stage_actor_update',0),
    ('actor_updates_consumed',2945), ('global_actor_update_offset',1),
])
def test_external_clocks_and_budget_reset_rejected(target,field,value):
    from test_local_scratch import invocation
    args=invocation();args.resume=Path(revision.BASE['checkpoint_file'])
    args.resume_local_value_revision=True;args.updates=128
    setattr(args,field,value)
    with pytest.raises(ValueError):local.validate_invocation(target,args,ROOT)


@pytest.mark.parametrize('index',range(8))
def test_all_original_decision_and_actual_control_evidence_bound(index,monkeypatch):
    plan=local.read_plan(ROOT,local.VALUE_PLAN_FILE)
    row=plan['local_value_revision']['evidence'][index]
    original=local.digest
    monkeypatch.setattr(local,'digest',lambda path:'0'*64 if Path(path)==ROOT/row['file'] else original(path))
    with pytest.raises(ValueError):local.read_plan(ROOT,local.VALUE_PLAN_FILE)


def test_control_receipt_cannot_refund_partially_or_use_an_old_lineage(tmp_path,monkeypatch):
    path=ROOT/revision.CONSUMPTION_FILE
    original=json.loads(path.read_text())
    for field,value in [('consumption',1216),('total_all_experiments_before_treatment',4161),
                        ('lineage_update_ceiling',18784),('no_refund',False)]:
        candidate=tmp_path/(field+'.json')
        candidate.write_text(json.dumps(original|{field:value}))
        monkeypatch.setattr(revision,'CONSUMPTION_FILE',str(candidate))
        with pytest.raises(ValueError,match='refund or duplicate'):revision.revision_binding(ROOT)


@pytest.mark.parametrize('kind', ['design','budget'])
def test_generated_binding_cannot_be_rehashed_to_override_the_plan(target,kind):
    path=ROOT/Path(target['design_preflight']['file']).parent/(kind+'.json')
    value=json.loads(path.read_text())
    if kind=='budget': value['inherited_consumption']=1216
    else: value['stages']['speed']['roles']['train']['contract']['use_clipped_value_loss']=True
    path.write_text(json.dumps(value,indent=2)+'\n')
    changed=deepcopy(target)
    if kind=='budget':changed['budget_spec_sha256']=local.digest(path)
    else:changed['design_preflight']['sha256']=local.digest(path)
    with pytest.raises(ValueError):local.validate_config(changed,ROOT)

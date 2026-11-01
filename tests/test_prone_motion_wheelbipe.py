"""Explicit wheelbipe workload overrides preserve the recorded plant and clocks."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from train_prone_motion import configuration, learner_config, check_first_update
from wheeled_tasks.chassis.prone_motion import validate, timing
from wheeled_tasks.chassis.prone_motion_cli import resolve_plan, train_parser, training_options
from wheeled_tasks.chassis.control_contract import load_chassis_control

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4.json').read_text())


def test_wheelbipe_cli_reaches_control_contract_and_actual_learner(tmp_path):
    args = train_parser().parse_args(['--physics-hz','200','--policy-hz','50',
        '--num-steps-per-env','24','--position-iterations','12','--velocity-iterations','6',
        '--hidden-dims','512','256','128','--ppo-epochs','5','--ppo-mini-batches','4'])
    _,plan,_ = training_options(args,ROOT)
    cfg,settings,_ = configuration(tmp_path,plan)
    control = load_chassis_control(ROOT/cfg['control_math_source'])
    agent = learner_config(plan)
    assert validate(plan) == plan
    assert timing(plan) == control['timing'] == dict(physics_dt=.005,policy_dt=.02,pc_control_dt=.005,decimation=4)
    assert settings['sensor_policy_dt'] == .02
    assert cfg['num_steps_per_env'] == agent['num_steps_per_env'] == 24
    assert (cfg['position_iterations'],cfg['velocity_iterations']) == (12,6)
    assert agent['actor']['hidden_dims'] == agent['critic']['hidden_dims'] == [512,256,128]
    assert agent['algorithm']['num_learning_epochs'] == 5 and agent['algorithm']['num_mini_batches'] == 4
    assert agent['algorithm']['gamma'] == pytest.approx(.99) and agent['algorithm']['lam'] == pytest.approx(.95)
    assert not agent['resume'] and not plan['simulation_timing_approximation']['hardware_equivalence_verified']
    original = load_chassis_control(ROOT/'contracts/wheel_leg_prone_motion_control_pd200hz_v1.json')
    for k in original:
        if k not in ('timing','control_id','simulation_timing_approximation'): assert control[k] == original[k]
    for k in ('global_curriculum','multi_pose_recovery','recovery_impact','roles_per20','standing_odometry'):
        assert plan[k] == BASE[k]
    assert timing(plan)['decimation']*agent['num_steps_per_env'] == 96


@pytest.mark.parametrize('overrides', [dict(policy_hz=49),dict(policy_hz=True),dict(policy_hz=50),
    dict(policy_hz=50,physics_hz=1000),dict(hidden_dims=[]),dict(hidden_dims=[0,32]),
    dict(hidden_dims=[True]),dict(ppo_epochs=0),dict(ppo_epochs=1.5),dict(ppo_mini_batches=0),
    dict(ppo_mini_batches=7),dict(physics_hz=200,position_iterations=12,velocity_iterations=6)])
def test_invalid_or_unacknowledged_rate_and_learning_overrides_fail_closed(overrides):
    with pytest.raises(ValueError): resolve_plan(BASE,**overrides)


def test_legacy_physics_options_revalidate_and_default_workload_stays_the_same():
    plan=resolve_plan(BASE,physics_hz=200)
    for key in ('policy_hz','hidden_dims','ppo_epochs','ppo_mini_batches'):
        del plan['runtime_cli']['options'][key]
    assert validate(plan)==plan
    agent=learner_config(plan)
    assert timing(plan)['decimation']==2 and plan['rollout_steps']==64
    assert agent['actor']['hidden_dims']==[256,128,64] and agent['algorithm']['num_learning_epochs']==3
    changed=deepcopy(resolve_plan(BASE,physics_hz=200,policy_hz=50,hidden_dims=[512,256,128]))
    changed['ppo']['hidden_dims'][0]=1024
    with pytest.raises(ValueError): validate(changed)


@pytest.mark.parametrize('epochs,mini_batches',[(3,4),(5,4),(2,8)])
def test_first_native_update_audit_uses_configured_optimizer_budget(epochs,mini_batches):
    cfg=learner_config(resolve_plan(BASE,ppo_epochs=epochs,ppo_mini_batches=mini_batches))
    recorded=dict(all_recorded_values_finite=True,committed_optimizer_calls=epochs*mini_batches,
        actor_changed_parameter_tensors=9,critic_changed_parameter_tensors=8)
    assert check_first_update(recorded,cfg)
    assert recorded['expected_optimizer_calls']==epochs*mini_batches
    assert all(recorded['checks'].values())
    for field,value in (('committed_optimizer_calls',epochs*mini_batches-1),
                        ('all_recorded_values_finite',False),
                        ('actor_changed_parameter_tensors',0),('critic_changed_parameter_tensors',0)):
        bad=deepcopy(recorded);bad[field]=value
        assert not check_first_update(bad,cfg)

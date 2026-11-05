"""Versioned rate comparison retains causal references and original boundaries."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from wheeled_tasks.chassis.recovery_actions import CONTRACT, RANGE_CONTRACT, targets, validate, EncoderIncrementActions
from wheeled_tasks.chassis.paired_feedback import paired_position_error
from wheeled_tasks.chassis.prone_motion import validate as validate_plan

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'contracts/wheel_leg_self_righting_encoder_increment_range_v1.json'
PLAN = json.loads(PATH.read_text())
OLD = json.loads((ROOT/'contracts/wheel_leg_self_righting_encoder_increment_v1.json').read_text())
LIMIT_CONFIG = PLAN['gas_spring_feedforward']['relative_soft_limits']
LIMITS = (LIMIT_CONFIG['minimum']+LIMIT_CONFIG['margin'], LIMIT_CONFIG['maximum']-LIMIT_CONFIG['margin'])


def test_only_versioned_rates_and_their_matching_metadata_change():
    assert validate_plan(PLAN) == PLAN
    assert validate_plan(OLD) == OLD
    changes = {key for key in PLAN if PLAN[key] != OLD[key]}
    assert changes == {'encoder_increment_actions','learning_interface','exploration_distribution'}
    assert {key for key in RANGE_CONTRACT if RANGE_CONTRACT[key] != CONTRACT[key]} == {
        'schema','common_rate_rad_s','relative_rate_rad_s'}
    assert PLAN['encoder_increment_actions'] == RANGE_CONTRACT
    for key in changes:
        broken=deepcopy(PLAN);broken[key]=OLD[key]
        with pytest.raises(ValueError):validate_plan(broken)
    broken=deepcopy(RANGE_CONTRACT);broken['common_rate_rad_s']=float('nan')
    with pytest.raises(ValueError):validate(broken)


@pytest.mark.parametrize('relative',[LIMITS[0]-.2,LIMITS[0],.4,LIMITS[1],LIMITS[1]+.2])
@pytest.mark.parametrize('turns',[-3,0,4])
def test_boundaries_winding_and_all_action_corners_use_declared_range(relative,turns):
    actions=torch.cartesian_prod(*([torch.tensor([-2.,0.,2.],dtype=torch.float64)]*4))
    q=torch.tensor([[3.1,3.1+relative,-3.1,-3.1-relative]],dtype=torch.float64).expand(len(actions),-1).clone()
    q+=q.new_tensor([[turns*math.tau,-2*math.tau,2*math.tau,turns*math.tau]])
    moved,raw=targets(actions,q,policy_dt=.01,relative_limits=LIMITS,config=RANGE_CONTRACT)
    error=paired_position_error(moved,q).unflatten(-1,(2,2))
    assert error.abs().max() <= .10+3e-14
    torch.testing.assert_close(error.mean(-1),raw.unflatten(-1,(2,2))[...,0]*.08,rtol=0,atol=3e-14)
    sign=q.new_tensor([1.,-1.])
    d0=sign*torch.atan2((q[:,[1,3]]-q[:,[0,2]]).sin(),(q[:,[1,3]]-q[:,[0,2]]).cos())
    d1=sign*(moved[:,[1,3]]-moved[:,[0,2]])
    assert (d1-d0).abs().max() <= .04+3e-14
    if relative < LIMITS[0]:assert (d1>d0).all()
    elif relative > LIMITS[1]:assert (d1<d0).all()
    else:assert (d1>=LIMITS[0]-3e-14).all() and (d1<=LIMITS[1]+3e-14).all()
    legacy,_=targets(actions,q,policy_dt=.01,relative_limits=LIMITS)
    explicit,_=targets(actions,q,policy_dt=.01,relative_limits=LIMITS,config=CONTRACT)
    assert torch.equal(legacy,explicit)


def test_adapter_passes_selected_range_and_never_accumulates_or_samples_again():
    from wheeled_tasks.chassis.getup_feedback import GetupFeedback
    settings=dict(paired_feedback_winding=True,cyclic_leg_targets=True,
        pd_feedback_period_ms=[1,1],pd_feedback_delay_ms=[0,0],pd_encoder_bias_rad=0.)
    env=SimpleNamespace(tick=0,num_envs=1,device='cpu',policy_dt=.01,decimation=10,getup_cfg=settings,
        domain=SimpleNamespace(enabled=torch.tensor([False]),rng=torch.Generator().manual_seed(11)),
        control_audit=SimpleNamespace(control_stride=5),startup_report={},actions=torch.ones(1,6),
        quiet_standing=SimpleNamespace(update=lambda active,xy,target,velocity:target))
    env.getup_feedback=GetupFeedback(1,'cpu',env.domain.enabled,settings,env.domain.rng,dt=.001)
    env.v5=SimpleNamespace(LEGS=(0,1,3,4),WHEELS=(2,5),action_bounds=torch.tensor([16.,16.,9.,16.,16.,9.]),
        settings=dict(wheel_velocity_scale=10.),gas_compensation=SimpleNamespace(cfg=dict(relative_soft_limits=LIMIT_CONFIG)))
    adapter=EncoderIncrementActions(env,RANGE_CONTRACT)
    q=torch.tensor([[0.,.4,0.,0.,-.4,0.]])
    env.getup_feedback.reset(torch.tensor([0]),q,torch.zeros_like(q))
    raw=torch.tensor([[1.,1.,.3,-1.,1.,-.4]])
    before={k:v.clone() for k,v in vars(env.getup_feedback).items() if torch.is_tensor(v)}
    first=env.v5.decode(raw,torch.full_like(q,100.))
    second=env.v5.decode(raw,torch.full_like(q,-100.))
    assert all(torch.equal(a,b) for a,b in zip(first,second))
    torch.testing.assert_close(first[0]-q[:,[0,1,3,4]],torch.tensor([[.06,.10,-.06,-.10]]),rtol=0,atol=1e-7)
    assert all(torch.equal(getattr(env.getup_feedback,k),v) for k,v in before.items())
    assert env.getup_feedback.tick==0 and env.getup_feedback.dt==.005
    env._measured_action_history(torch.tensor([0]));assert not env.actions.any()
    assert adapter.config==RANGE_CONTRACT


def test_authenticated_geometry_stays_in_40_to_110_degree_mechanical_domain():
    from wheeled_algo.encoder_leg_geometry import EncoderLegGeometry
    table=json.loads((ROOT/PLAN['standing_odometry']['geometry_file']).read_text())
    geometry=EncoderLegGeometry(table,expected_spec_sha256=PLAN['standing_odometry']['geometry_model_spec_sha256'])
    relative=torch.linspace(LIMITS[0],LIMITS[1],257)
    hip=torch.linspace(-4*math.pi,4*math.pi,257)
    q=torch.stack((hip,hip+relative,-hip,-hip-relative),-1)
    raw=torch.rand(257,4,generator=torch.Generator().manual_seed(29))*6.-3.
    moved,_=targets(raw,q,policy_dt=.01,relative_limits=LIMITS,config=RANGE_CONTRACT)
    state=geometry(moved,torch.zeros_like(moved))
    assert state['valid'].all() and state['knee_deg'].min()>=40. and state['knee_deg'].max()<=110.
    held,_=targets(torch.tensor([[.8,0.,-.6,0.]]).expand_as(q),q,policy_dt=.01,relative_limits=LIMITS,config=RANGE_CONTRACT)
    before=geometry(q,torch.zeros_like(q));after=geometry(held,torch.zeros_like(q))
    torch.testing.assert_close(after['knee_deg'],before['knee_deg'],rtol=0,atol=2e-4)
    torch.testing.assert_close(after['leg_length_m'],before['leg_length_m'],rtol=0,atol=2e-6)
    assert (after['wheel_body_m']-before['wheel_body_m']).abs().max()>.001


def test_new_range_keeps_the_same_fresh_raw_network_and_empty_optimizer():
    from train_prone_motion import learner_config,configuration,source_hashes
    from wheeled_algo.prone_initialization import initialize_prone_fresh
    from rsl_rl.algorithms import PPO
    obs=TensorDict(dict(policy=torch.randn(32,35),critic=torch.randn(32,87)),batch_size=[32])
    outputs=[]
    for plan in (OLD,PLAN):
        cfg=learner_config(plan,'cpu');cfg['multi_gpu']=None
        torch.manual_seed(107)
        algorithm=PPO.construct_algorithm(obs,SimpleNamespace(num_envs=32,num_actions=6),cfg,'cpu')
        receipt=initialize_prone_fresh(algorithm,torch.ones(6)*12.,plan)
        outputs.append(algorithm.actor(obs).detach())
        assert receipt['output_bias_policy_order']==[0.]*6 and not algorithm.optimizer.state
        assert not receipt['checkpoint_loaded'] and not receipt['teacher_loaded']
        assert receipt['learning_interface']==plan['learning_interface']
        algorithm.actor.distribution.update(outputs[-1])
        torch.testing.assert_close(algorithm.actor.distribution.std[0],torch.tensor(plan['initial_std']))
    assert torch.equal(*outputs)
    hashes=source_hashes(PATH,PLAN)
    assert 'tests/test_increment_range.py' in hashes and str(PATH.relative_to(ROOT)) in hashes

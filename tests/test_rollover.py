"""Goal semantics, independent leg preparation, curriculum and integration."""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis import rollover as r

ROOT = Path(__file__).resolve().parents[1]


def condition(**overrides):
    values = dict(gravity=torch.tensor([[0.,0.,-1.]]), height=torch.tensor([.2]),
        linear_velocity=torch.zeros(1,3), omega=torch.zeros(1,3), motor_velocity=torch.zeros(1,6),
        support_n=torch.tensor([200.]), weight_n=torch.tensor([250.]))
    values.update(overrides)
    return bool(r.prone_condition(**values))


def test_legs_can_support_prone_without_wheel_contact():
    assert condition()
    assert "wheel_contact" not in __import__('inspect').signature(r.prone_condition).parameters
    assert not condition(gravity=torch.tensor([[0.,0.,1.]]))
    assert not condition(height=torch.tensor([.305]))


@pytest.mark.parametrize('name,value', [
    ('linear_velocity',torch.tensor([[.1,0.,0.]])),
    ('linear_velocity',torch.tensor([[0.,0.,-.09]])),
    ('omega',torch.tensor([[0.,.4,0.]])),
    ('motor_velocity',torch.tensor([[0.,0.,1.1,0.,0.,0.]])),
    ('support_n',torch.tensor([0.])), ('height',torch.tensor([.3]))])
def test_capture_rejects_motion_and_airborne_and_standing(name,value):
    assert not condition(**{name:value})


def test_110_is_not_a_hard_capture_equality():
    assert 'knee_deg' not in __import__('inspect').signature(r.prone_condition).parameters
    assert r.CONTRACT['knee_is_soft_guidance_not_completion']


def test_physical_common_coordinates_ignore_shared_turns():
    q=torch.tensor([[-.42,1.00828,.42,-1.00828]],dtype=torch.float64)
    dq=torch.tensor([[3.,3.,-3.,-3.]],dtype=torch.float64)
    common,relative,rate=r.leg_coordinates(q,dq)
    assert torch.allclose(common[:,0],common[:,1])
    assert torch.allclose(relative[:,0],relative[:,1])
    assert torch.allclose(rate,torch.tensor([[3.,3.]],dtype=q.dtype))
    shifted=q+torch.tensor([[4*math.pi,4*math.pi,-6*math.pi,-6*math.pi]])
    assert torch.allclose(r.leg_coordinates(shifted,dq)[0],common,atol=2e-6)


def test_prefix_prepares_different_legs_while_already_rotating():
    q=torch.tensor([[-1.2,-.7,.2,-.8]])
    raw=r.prefix_action(q,torch.tensor([[0.,0.,1.]]),torch.tensor([1.428,1.428]),r.CONTRACT,policy_dt=.01)
    assert raw.shape==(1,4)
    assert raw[0,0]>0 and raw[0,2]<0  # mirrored numerical axes, same physical sweep
    assert not torch.isclose(raw[0,0],-raw[0,2])  # independent alignment correction
    assert raw.abs().max()<=1.


def test_prefix_relaxes_shape_and_stops_sweep_when_normally_up():
    q=torch.tensor([[-1.2,-.7,.2,-.8]])
    raw=r.prefix_action(q,torch.tensor([[0.,0.,-1.]]),torch.tensor([1.428,1.428]),r.CONTRACT,policy_dt=.01)
    assert not bool(raw.any())


def fixture_scores(gravity,common=None):
    n=len(gravity)
    return r.scores(gravity,torch.full((n,),.2),torch.zeros(n,3),torch.zeros(n,3),
        torch.full((n,2),110.),torch.zeros(n,2) if common is None else common,
        torch.zeros(n,2),torch.ones(n,dtype=torch.bool))


def test_orientation_progress_does_not_wait_for_leg_alignment():
    g=torch.tensor([[0.,0.,1.],[0.,.8,.6],[0.,0.,-1.]])
    s=fixture_scores(g,torch.tensor([[0.,math.pi]]*3))
    phi=r.potential(s)
    assert phi[0]<phi[1]<phi[2]
    assert s['shape'][-1]==1. and s['inverted'][-1]==0.


def test_potential_is_bounded_and_does_not_pay_winding_distance():
    n=1000
    theta=torch.linspace(-math.pi,math.pi,n)
    g=torch.stack((theta.sin(),torch.zeros(n),-theta.cos()),-1)
    s=fixture_scores(g)
    phi=r.potential(s)
    assert phi.min()>=0 and phi.max()<=13.
    assert torch.allclose(phi[0],phi[-1])


def test_normal_support_not_equivalent_to_excess_impact():
    s=fixture_scores(torch.tensor([[0.,0.,-1.]]))
    terms=r.densities(s,torch.zeros(1,3),torch.zeros(1,6),torch.zeros(1,6),
        torch.zeros(1),torch.tensor([[0.,0.,4.,0.]]),torch.tensor([250.]))
    assert terms['impact_excess'].item()==0.
    terms2=r.densities(s,torch.zeros(1,3),torch.zeros(1,6),torch.zeros(1,6),
        torch.zeros(1),torch.tensor([[0.,0.,14.,0.]]),torch.tensor([250.]))
    assert terms2['impact_excess'].item()<0.


def test_course_needs_actual_current_birth_outcomes():
    c=r.PrefixCurriculum(12)
    assert [c.sample(i)[0] for i in range(4)]==[0.,1.7,1.7,1.7]
    for _ in range(100): c.observe(-1,True)
    assert c.phase==0 and len(c.window)==0
    for _ in range(63): c.observe(0,True)
    assert c.phase==0
    c.observe(0,True)
    assert c.phase==1 and len(c.window)==0
    for _ in range(100): c.observe(0,True)
    assert c.phase==1 and len(c.window)==0


def test_course_failure_denominator_and_roundtrip():
    c=r.PrefixCurriculum(7)
    for success in [False]*20+[True]*44: c.observe(0,success)
    assert c.phase==0
    saved=c.state_dict()
    d=r.PrefixCurriculum(99); d.load_state_dict(saved)
    assert d.state_dict()==saved
    assert d.rng.random()==c.rng.random()


@pytest.mark.parametrize('field,value',[('tilt_deg',30.),('wheel_contact_required',True),('capture_hold_s',.2)])
def test_formal_goal_cannot_drift_silently(field,value):
    config=deepcopy(r.CONTRACT); config[field]=value
    with pytest.raises(ValueError): r.validate(config)


def test_plan_validation_and_original_admission_sources_unchanged():
    from wheeled_tasks.chassis.prone_motion import validate
    p=json.loads((ROOT/'contracts/wheel_leg_rollover_learning_v1.json').read_text())
    assert validate(p)['rollover_training']==r.CONTRACT
    assert p['completion_hold_seconds']==.01


def test_prefix_release_contains_no_physical_or_controller_reset():
    tree=ast.parse((ROOT/'src/wheeled_tasks/chassis/rollover_env.py').read_text())
    release=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='_release_prefix')
    calls=[x.func.attr for x in ast.walk(release) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)]
    assert not any(name.startswith(('write_','reset')) for name in calls)
    text=ast.unparse(release)
    for name in ['wheel_delay','motor_filter','encoder_q','domain_delay','sensor_history','gimbal_phase']:
        assert name in text


def test_training_selects_scoped_environment_and_probe_and_saves_course():
    text=(ROOT/'scripts/train_prone_motion.py').read_text()
    for required in ['rollover_env import RolloverEnv','probe_rollover import probe',
                     'rollover_curriculum=', 'prefix_release_integrity=']:
        assert required in text
    env=(ROOT/'src/wheeled_tasks/chassis/rollover_env.py').read_text()
    assert 'info["policy_learning_mask"] &= ~prefix' in env
    assert 'info["policy_bootstrap_mask"]' in env

"""Motor-coordinate recovery actions, causal feedback and fresh PPO semantics."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from wheeled_tasks.chassis.recovery_actions import contract, targets, EncoderIncrementActions, validate
from wheeled_tasks.chassis.paired_feedback import paired_position_error
from wheeled_tasks.chassis.prone_motion import validate as validate_plan
from wheeled_tasks.chassis.getup_feedback import GetupFeedback
from wheeled_algo.prone_initialization import initialize_prone_fresh

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'contracts/wheel_leg_self_righting_encoder_increment_v1.json'
PLAN = json.loads(PATH.read_text())
LIMITS_CFG = PLAN['gas_spring_feedforward']['relative_soft_limits']
LIMITS = (LIMITS_CFG['minimum']+LIMITS_CFG['margin'], LIMITS_CFG['maximum']-LIMITS_CFG['margin'])


def test_common_rotation_preserves_relative_angle_and_signs_match_both_sides():
    q = torch.tensor([[1., 1.4, -1., -1.4]], dtype=torch.float64)
    moved, _ = targets(torch.tensor([[1., 0., -.5, 0.]]), q, policy_dt=.01, relative_limits=LIMITS)
    torch.testing.assert_close(moved-q, q.new_tensor([[.02,.02,-.01,-.01]]), rtol=0, atol=1e-14)
    moved, _ = targets(torch.tensor([[0., 1., 0., 1.]]), q, policy_dt=.01, relative_limits=LIMITS)
    torch.testing.assert_close(moved-q, q.new_tensor([[-.01,.01,.01,-.01]]), rtol=0, atol=1e-14)


@pytest.mark.parametrize('turns', [-4,-1,0,1,5])
def test_independent_encoder_seams_use_one_shared_hip_chart(turns):
    q = torch.tensor([[3.1,3.5,-3.1,-3.5]], dtype=torch.float64)
    raw = torch.tensor([[.7,-.3,-.2,.6]], dtype=torch.float64)
    expected, _ = targets(raw, q, policy_dt=.01, relative_limits=LIMITS)
    shifted = q+q.new_tensor([[turns*math.tau,-2*math.tau,2*math.tau,turns*math.tau]])
    actual, _ = targets(raw, shifted, policy_dt=.01, relative_limits=LIMITS)
    torch.testing.assert_close(actual, expected+q.new_tensor([[turns*math.tau]*2+[2*math.tau]*2]),
                               rtol=0, atol=3e-14)
    torch.testing.assert_close(paired_position_error(actual, shifted),
                               paired_position_error(expected, q), rtol=0, atol=3e-14)


@pytest.mark.parametrize('relative', [LIMITS[0]-.2,LIMITS[0],.4,LIMITS[1],LIMITS[1]+.2])
def test_boundary_correction_is_inward_bounded_without_an_instant_projection(relative):
    q = torch.tensor([[0.,relative,0.,-relative]], dtype=torch.float64)
    for command in (-100.,0.,100.):
        moved, raw = targets(torch.full((1,4),command),q,policy_dt=.01,relative_limits=LIMITS)
        assert raw.abs().max() <= 1.
        assert paired_position_error(moved,q).abs().max() <= .03+1e-14
        result = moved[0,1]-moved[0,0]
        if LIMITS[0] <= relative <= LIMITS[1]:
            assert LIMITS[0]-1e-14 <= result <= LIMITS[1]+1e-14
        elif relative < LIMITS[0]:
            assert relative < result <= relative+.02+1e-14
        else:
            assert relative-.02-1e-14 <= result < relative


def test_time_scaling_changes_displacement_not_the_declared_target_rate():
    q = torch.tensor([[0.,.4,0.,-.4]],dtype=torch.float64)
    raw = torch.tensor([[.8,.4,-.2,-.7]],dtype=torch.float64)
    a,_=targets(raw,q,policy_dt=.01,relative_limits=LIMITS)
    b,_=targets(raw,q,policy_dt=.02,relative_limits=LIMITS)
    torch.testing.assert_close(b-q,2*(a-q),rtol=0,atol=1e-14)


def test_authenticated_full_vehicle_geometry_retains_mechanical_domain_and_common_knee_angle():
    from wheeled_algo.encoder_leg_geometry import EncoderLegGeometry
    table=json.loads((ROOT/PLAN['standing_odometry']['geometry_file']).read_text())
    geometry=EncoderLegGeometry(table,expected_spec_sha256=PLAN['standing_odometry']['geometry_model_spec_sha256'])
    generator=torch.Generator().manual_seed(29)
    relative=torch.linspace(LIMITS[0],LIMITS[1],257)
    hip=torch.linspace(-4*math.pi,4*math.pi,257)
    q=torch.stack((hip,hip+relative,-hip,-hip-relative),-1)
    raw=torch.rand(257,4,generator=generator)*6.-3.
    moved,_=targets(raw,q,policy_dt=.01,relative_limits=LIMITS)
    state=geometry(moved,torch.zeros_like(moved))
    assert state['valid'].all() and state['knee_deg'].min() >= 40. and state['knee_deg'].max() <= 110.
    held,_=targets(torch.tensor([[.8,0.,-.6,0.]]).expand_as(q),q,policy_dt=.01,relative_limits=LIMITS)
    before=geometry(q,torch.zeros_like(q));after=geometry(held,torch.zeros_like(q))
    torch.testing.assert_close(after['knee_deg'],before['knee_deg'],rtol=0,atol=2e-4)
    torch.testing.assert_close(after['leg_length_m'],before['leg_length_m'],rtol=0,atol=2e-6)
    assert (after['wheel_body_m']-before['wheel_body_m']).abs().max() > .001


def environment():
    settings = dict(paired_feedback_winding=True, cyclic_leg_targets=True,
        pd_feedback_period_ms=[1,3],pd_feedback_delay_ms=[0,5],pd_encoder_bias_rad=.02)
    env = SimpleNamespace(tick=0,num_envs=3,device='cpu',policy_dt=.01,decimation=10,
        getup_cfg=settings,domain=SimpleNamespace(enabled=torch.tensor([False,True,True]),
            rng=torch.Generator().manual_seed(11)),control_audit=SimpleNamespace(control_stride=5),
        startup_report={},actions=torch.ones(3,6))
    env.quiet_standing=SimpleNamespace(update=lambda active,xy,target,velocity: target)
    env.getup_feedback=GetupFeedback(3,'cpu',env.domain.enabled,settings,env.domain.rng,dt=.001)
    env.v5=SimpleNamespace(LEGS=(0,1,3,4),WHEELS=(2,5),action_bounds=torch.tensor([16.,16.,9.,16.,16.,9.]),
        settings=dict(wheel_velocity_scale=10.),gas_compensation=SimpleNamespace(cfg=dict(relative_soft_limits=LIMITS_CFG)))
    return env


def test_controller_has_no_target_accumulator_and_uses_existing_sample_without_advancing_it():
    env=environment(); adapter=EncoderIncrementActions(env,contract())
    ids=torch.arange(3); q=torch.tensor([[0.,.4,0.,0.,-.4,0.]]).expand(3,-1).clone()
    env.getup_feedback.reset(ids,q,torch.zeros_like(q))
    before={k:v.clone() for k,v in vars(env.getup_feedback).items() if torch.is_tensor(v)}
    a=torch.tensor([[.5,.2,.3,-.4,-.2,-.8]]).expand(3,-1).clone()
    first=env.v5.decode(a,torch.full_like(q,100.))
    second=env.v5.decode(a,torch.full_like(q,-100.))
    for left,right in zip(first,second): assert torch.equal(left,right)
    assert torch.equal(first[1],a[:,[2,5]]*10.)
    physical=adapter.physical_targets()
    assert torch.equal(physical[:,[0,1,3,4]],first[0]) and torch.equal(physical[:,[2,5]],first[1])
    assert torch.equal(env.quiet_standing.update(None,None,torch.zeros_like(physical),None),physical)
    assert env.getup_feedback.tick == env.getup_feedback.cursor == 0
    assert all(torch.equal(getattr(env.getup_feedback,k),v) for k,v in before.items())
    env.getup_feedback.q[:,0] += .1
    third=env.v5.decode(a,q)[0]
    assert not torch.equal(third,first[0])  # Measurement, never previous target, is the reference.
    env._measured_action_history(torch.tensor([0,2]))
    assert not env.actions[[0,2]].any() and (env.actions[1] == 1.).all()
    assert env.getup_feedback.dt == .005 and env.getup_cfg['cyclic_leg_targets'] is False


def test_feedback_draws_are_quantized_to_actual_PD_clock_and_reset_primes_birth_pose():
    env=environment(); EncoderIncrementActions(env,contract())
    q=torch.tensor([[.3,.6,1.,-.4,-.7,2.]]).expand(3,-1).clone(); dq=torch.zeros_like(q)
    env.getup_feedback.reset(torch.arange(3),q,dq)
    assert torch.equal(env.getup_feedback.q,q) and torch.equal(env.getup_feedback.q_history,q[None].expand_as(env.getup_feedback.q_history))
    assert (env.getup_feedback.period == 1).all()  # 1..3ms at5ms physical PD cadence.
    assert ((env.getup_feedback.delay+env.getup_feedback.delay_fraction)*5 <= 5).all()
    before=env.getup_feedback.tick
    env.getup_feedback.observe(q,dq); env.getup_feedback.observe(q,dq)
    assert env.getup_feedback.tick-before == 2


def test_cannot_install_into_a_controller_with_existing_history():
    env=environment();env.getup_feedback.tick=1
    with pytest.raises(ValueError,match='before any physics'):EncoderIncrementActions(env,contract())


def test_policy_previous_action_slots_keep_new_raw_coordinates_separate_from_encoders():
    from wheeled_tasks.chassis.scut_observation import build_scut35, CONTROL_FROM_POLICY
    q=torch.tensor([[.2,.6,0.,-.2,-.6,0.]])
    raw=torch.tensor([[.1,.2,.3,.4,.5,.6]])
    frame=build_scut35(torch.zeros(1,3),torch.tensor([[0.,0.,-1.]]),torch.tensor([[0.,0.,.305]]),
        q,torch.zeros(1,6),raw[:,CONTROL_FROM_POLICY],torch.zeros(6),torch.zeros(1,dtype=torch.bool),
        torch.zeros(1),torch.zeros(1))
    assert torch.equal(frame[:,22:28],raw)
    torch.testing.assert_close(frame[:,10:14],q[:,[0,1,3,4]],rtol=0,atol=1e-7)


def test_versioned_action_initialization_and_exploration_cannot_be_mixed():
    old=json.loads((ROOT/'contracts/wheel_leg_self_righting_near_start_v1.json').read_text())
    assert validate_plan(old) == old and validate_plan(PLAN) == PLAN
    assert set(PLAN)-set(old) == {'encoder_increment_actions'}
    for key in ('multi_pose_recovery','recovery_impact','global_curriculum','timing','pose_bank','terminal_transition'):
        assert PLAN[key] == old[key]
    for key in ('encoder_increment_actions','learning_interface','exploration_distribution'):
        bad=deepcopy(PLAN);bad.pop(key)
        with pytest.raises(ValueError):validate_plan(bad)
    bad=contract();bad['privileged_input']=True
    with pytest.raises(ValueError):validate(bad)


def test_new_fresh_actor_has_no_absolute_pose_hold_and_all_six_outputs_train():
    from train_prone_motion import learner_config
    from rsl_rl.algorithms import PPO
    obs=TensorDict(dict(policy=torch.randn(32,35),critic=torch.randn(32,87)),batch_size=[32])
    cfg=learner_config(PLAN,'cpu');cfg['multi_gpu']=None
    torch.manual_seed(107)
    alg=PPO.construct_algorithm(obs,SimpleNamespace(num_envs=32,num_actions=6),cfg,'cpu')
    receipt=initialize_prone_fresh(alg,torch.ones(6)*12.,PLAN)
    assert receipt['output_bias_policy_order'] == [0.]*6 and not alg.optimizer.state
    assert not receipt['checkpoint_loaded'] and not receipt['teacher_loaded']
    assert receipt['runtime_action_decoder_changed'] and receipt['optimizer_states'] == 0
    before=alg.actor(obs);assert before.abs().max() < .1
    alg.actor.distribution.update(before)
    torch.testing.assert_close(alg.actor.distribution.std[0],torch.tensor(PLAN['initial_std']))
    loss=-(alg.actor.distribution.log_prob(before.detach()+torch.tensor([.2,-.1,.1,-.2,.3,-.3]))
        *torch.linspace(-1.,1.,32)).mean()+alg.critic(obs).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in alg.actor.parameters())
    alg.optimizer.step()
    assert ((alg.actor(obs)-before).abs().sum(0) > 0).all() and alg.optimizer.state


def test_native_contract_and_frozen_sources_include_decoder_and_initialization(tmp_path):
    from train_prone_motion import configuration, source_hashes
    configuration(tmp_path,PLAN)
    saved=json.loads((tmp_path/'task_contract.json').read_text())
    assert saved['encoder_increment_actions'] == contract()
    assert saved['action_order'] == contract()['policy_order']
    assert saved['leg_position_scale_rad'] is None and saved['leg_action_clip'] == 1.
    assert saved['settings']['cyclic_leg_targets'] is False
    hashes=source_hashes(PATH,PLAN)
    assert all(n in hashes for n in ('src/wheeled_tasks/chassis/recovery_actions.py',
        'src/wheeled_algo/prone_initialization.py','tests/test_recovery_actions.py','docs/RECOVERY_ACTIONS.md'))

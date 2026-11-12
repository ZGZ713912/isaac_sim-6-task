"""Regressions for measured v3 timing faults and selective actuator assistance."""
import json
import pytest
import torch

from wheeled_tasks.wheelbipe_port.control import tuck_spring_effort, validate_tuck_compensation
from wheeled_tasks.wheelbipe_port.model import BUNDLE, PairedTargets
from wheeled_tasks.wheelbipe_port.mixed_curriculum import JumpObserver, TaskSchedule, quota_counts
from wheeled_tasks.wheelbipe_port.training_profile import (
    apply_control_overrides, apply_phase, configure_profile, phase_for_update,
    profile_sha256, specification,
)
from test_wheelbipe_foundation_profile import fake_cfg


def setup():
    spec = specification("foundation-jump-v4")
    schedule = TaskSchedule(100,"cpu",119,spec)
    observer = JumpObserver(schedule,.005,torch.full((100,),20.))
    return schedule,observer


def observe(s,o,length=.38,vz=.6,clear=.03,force=0.):
    velocity=torch.zeros(100,3);velocity[:,2]=vz
    gravity=torch.zeros_like(velocity);gravity[:,2]=-1.
    o.observe(s.delay+1.,torch.full((100,),.5),velocity,gravity,
        torch.full((100,2),clear),torch.full((100,2),force),torch.zeros(100),
        leg_lengths=torch.full((100,2),length))


def test_first_air_baseline_preserves_retraction_during_confirmation():
    s,o=setup();jump=s.mask("jump")
    observe(s,o)
    assert not o.flown[jump].any()
    for _ in range(7):observe(s,o,length=.32)
    assert o.flown[jump].all() and o.tuck_seen[jump].all()
    torch.testing.assert_close(o.release_leg_lengths[jump],torch.full((int(jump.sum()),2),.38))
    torch.testing.assert_close(o.max_bilateral_tuck_delta[jump],torch.full((int(jump.sum()),),.06))


def test_failed_provisional_flight_cannot_donate_baseline_or_tuck():
    s,o=setup();jump=s.mask("jump")
    observe(s,o);observe(s,o,length=.30)
    observe(s,o,length=.30,clear=0.,force=100.)
    assert not o.flown[jump].any() and not o.tuck_seen[jump].any()
    assert (o.release_leg_lengths[jump]==0.).all()
    assert (o.max_bilateral_tuck_delta[jump]==0.).all()
    for _ in range(8):observe(s,o,length=.30)
    assert o.flown[jump].all() and not o.tuck_seen[jump].any()
    torch.testing.assert_close(o.release_leg_lengths[jump],torch.full((int(jump.sum()),2),.30))


def test_apex_hold_is_independent_of_air_counter_and_cannot_regress():
    s,o=setup();jump=s.mask("jump")
    for _ in range(8):observe(s,o)
    observe(s,o,vz=-.2)
    assert o.apex_seen[jump].all() and not o.extension_started[jump].any()
    for _ in range(7):observe(s,o,vz=-.2)
    assert not o.extension_started[jump].any()
    observe(s,o,vz=-.2)
    assert o.extension_started[jump].all()
    # The exact observed v3 fault: below 15mm but above landing clearance,
    # still descending, air_time resets yet physical landing has not occurred.
    observe(s,o,vz=-.5,clear=.01)
    assert (o.air_time[jump]==0.).all() and not o.landed[jump].any()
    wheel=torch.zeros(100,2,3);wheel[:,:,2]=.35
    o.reward_terms(s.delay+1.,torch.full((100,),.5),torch.full((100,),-.5),wheel,.02)
    assert (o.last_air_tuck[jump]==0.).all() and (o.last_air_extension[jump]>0.).all()
    observe(s,o,vz=.2)
    o.reward_terms(s.delay+1.,torch.full((100,),.5),torch.full((100,),.2),wheel,.02)
    assert (o.last_air_tuck[jump]==0.).all()
    o.reset(torch.arange(100))
    assert not o.apex_seen.any() and not o.extension_started.any()


def test_one_leg_progress_earns_dense_credit_but_not_bilateral_success():
    s,o=setup();jump=s.mask("jump")
    for _ in range(8):observe(s,o)
    wheel=torch.zeros(100,2,3);wheel[:,:,2]=.38
    args=(s.delay+1.,torch.full((100,),.5),torch.full((100,),.5))
    o.reward_terms(*args,wheel,.02);before=o.last_air_tuck.clone()
    wheel[:,0,2]=.30
    o.reward_terms(*args,wheel,.02)
    assert (o.last_air_tuck[jump]>before[jump]).all()
    assert not o.tuck_seen[jump].any()


def test_virtual_range_changes_targets_and_cost_but_preserves_mechanical_bounds():
    model=json.loads((BUNDLE/'gas_spring_model.json').read_text())
    old=PairedTargets(model,'cpu');new=PairedTargets(model,'cpu',.2)
    torch.testing.assert_close(new.mechanical_bounds,old.bounds)
    desired=torch.tensor([[-.75,.75,.75,-.75]],dtype=torch.float64)
    torch.testing.assert_close(new(desired,desired),desired)
    assert new.projection_cost(desired)==0. and old.projection_cost(desired)>0.
    shifted=desired+torch.tensor([[4.,4.,-6.,-6.]])*torch.pi
    torch.testing.assert_close(new(shifted,desired),desired,atol=1.e-6,rtol=0.)
    far=desired*3
    assert new.projection_cost(far)>0.
    # A mixed batch must retain the historical target and projection cost on
    # ordinary lanes while exposing the extra actuator authority to jumps.
    batch=desired.repeat(2,1)
    eligible=torch.tensor([False,True])
    targets=new(batch,batch,extended_mask=eligible)
    torch.testing.assert_close(targets[0:1],old(desired,desired))
    torch.testing.assert_close(targets[1:2],desired)
    costs=new.projection_cost(batch,extended_mask=eligible)
    torch.testing.assert_close(costs[:1],old.projection_cost(desired))
    assert costs[1]==0.
    for value in (-.1,float('nan'),.36):
        with pytest.raises(ValueError):PairedTargets(model,'cpu',value)


def test_model_compensation_only_cancels_inward_intent_per_leg_and_is_capped():
    settings=specification('foundation-jump-v4')['control']['tuck_spring_compensation']
    q=torch.tensor([[0.,1.,1.,0.]]).repeat(5,1)
    target=torch.tensor([[.1,.9,.9,.1]]).repeat(5,1)
    inward=torch.tensor([[10.,-10.,-10.,10.]]).repeat(5,1)
    holding=inward*4.
    eligible=torch.tensor([True,False,True,True,True])
    target[2]=q[2]  # No leg-length change; common rotation must also be zero.
    target[2]+=.4
    target[3]=q[3]*2  # Extension request, retain gas assistance for pushing.
    inward[4]*=-1.  # PD wants braking, not additional inward torque.
    result,gate=tuck_spring_effort(q,target,inward,eligible,holding,settings)
    torch.testing.assert_close(result[0],torch.tensor([16.,-16.,-16.,16.]))
    assert (result[1:]==0.).all() and (gate[1:]==0.).all()
    # Left/right symmetry, and no common-rotation effort is introduced.
    assert (result.reshape(-1,2,2).sum(-1)==0.).all()
    target[0,2:]=q[0,2:]
    result,_=tuck_spring_effort(q,target,inward,eligible,holding,settings)
    assert (result[0,2:]==0.).all() and (result[0,:2].abs()==16.).all()
    off={**settings,'mode':'off'}
    assert not tuck_spring_effort(q,target,inward,eligible,holding,off)[0].any()


def test_v4_defaults_are_versioned_without_changing_ordinary_rewards_or_quotas():
    old,new=fake_cfg(),fake_cfg()
    configure_profile(old,'foundation-jump-v3');configure_profile(new,'foundation-jump-v4')
    assert new.port_leg_kp==80. and new.port_leg_kd==1.5
    assert new.port_leg_target_overtravel_rad==.2 and not new.port_gas_feedforward
    assert new.port_leg_target_overtravel_scope=='jump-request'
    assert new.rewards==old.rewards
    assert quota_counts(2048,new.port_mixed_spec)==quota_counts(2048,old.port_mixed_spec)
    settings=new.port_tuck_spring_compensation
    for name,value in (('gain',1.1),('cap_nm',41.),('error_full_rad',.01)):
        with pytest.raises(ValueError):validate_tuck_compensation({**settings,name:value})


def test_cli_control_overrides_and_profile_switch_do_not_leak_selective_compensation():
    cfg=fake_cfg()
    configure_profile(cfg,'foundation-jump-v4')
    apply_control_overrides(cfg,kp=90.,kd=2.,overtravel=.1,tuck_mode='off',tuck_cap_nm=8.)
    assert (cfg.port_leg_kp,cfg.port_leg_kd)==(90.,2.)
    assert cfg.port_leg_target_overtravel_rad==.1
    assert cfg.port_tuck_spring_compensation['mode']=='off'
    configure_profile(cfg,'foundation-jump-v4')
    assert cfg.port_tuck_spring_compensation['mode']=='model'
    configure_profile(cfg,'foundation-jump-v3')
    assert cfg.port_leg_target_overtravel_rad==0.
    assert cfg.port_tuck_spring_compensation['mode']=='off'
    for kwargs in ({'overtravel':float('nan')},{'kp':0.},{'tuck_gain':float('inf')}):
        with pytest.raises(ValueError):apply_control_overrides(cfg,**kwargs)


def test_10000_update_course_reaches_all_skills_without_changing_episode_or_reward_scales():
    original=specification('foundation-jump-v4')
    spec=specification('foundation-jump-v4',10000)
    assert [p['start_update'] for p in spec['phases']]==[0,1000,3000,6000]
    assert spec['special_mode_starts']==dict(dash=6000,spin_low=6000,spin_mid=7000)
    assert spec['jump']['difficulty_full_update']==5000
    assert spec['jump']['moving_jump']['start_update']==3000
    assert spec['jump']['moving_jump']['full_update']==6000
    assert spec['jump']['crouch_seconds']==original['jump']['crouch_seconds']
    assert spec['jump']['landing_buffer_duration_s']==.3
    assert spec['reward_policy']==original['reward_policy']
    assert spec['task_fractions']==original['task_fractions']
    assert original['total_updates']==20000 and spec['total_updates']==10000
    assert profile_sha256('foundation-jump-v4',10000)!=profile_sha256('foundation-jump-v4')
    cfg=fake_cfg();configure_profile(cfg,'foundation-jump-v4',course_updates=10000)
    assert apply_phase(cfg,1000)['start_update']==1000
    assert apply_phase(cfg,6000)['vx_max_m_s']==2.7
    assert cfg.port_mixed_spec==spec
    for budget in (-1,0,1,True):
        with pytest.raises(ValueError):specification('foundation-jump-v4',budget)


def test_compressed_course_unlocks_full_spin_and_moving_jumps_before_the_end():
    spec=specification('foundation-jump-v4',10000)
    s=TaskSchedule(2048,'cpu',119,spec)
    s.reset(torch.arange(2048),7000)
    phase=dict(phase_for_update(7000,spec),completed_update=7000)
    commands,_=s.update_commands(torch.zeros(2048),phase)
    spin=s.mask('spin');moving=s.mask('jump')&(commands[:,0].abs()>.01)
    assert commands[spin,2].abs().max()>100*2*torch.pi/60.
    assert (commands[spin,2]>0.).any() and (commands[spin,2]<0.).any()
    assert moving.sum()>=200
    torch.testing.assert_close(commands[moving,0].abs(),torch.full((int(moving.sum()),),.5))

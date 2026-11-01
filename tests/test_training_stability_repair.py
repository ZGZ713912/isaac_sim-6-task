"""Regression cases from frozen 4000/4500/4739 diagnostics."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from train_prone_motion import learner_config
from play_prone_motion import build_actor
from wheeled_algo.prone_observation import GatedObservationNormalization
from wheeled_tasks.chassis.standing_local_odometry import LocalStandingOdometry, observation_contract
from wheeled_tasks.chassis.global_curriculum import GlobalCurriculum, assessment_descriptors
from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery
from wheeled_tasks.chassis.multi_pose_recovery import validate_bank,validate_native_admission,recovery_terms

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4_multi_pose_repair_20261102.json').read_text())


def course():return GlobalCurriculum(PLAN,RecoveryMastery(PLAN['multi_pose_recovery']['mastery_gate']))


def rows(c,s,level=0,passed=True):
    return [dict(r,passed=passed) for r in assessment_descriptors(c.config,s,level)]


def test_rare_gated_channels_never_gain_empirical_scale_and_gradients_survive():
    normalizer=GatedObservationNormalization(35)
    x=torch.zeros(8192,35);x[-1,[14,15,31]]=torch.tensor([.05,-.05,1.])
    normalizer.update(x)
    normalized=normalizer(x)
    assert torch.equal(normalized[:,[14,15,31]],x[:,[14,15,31]])
    x=torch.ones(2,35,requires_grad=True)
    normalizer(x).sum().backward()
    assert torch.equal(x.grad[:,[14,15,31]],torch.ones(2,3))
    assert normalizer(torch.full((1,35),100.))[0,[14,15,31]].tolist()==[1.,1.,1.]
    restored=GatedObservationNormalization(35);restored.load_state_dict(normalizer.state_dict(),strict=True)
    torch.testing.assert_close(restored(x),normalizer(x),atol=0.,rtol=0.)


def test_new_reference_captures_during_balance_wheel_corrections_and_is_bounded():
    contract=observation_contract(2)
    odom=LocalStandingOdometry(2,'cpu',.01,contract)
    frame=torch.zeros(2,35);frame[:,9]=-1;frame[:,20:22]=.6 # 6 rad/s, blocked by legacy gate.
    wheel=torch.zeros(2,2,dtype=torch.float64)
    for tick in range(5):
        frame[:,14:16]=odom.local_wheel_angles(frame,wheel,torch.ones(2,dtype=torch.bool),tick)
        value=odom.update(frame,odom.legs.nominal,torch.ones(2,dtype=torch.bool),tick)
    assert value[:,31].eq(1).all()
    assert odom.displacement.eq(0).all()
    for tick in range(5,105):
        wheel += .06
        frame[:,14:16]=odom.local_wheel_angles(frame,wheel,torch.ones(2,dtype=torch.bool),tick)
        value=odom.update(frame,odom.legs.nominal,torch.ones(2,dtype=torch.bool),tick)
    assert value[:,14:16].abs().max()<=1.
    assert value[:,14:16].abs().max()>.5
    frame[:,0]=1.
    frame[:,14:16]=odom.local_wheel_angles(frame,wheel,torch.ones(2,dtype=torch.bool),105)
    value=odom.update(frame,odom.legs.nominal,torch.ones(2,dtype=torch.bool),105)
    assert not value[:,31].any() and not value[:,14:16].any()


def test_retention_failure_preserves_history_but_blocks_dependent_training_and_restores():
    c=course();c.review_skill('standing',0,rows(c,'standing'),250)
    assert c.levels['standing']==c.retained_levels['standing']==1
    allocation=c.training_descriptors(256,0)
    assert sum(r['curriculum_skill']=='recovery' for r in allocation)==77
    assert sum(r['curriculum_skill']=='static_height' for r in allocation)==8
    saved=deepcopy(c.state_dict())
    c.review_skill('standing',0,rows(c,'standing',passed=False),500)
    assert c.levels['standing']==1 and c.retained_levels['standing']==0
    assert c.learning_level('straight') is None
    assert {r['curriculum_skill'] for r in c.training_descriptors(256,1)}=={'standing'}
    # A previously queued dependent trial remains recordable after a parent fails.
    c.review_skill('straight',0,rows(c,'straight'),500)
    assert c.levels['straight']==0
    restored=course();restored.load_state_dict(c.state_dict())
    assert restored.retained_levels==c.retained_levels
    c.review_skill('standing',0,rows(c,'standing'),750)
    assert c.retained_levels['standing']==1 and c.learning_level('straight')==0
    restored.load_state_dict(saved);assert restored.retained_levels['standing']==1


def test_native_admission_is_complete_geometry_disjoint_and_cannot_change_clock():
    bank=json.loads((ROOT/PLAN['pose_bank']['file']).read_text())
    evidence=validate_native_admission(bank,ROOT,PLAN['timing'])
    assert len(evidence['cases'])==2*sum('recovery' in p for p in bank['poses'])
    signatures={split:set() for split in ['train','holdout']}
    groups={}
    for p in bank['poses']:
        if not p.get('recovery'):continue
        r=p['recovery'];groups.setdefault((r['family'],r['leg_class']),[]).append(r['split'])
        signatures[r['split']].add(json.dumps([p['root_position_m'],p['root_quaternion_xyzw'],p['full_joint_positions']],sort_keys=True))
    assert len(groups)==28 and all(v.count('holdout')==2 and 'train' in v for v in groups.values())
    assert not signatures['train'] & signatures['holdout']
    with pytest.raises(ValueError,match='physical contract'):
        validate_native_admission(bank,ROOT,{**PLAN['timing'],'physics_dt':.001})
    bad=deepcopy(bank);bad['multi_pose_recovery']['native_reset_admission']['sha256']='0'*64
    with pytest.raises(ValueError,match='changed'):validate_native_admission(bad,ROOT,PLAN['timing'])


def test_recovery_retention_relocks_after_any_class_fails():
    c=course().recovery
    values=[dict(family='belly_down',leg_class=cls,split='holdout',pose_index=4*i+j,
        heading_rad=0.,passed=True) for i,cls in enumerate(['front_front','rear_rear','front_rear','rear_front'])
        for j in range(4)]
    assert c.review(values,250)['complex_recovery_unlocked'] and c.unlocked
    values[-1]['passed']=False
    assert not c.review(values,500)['complex_recovery_unlocked'] and not c.unlocked
    values[-1]['passed']=True
    assert c.review(values,750)['complex_recovery_unlocked'] and c.unlocked


def test_queued_complex_review_is_recorded_but_cannot_restore_failed_parent():
    from wheeled_tasks.chassis.multi_pose_recovery import FAMILIES,LEG_CLASSES
    c=course();c.levels['standing']=c.retained_levels['standing']=1
    c.recovery.unlocked_at=250
    rows=[dict(family=f,leg_class=leg,split='holdout',pose_index=i*2+j,heading_rad=0.,passed=True)
        for i,(f,leg) in enumerate((f,l) for f in FAMILIES if f!='belly_down' for l in LEG_CLASSES) for j in range(2)]
    assert c.review_complex(rows,500)['passed']
    c.retained_levels['standing']=0
    assert not c.review_complex(rows,750)['passed'] and not c.complex_passed


def test_recovery_objective_excludes_the_old_large_tilt_standing_cost():
    c=PLAN['multi_pose_recovery']
    state=dict(height=torch.tensor([.288]),torque=torch.zeros(1,6),action_delta=torch.zeros(1))
    terms=recovery_terms(state,c,torch.zeros(1))
    assert set(terms)=={'rise_time','effort','action_rate','impact'}
    assert sum(terms.values()).item()==-1.
    assert c['failure_cost']==40. and c['failure_cost']>c['time_cost_per_second']*2


def test_actor_and_critic_use_identical_gated_normalizer_and_legacy_remains_compatible():
    from rsl_rl.utils import resolve_class
    obs=TensorDict(dict(policy=torch.zeros(2,35),critic=torch.zeros(2,87)),batch_size=[2])
    cfg=learner_config(PLAN)
    for key in ['actor','critic']:
        cls,settings=resolve_class(deepcopy(cfg[key]))
        model=cls(obs,cfg['obs_groups'],key,6 if key=='actor' else 1,**settings)
        assert isinstance(model.obs_normalizer,GatedObservationNormalization)
        assert model(obs).shape==(2,6 if key=='actor' else 1)
    old=json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4_before_repair_20261102.json').read_text())
    assert learner_config(old)['actor']['class_name']=='MLPModel'

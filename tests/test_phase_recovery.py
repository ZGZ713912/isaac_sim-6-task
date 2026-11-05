"""Recovery practice and contact-aware guidance cannot grant skill admission."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis import recovery_reward
from wheeled_tasks.chassis.global_curriculum import GlobalCurriculum
from wheeled_tasks.chassis.multi_pose_recovery import ResetSampler, FAMILIES
from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4_multi_pose_repair_20261102.json').read_text())


def test_multi_pose_practice_exists_before_standing_but_does_not_unlock_skills():
    plan=deepcopy(PLAN)
    plan['global_curriculum']['recovery_training']='persistent_multi_pose'
    mastery=RecoveryMastery(plan['multi_pose_recovery']['mastery_gate'])
    course=GlobalCurriculum(plan,mastery)
    for update in (0,1000,10000):
        rows=course.training_descriptors(100,update)
        assert sum(r['recovery']for r in rows)==30
        assert {r['role']for r in rows}=={'stand','prone'}
        assert not mastery.unlocked
        assert course.domain_context()==('nominal',0.)
        assert course.learning_level('straight')is None


def test_all_family_practice_reaches_every_family_without_qualifying_mastery():
    config=deepcopy(PLAN['multi_pose_recovery'])
    config['practice_distribution']='all_families_from_start'
    bank=json.loads((ROOT/PLAN['pose_bank']['file']).read_text())
    sampler=ResetSampler(bank,config,261105)
    selected=[bank['poses'][sampler.sample(0)]['recovery']for _ in range(1000)]
    assert {r['family']for r in selected}==set(FAMILIES)
    assert {r['split']for r in selected}=={'train'}
    assert not sampler.mastery.unlocked


def states():
    return dict(gravity=torch.tensor([[0.,0.,-1.]]*3),height=torch.full((3,),.30),
        omega=torch.tensor([[1.,0.,0.]]*3),velocity=torch.zeros(3,3),
        recovery_support_error_m=torch.zeros(3),
        recovery_wheel_load_fraction=torch.tensor([[.5,.5],[.5,0.],[0.,0.]]),
        recovery_chassis_load_fraction=torch.zeros(3))


def test_lift_needs_both_wheels_and_rate_cost_increases_on_support_capture():
    values=recovery_reward.dense_terms(states(),recovery_reward.contract())
    assert values['recovery_lift'][0]>0
    assert values['recovery_lift'][1:].count_nonzero()==0
    assert values['recovery_phase_body_rate'][0]<values['recovery_phase_body_rate'][2]<0


def test_inverted_body_has_orientation_gradient_but_no_lift_credit():
    state=states();state['gravity'][:,2]=1.
    values=recovery_reward.dense_terms(state,recovery_reward.contract())
    assert (values['recovery_orientation']<0).all()
    assert values['recovery_lift'].count_nonzero()==0
    assert all(torch.isfinite(v).all()for v in values.values())


def test_partial_reward_contract_is_rejected():
    config=recovery_reward.contract();config.pop('motor_targets')
    with pytest.raises(ValueError):recovery_reward.validate(config)

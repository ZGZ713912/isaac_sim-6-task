"""Recovery discovery may change training exposure, never independent admission."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.global_curriculum import GlobalCurriculum, SKILLS
from wheeled_tasks.chassis.prone_motion import validate as validate_plan
from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery
from wheeled_tasks.chassis.recovery_discovery import contract, completed_timeout, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT/'contracts/wheel_leg_self_righting_phase_exploration_v2.json').read_text())


def course(plan):
    return GlobalCurriculum(plan, RecoveryMastery(plan['multi_pose_recovery']['mastery_gate']))


def focused_plan():
    value = deepcopy(BASE)
    value['global_curriculum']['recovery_focus'] = contract()
    return value


def test_recovery_focus_allocates_61_of_64_lanes_without_unlocking_motion_or_domains():
    c = course(focused_plan())
    for count in (20, 32, 64, 256):
        for generation in (0, 1, 250, 10000):
            rows = c.training_descriptors(count, generation)
            n = sum(r['recovery'] for r in rows)
            assert len(rows) == count and abs(n/count-.95) <= 1/count
            assert {r['role'] for r in rows} == {'stand','prone'}
            assert all(r['target'] == [0.,0.,.305] for r in rows if r['recovery'])
            assert all(r['nominal_birth'] for r in rows if not r['recovery'])
            assert c.levels == dict.fromkeys(SKILLS,0)
            assert not c.domains_admitted and c.learning_level('straight') is None
    assert sum(r['recovery'] for r in c.training_descriptors(64,0)) == 61
    assert not c.recovery.unlocked


def test_legacy_plan_preserves_19_of_64_recovery_lanes_and_full_cold_jobs():
    old, focused = course(BASE), course(focused_plan())
    assert sum(r['recovery'] for r in old.training_descriptors(64,0)) == 19
    assert focused.jobs(250) == old.jobs(250)
    assert focused.config['training_weights'] == old.config['training_weights']
    assert focused.config['skills'] == old.config['skills']


def test_focus_does_not_replace_independent_recovery_thresholds_or_physical_contract():
    plan = focused_plan()
    validate_plan(plan)
    for key in ('timing','wheel_model','gas_spring_model','recovery_impact','multi_pose_recovery',
                'learning_interface','completion_hold_seconds','evaluation'):
        assert plan[key] == BASE[key]
    assert plan['global_curriculum']['recovery_focus']['evaluation_early_end'] is False
    standard = deepcopy(plan)
    standard['global_curriculum']['recovery_training'] = 'persistent_standard_prone'
    standard['global_curriculum']['recovery_scope'] = 'standard_prone_only'
    with pytest.raises(ValueError):
        course(standard)


@pytest.mark.parametrize('key,value', [('recovery_fraction',1.), ('standing_fraction',0.),
    ('completion_stable_seconds',.3), ('evaluation_early_end',True), ('teacher',True),
    ('actor_observation_additions',True), ('admission_gates','auto_pass')])
def test_incomplete_or_relaxed_focus_contract_is_rejected(key,value):
    value_dict = contract()
    value_dict[key] = value
    with pytest.raises(ValueError):
        validate(value_dict)


def test_success_timeout_is_training_only_and_never_masks_a_failure_or_standing_birth():
    hold = torch.tensor([2.99,3.,4.,4.,4.])
    recovery = torch.tensor([True,True,True,False,True])
    terminated = torch.tensor([False,False,True,False,False])
    mask = completed_timeout(contract(),hold,recovery,terminated,training=True,evaluating=False)
    assert mask.tolist() == [False,True,False,False,True]
    for training,evaluating in ((False,False),(True,True),(False,True)):
        assert not completed_timeout(contract(),hold,recovery,terminated,training=training,evaluating=evaluating).any()
    assert not completed_timeout(None,hold,recovery,terminated,training=True,evaluating=False).any()

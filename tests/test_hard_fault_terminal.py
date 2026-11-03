"""Protect PPO terminal targets without concealing faults or clipping normal reward."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from train_prone_motion import configuration,source_hashes
from wheeled_tasks.chassis.prone_motion import validate as validate_plan
from wheeled_tasks.chassis.terminal_transition import CONTRACT,settle_reward,validate

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'contracts/wheel_leg_prone_motion_v4_terminal_settlement_v1.json'
PLAN=json.loads(PATH.read_text())
OLD=json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4_physical_interface_v1.json').read_text())


def test_new_contract_changes_only_hard_terminal_settlement():
    same=deepcopy(PLAN)
    assert same.pop('terminal_transition')==CONTRACT and same==OLD
    assert validate_plan(PLAN)==PLAN and validate_plan(OLD)==OLD
    assert PLAN['closure_guard']['mode']=='record_only'
    assert PLAN['multi_pose_recovery']['dense_reward']==OLD['multi_pose_recovery']['dense_reward']


@pytest.mark.parametrize('fault',['mechanical','impact','part'])
@pytest.mark.parametrize('cost',[3.,40.])
def test_extreme_postfailure_cost_keeps_preaction_learning_failure_signal(fault,cost):
    reward=torch.tensor([-1e15,-7.5,2.])
    flags={k:torch.zeros(3,dtype=torch.bool) for k in ['mechanical','impact','part']}
    flags[fault][0]=True
    new,hard=settle_reward(reward,cost,flags['mechanical'],flags['impact'],flags['part'])
    assert new.tolist()==[-cost,-7.5,2.] and hard.tolist()==[True,False,False]
    # No change to the caller's raw diagnostic; a terminal target is bounded
    # by the unchanged failure penalty rather than the failed physical state.
    assert reward[0]<-1e14 and new.square().max()<1e5


def test_ordinary_refall_timeout_excursion_and_closure_only_are_not_clipped():
    raw=torch.tensor([-1e15,-3.75,-40.5,6.25])
    none=torch.zeros(4,dtype=torch.bool)
    new,hard=settle_reward(raw,40.,none,none)
    assert torch.equal(raw,new) and not hard.any()


def test_per_role_penalty_and_nonfinite_failed_poststate():
    raw=torch.tensor([float('-inf'),-9.,float('nan'),.5])
    mechanical=torch.tensor([True,False,False,False]);impact=torch.tensor([False,False,True,False])
    new,hard=settle_reward(raw,torch.tensor([40.,40.,3.,3.]),mechanical,impact)
    assert new.tolist()==[-40.,-9.,-3.,.5] and torch.isfinite(new).all()
    assert hard.tolist()==[True,False,True,False]


@pytest.mark.parametrize('field,value',[('mode','clip_all'),('existing_faults',['closure']),
    ('closure_error_is_not_a_fault',False),('failure_cost',1.),('extra',True)])
def test_incomplete_or_different_contract_is_rejected(field,value):
    bad=deepcopy(CONTRACT);bad[field]=value
    with pytest.raises(ValueError,match='complete versioned'):validate(bad)


@pytest.mark.parametrize('mask',[torch.zeros(2,dtype=torch.bool),torch.zeros(3)])
def test_wrong_lane_masks_are_rejected(mask):
    with pytest.raises(ValueError,match='per-lane reward'):
        settle_reward(torch.ones(3),40.,mask,torch.zeros(3,dtype=torch.bool))


def test_native_contract_and_sources_preserve_explicit_settlement(tmp_path):
    configuration(tmp_path,PLAN)
    task=json.loads((tmp_path/'task_contract.json').read_text())
    assert task['terminal_transition']==CONTRACT
    hashes=source_hashes(PATH,PLAN)
    for p in ['src/wheeled_tasks/chassis/terminal_transition.py','tests/test_hard_fault_terminal.py',
              'docs/HARD_FAULT_TERMINAL_20261103.md',str(PATH.relative_to(ROOT))]:assert p in hashes

"""Behavioral regressions observed in the first37-case tracking review."""
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.standing009_curriculum import validate, tracking_density
from wheeled_tasks.chassis.standing009_retention import costs, height_failure, critic_anchor

ROOT = Path(__file__).resolve().parents[1]
OLD = json.loads((ROOT/"contracts/wheel_leg_009_reproduction_v1.json").read_text())
PLAN = json.loads((ROOT/"contracts/wheel_leg_009_reproduction_v2.json").read_text())


def test_repair_keeps_009_foundation_and_all_declared_behavior_limits():
    validate(PLAN)
    assert PLAN["standing_recipe"] == OLD["standing_recipe"]
    assert PLAN["evaluation"] == OLD["evaluation"]
    assert PLAN["phases"] == OLD["phases"]
    assert PLAN["repair"]["standing_episode_seconds"] == 30.
    assert PLAN["repair"]["motion_episode_seconds"] == 12.
    assert PLAN["repair"]["training_birth_height_m"] == .305


def test_old_reward_cannot_distinguish_position_but_repair_can():
    commands = torch.tensor([[0.,0.,.305]]*3)
    height = commands[:,2].clone()
    gravity = torch.tensor([[0.,0.,-1.]]*3)
    v = torch.tensor([[.01,0.,0.]]*3)
    omega = torch.zeros_like(v)
    old = tracking_density(height,gravity,v,omega,commands,OLD)
    assert torch.equal(old,old[0].expand(3))
    paid = costs(torch.tensor([[0.,0.],[.05,0.],[.38,0.]]),height,commands,PLAN)
    assert paid["position"].tolist() == pytest.approx([0.,8.,32.])
    assert old[0]-paid["position"][0] > old[1]-paid["position"][1] > old[2]-paid["position"][2]


def test_one_centimeter_is_a_good_region_and_movement_does_not_penalize_travel():
    commands = torch.tensor([[0.,0.,.305],[0.,0.,.305],[.3,0.,.305],[0.,.6,.305]])
    result = costs(torch.tensor([[0.,0.],[.006,.008],[1.,0.],[1.,0.]]),commands[:,2],commands,PLAN)
    assert result["position"].abs().max() < 1e-10
    assert result["height"].eq(0).all()


def test_height_signal_remains_dense_far_from_target_and_prefers_closer_states():
    cmd = torch.tensor([[0.,0.,.42]]*3)
    result = costs(torch.zeros(3,2),torch.tensor([.305,.35,.42]),cmd,PLAN)
    assert result["height"][0] > result["height"][1] > result["height"][2] == 0.


@pytest.mark.parametrize("command", [.23,.42])
def test_height_learning_is_not_killed_at_25mm_but_static_mismatch_still_is(command):
    height = torch.tensor([.305,.305])
    target = torch.full((2,),command)
    learning = torch.tensor([True,False])
    assert height_failure(height,target,learning,PLAN).tolist() == [False,True]
    assert height_failure(height,target,learning,OLD).all()


@pytest.mark.parametrize("height", [.1,.5])
def test_physical_height_safety_remains_for_every_role(height):
    assert height_failure(torch.tensor([height,height]),torch.tensor([.305,.305]),
                          torch.tensor([True,False]),PLAN).all()


def test_anchor_is_privileged_and_changes_only_two_previously_unused_entries():
    original = torch.randn(4,81)
    original[:,[43,46]] = 0.
    error = torch.tensor([[.01,.02],[1.,-1.],[3.,-3.],[0.,0.]])
    changed = critic_anchor(original,error,(43,46))
    keep = [i for i in range(81) if i not in [43,46]]
    assert torch.equal(changed[:,keep],original[:,keep])
    assert torch.equal(original[:,[43,46]],torch.zeros(4,2))
    assert torch.equal(changed[:,[43,46]],(error*10).clamp(-10,10))


def test_v1_keeps_original_paid_costs_zero():
    result = costs(torch.ones(2,2),torch.zeros(2),torch.ones(2,3),OLD)
    assert all(x.eq(0).all() for x in result.values())

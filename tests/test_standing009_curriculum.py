"""Historical equivalence and the behavior failures a curriculum must expose."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.standing009_curriculum import (
    validate, case, commands_at, domain_stage, evaluation_cases, tracking_density)
from wheeled_algo.scratch_standing import agent_config

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/"contracts/wheel_leg_009_reproduction_v1.json").read_text())
sys.path.insert(0, str(ROOT/"scripts"))
from evaluate_009_curriculum import review_arrays
from train_009_curriculum import configuration


def old_source(name):
    return subprocess.check_output(["git", "show", f"643a0de0:{name}"], cwd=ROOT, text=True)


def test_009_actor_and_zero_request_reward_sources_remain_exact():
    for name in ("src/wheeled_algo/leg_velocity_separated.py", "src/wheeled_algo/wheel_velocity_increment.py",
                 "src/wheeled_tasks/chassis/precise_standing_reward.py"):
        assert (ROOT/name).read_text() == old_source(name)


def test_agent_parameters_equal_historical009():
    namespace = {}
    exec(compile(old_source("src/wheeled_algo/scratch_standing.py"), "historical009", "exec"), namespace)
    actual = agent_config(261001, "cuda:0", 128, .0003, normalize_observations=True,
                          wheel_velocity_increment=True, separate_leg_velocity_feedback=True)
    expected = namespace["agent_config"](261001, "cuda:0", 128, .0003, normalize_observations=True,
                                       wheel_velocity_increment=True, separate_leg_velocity_feedback=True)
    assert actual == expected


def test_zero_motion_density_is_bit_exact_historical_reward_even_when_fallen():
    namespace = {}
    exec(compile(old_source("src/wheeled_tasks/chassis/precise_standing_reward.py"), "historical009", "exec"), namespace)
    rng = torch.Generator().manual_seed(271028)
    gravity = torch.randn(128, 3, generator=rng)
    gravity /= gravity.norm(dim=-1, keepdim=True)
    height = .1+.4*torch.rand(128, generator=rng)
    velocity = torch.randn(128, 3, generator=rng)*.1
    omega = torch.randn(128, 3, generator=rng)*.3
    command = torch.zeros(128, 3)
    command[:, 2] = .305
    expected = namespace["scores"](height, gravity, velocity, omega, command[:, 2])
    actual = tracking_density(height, gravity, velocity, omega, command, PLAN)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("axis,target", [(0,.3),(0,-.3),(1,.6),(1,-.6)])
def test_motion_reward_prefers_measured_requested_direction_over_quiet_or_opposite(axis, target):
    g = torch.tensor([[0.,0.,-1.]]*3)
    velocity, omega, command = torch.zeros(3,3), torch.zeros(3,3), torch.tensor([[0.,0.,.305]]*3)
    command[:, axis] = target
    (velocity if axis == 0 else omega)[:, 0 if axis == 0 else 2] = torch.tensor([target,0.,-target])
    density = tracking_density(torch.full((3,),.305), g, velocity, omega, command, PLAN)
    assert density[0] > density[1] > density[2]
    assert density[0].item() == pytest.approx(12.)


def test_command_ramps_reach_all_targets_and_return_to_standing_without_motor_commands():
    rows = [case(PLAN,r) for r in PLAN["tracking"]["roles_per20"]]
    for t in (0., 1., 12., 30.):
        c = commands_at(rows, torch.full((20,),t), PLAN)
        assert torch.equal(c, torch.tensor([[0.,0.,.305]]*20))
    peak = commands_at(rows, torch.full((20,),6.), PLAN)
    assert peak[:,0].min() == pytest.approx(-.3) and peak[:,0].max() == pytest.approx(.3)
    assert peak[:,1].min() == pytest.approx(-.6) and peak[:,1].max() == pytest.approx(.6)
    assert peak[:,2].min() == pytest.approx(.23) and peak[:,2].max() == pytest.approx(.42)
    assert torch.equal(peak[:8], torch.tensor([[0.,0.,.305]]*8))
    assert all(not r["recovery"] and "actions" not in r for r in rows)


def test_only_later_curriculum_adds_domains_and_final_ranges_are_not_dropped():
    validate(PLAN)
    assert PLAN["phases"][0]["min_updates"] == 512
    assert PLAN["phases"][0]["profile"] == PLAN["phases"][1]["profile"] == "nominal"
    assert domain_stage(PLAN,0) == ("sensors_only",.25)
    assert domain_stage(PLAN,128) == ("motor_only",.25)
    assert domain_stage(PLAN,896) == ("requested",1.)
    assert len(evaluation_cases(PLAN,"standing")) == 7
    assert len(evaluation_cases(PLAN,"tracking")) == 37


def arrays_for(rows):
    n, count = 6000, len(rows)
    return dict(valid=np.ones((n,count),dtype=bool), terminated=np.zeros((n,count),dtype=bool),
        velocity=np.zeros((n,count,3)), omega=np.zeros((n,count,3)),
        commands=np.tile(np.array([r["command"] for r in rows])[None],(n,1,1)),
        height=np.tile(np.array([r["birth_height_m"] for r in rows])[None],(n,1)),
        gravity=np.tile([0.,0.,-1.],(n,count,1)), force=np.full((n,5,count,2,3),10.),
        position=np.zeros((n,count,3)), dq=np.zeros((n,5,count,4)), gap=np.zeros((n,count)),
        nonwheel_force=np.zeros((n,count)), clearance=np.zeros((n,5,count,2)))


def test_contact_tolerance_does_not_hide_later_fall_or_choose_a_better_suffix():
    rows = [case(PLAN,"stand")]
    a = arrays_for(rows)
    a["force"][0,:2,0] = 0.
    good = review_arrays(a,rows,np.zeros((1,3)),PLAN)[0]
    assert good["baseline_pass"] and good["support"]["ignored_loss_ms"] == 2
    a["force"][3000,:,0] = 0.
    a["height"][3000:,0] = .2
    bad = review_arrays(a,rows,np.zeros((1,3)),PLAN)[0]
    assert not bad["baseline_pass"] and bad["support"]["retained_loss_ms"] == 5
    assert bad["support"]["standing_start_sample"] == 0


@pytest.mark.parametrize("role", ["forward","left","height_high","forward_height"])
def test_never_executed_motion_or_height_request_cannot_pass_as_standing(role):
    rows = [case(PLAN,role)]
    result = review_arrays(arrays_for(rows),rows,np.zeros((1,3)),PLAN)[0]
    assert not result["baseline_pass"]


def test_initial_configuration_keeps_original_material_and_current_wheel_ratio(tmp_path):
    cfg, settings = configuration(tmp_path,PLAN)
    contract = json.loads((tmp_path/"task_contract.json").read_text())
    assert settings["domains_enabled"] is False and "standing009_curriculum" not in settings
    assert cfg["contact_material"]["wheel_static_friction"] == .1
    assert cfg["contact_material"]["ground_static_friction_range"] == [.5,.5]
    assert cfg["wheel_reduction"]["gear_ratio"] == 15.62
    assert contract["precise_standing_objective"] and contract["static_load_initialization"]
    assert contract["wheel_action_parameterization"] == "encoder_velocity_plus_learned_increment_v1"
    assert contract["leg_velocity_feedback_architecture"]["neutral_raw_features"] == [16,17,18,19]


def test_final_heldout_domains_use_all_randomized_lanes_and_keep_user_fixed_parameters(tmp_path):
    cfg, settings = configuration(tmp_path,PLAN,task="tracking",profile="requested",strength=1.,fully_randomized=True)
    assert cfg["dynamics_randomization"]["enabled_fraction"] == 1.
    assert settings["nominal_fraction"] == 0.
    assert settings["leg_kp_scale_range"] == settings["leg_kd_scale_range"] == settings["spring_strength_range"] == [1.,1.]
    assert cfg["dynamics_randomization"]["inertia_scale"] == [.94,1.06]
    assert settings["imu_mount"]["wobble_degrees"] == [0.,0.]
    assert cfg["contact_material"]["ground_static_friction_range"] == [.3,.5]
    assert settings["sensor_delay_policy_ticks"] == [0,1]

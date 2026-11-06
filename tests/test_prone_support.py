from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from wheeled_tasks.chassis import prone_support as task


def plan():
    return json.loads((ROOT/"contracts/wheel_leg_prone_support_v1.json").read_text())


def test_explicit_scope_and_mechanical_bindings():
    p = plan()
    c = task.validate_plan(p)
    path = ROOT/c["reset_parameters"]["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == c["reset_parameters"]["sha256"]
    assert "rollover_training" not in p
    assert p["encoder_increment_actions"]["schema"] == "wheel-leg-encoder-increment-actions-v2"


def test_resolved_small_budget_keeps_all_prone_births_and_physical_goal_clock(tmp_path):
    from train_prone_motion import configuration, learner_config
    from wheeled_tasks.chassis.prone_motion_cli import resolve_plan
    p = resolve_plan(plan(),num_envs=64,max_iterations=250,domain_randomization="off")
    cfg,settings,_ = configuration(tmp_path,p,performance_backend="eager")
    learner = learner_config(p,"cpu")
    assert settings["prone_motion_task"]["roles_per20"] == ["prone"]*20
    assert settings["stable_seconds"] == .01  # Actual3s test belongs to native-step capture.
    assert settings["reward_gamma"] == learner["algorithm"]["gamma"]
    assert learner["algorithm"]["gamma"] == pytest.approx(math.exp(-.01/10.))
    assert cfg["target_num_envs"] == 64 and p["max_updates"] == 250
    assert p["prone_support"]["goal_heights_m"] == [.23,.265,.305]
    assert "global_curriculum" not in p and "rollover_training" not in p


@pytest.mark.parametrize("change", ["rollover", "clock", "goal", "actor", "script", "hold", "contact_command"])
def test_reject_silent_task_changes(change):
    p = plan()
    if change == "rollover":
        p["rollover_training"] = {"schema":"other"}
    elif change == "clock":
        p["timing"]["physics_dt"] = .005
    else:
        key, value = {"goal":("goal_heights_m",[.23,.32]), "actor":("actor_inputs_added",True),
            "script":("scripted_actions",True), "hold":("capture_hold_s",.01),
            "contact_command":("goal_changes","per_lane_contact_switch")}[change]
        p["prone_support"][key] = value
    with pytest.raises(ValueError):
        task.validate_plan(p)


def test_all_births_are_belly_grounded_and_independent_closed_chain_configurations():
    from build_full_vehicle_training import ResetGeometry, geometry
    p = plan()
    bank = json.loads((ROOT/p["prone_support"]["reset_parameters"]["file"]).read_text())
    bundle = ROOT/"model/wheel_leg_v2.0.2/friction_v1/training"
    spec = json.loads((bundle/"model_spec.json").read_text())
    builder = ResetGeometry(spec,bundle)
    exact = json.loads((ROOT/p["chassis_collision"]["geometry"]["file"]).read_text())
    minimum_base = np.array(exact["source_extremes"])[:,2].min()
    by_split = {name:[] for name in ("train","holdout")}
    for pose in bank["poses"]:
        q = pose["full_joint_positions"]
        assert pose["root_quaternion_xyzw"] == [0.,0.,0.,1.]
        assert pose["root_position_m"][:2] == [0.,0.]
        assert 40. < min(pose["knee_deg"]) < max(pose["knee_deg"]) < 110.
        assert np.linalg.norm(geometry.closure_error(spec,q).reshape(-1,3),axis=1).max() < 1e-7
        minima = [v[2] for name,v in builder.bounds(q,np.eye(3)) if name != "base_link"]+[minimum_base]
        assert min(minima)+pose["root_position_m"][2] == pytest.approx(.002,abs=1e-8)
        assert minimum_base+pose["root_position_m"][2] == pytest.approx(.002,abs=1e-8)
        assert pose["recovery"]["native_passive_admitted"] is False
        by_split[pose["recovery"]["split"]].append(pose)
    assert [len(by_split[n]) for n in ("train","holdout")] == [32,8]
    for rows in by_split.values():
        assert {p["recovery"]["leg_class"] for p in rows} == set(task.LEG_CLASSES)
        assert all(p["recovery"]["family"] == "belly_down" for p in rows)
    assert len({tuple(p["full_joint_positions"].values()) for p in bank["poses"]}) == 40


def sample_scores(knee=(90.,90.), error=(0.,0.), height=.183, goal=.305):
    return task.scores(torch.tensor([[0.,0.,-1.]]),torch.tensor([height]),
        torch.zeros(1,3),torch.zeros(1,3),torch.tensor([knee]),torch.tensor([error]),
        torch.tensor([[.1,.1]]),torch.tensor([goal]),plan()["prone_support"])


def test_opposite_wheel_errors_do_not_cancel_and_retraction_releases():
    aligned = sample_scores(knee=(50.,50.))
    opposite = sample_scores(knee=(50.,50.),error=(.2,-.2))
    assert aligned["alignment"].item() > .9
    assert opposite["alignment"].item() < .01
    assert aligned["folding_weight"].item() < .05
    standing = sample_scores(knee=(100.,100.),height=.305)
    assert standing["folding_weight"].item() == 0.
    assert task.potential(standing).item() > task.potential(opposite).item()


def conditions(**overrides):
    args = dict(gravity=torch.tensor([[0.,0.,-1.]]),height=torch.tensor([.305]),
        velocity=torch.zeros(1,3),omega=torch.zeros(1,3),motor_velocity=torch.zeros(1,6),
        wheel_x_error=torch.zeros(1,2),clearance=torch.zeros(1,2),
        wheel_load_fraction=torch.tensor([[.5,.5]]),nonwheel_load_fraction=torch.zeros(1),
        excursion=torch.zeros(1),goal=torch.tensor([.305]),spec=plan()["prone_support"])
    args.update(overrides)
    return task.capture_conditions(**args)


@pytest.mark.parametrize("which",["clearance","unloaded","belly","low","moving","one_wheel_fore"])
def test_completion_requires_default_height_quiet_individual_wheel_support(which):
    assert all(bool(v.all()) for v in conditions())
    bad = {"clearance":dict(clearance=torch.tensor([[0.,.02]])),
        "unloaded":dict(wheel_load_fraction=torch.zeros(1,2)),
        "belly":dict(nonwheel_load_fraction=torch.tensor([.5])),
        "low":dict(height=torch.tensor([.23])),
        "moving":dict(velocity=torch.tensor([[.2,0.,0.]])),
        "one_wheel_fore":dict(wheel_x_error=torch.tensor([[.09,-.09]]))}[which]
    assert not all(bool(v.all()) for v in conditions(**bad))


def test_native_support_gap_two_ms_tolerance_and_no_airborne_start():
    good = torch.ones(1,dtype=torch.bool)
    bad = ~good
    streak,gap = torch.zeros(1,dtype=torch.long),torch.zeros(1,dtype=torch.long)
    streak,gap = task.advance_capture(streak,gap,good,bad,good,.001,.002)
    assert streak.item() == 0
    streak,gap = task.advance_capture(streak,gap,good,good,good,.001,.002)
    assert streak.item() == 1
    for expected in (2,3,0):
        streak,gap = task.advance_capture(streak,gap,good,bad,good,.001,.002)
        assert streak.item() == expected


def test_per_class_actual_outcomes_unlock_only_future_goal_levels():
    course = task.HeightCurriculum(7,plan()["prone_support"])
    for _ in range(64):
        course.observe(0,"front_front",True)
    assert course.phase == 0
    for cls in task.LEG_CLASSES:
        for _ in range(16):
            course.observe(0,cls,False)
    assert course.phase == 0
    for cls in task.LEG_CLASSES:
        for _ in range(16):
            course.observe(0,cls,True)
    assert course.phase == 1
    for cls in task.LEG_CLASSES:
        for _ in range(32):
            course.observe(0,cls,True)
    assert course.phase == 1
    assert not any(course.window.values())
    for cls in task.LEG_CLASSES:
        for n in range(16):
            course.observe(1,cls,n < 12)
    assert course.phase == 2
    assert course.state_dict()["height_m"] == .305


def test_normal_leg_support_is_not_an_impact_cost():
    s = sample_scores()
    common = dict(s=s,velocity=torch.zeros(1,3),omega=torch.zeros(1,3),torque=torch.zeros(1,6),
        motor_velocity=torch.zeros(1,6),action_delta=torch.zeros(1),excursion=torch.zeros(1),
        wheel_load_fraction=torch.zeros(1,2),nonwheel_load_fraction=torch.ones(1),
        impulse_peak=torch.tensor([[2.,0.,2.,0.]]),weight=torch.tensor([200.]),
        recovered=torch.zeros(1,dtype=torch.bool))
    reward = task.densities(**common)
    assert reward["impulse_excess"].item() == 0.
    common["impulse_peak"] = torch.tensor([[2.,0.,20.,0.]])
    assert task.densities(**common)["impulse_excess"].item() < 0.


def test_action_change_cost_keeps_independent_robot_lanes():
    s = {name:value.expand(3) for name,value in sample_scores().items()}
    reward = task.densities(s,torch.zeros(3,3),torch.zeros(3,3),torch.zeros(3,6),
        torch.zeros(3,6),torch.tensor([0.,1.,4.]),torch.zeros(3),torch.zeros(3,2),
        torch.zeros(3),torch.zeros(3,4),torch.full((3,),200.),torch.zeros(3,dtype=torch.bool))
    assert all(value.shape == (3,) for value in reward.values())
    assert torch.allclose(reward["action_rate"],torch.tensor([0.,-.01,-.04]))


def test_default_height_is_distinct_from_low_curriculum_outcomes():
    p = plan()
    assert p["prone_support"]["goal_heights_m"][-1] == .305
    assert "global_curriculum" not in p
    assert p["roles_per20"] == ["prone"]*20
    assert p["prone_support"]["capture_hold_s"] == 3.
    source = (ROOT/"scripts/evaluate_prone_support.py").read_text()
    assert 'target=[0.,0.,.305]' in source
    assert 'post_standing_10s' in source
    assert 'actual_16001_handoff_time_s=None' in source

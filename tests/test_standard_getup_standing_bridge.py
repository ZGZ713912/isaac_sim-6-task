"""Prevent paid belly-supported crouching and connect getup to quiet standing."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis import prone_motion as task
from wheeled_tasks.chassis.multi_pose_recovery import recovery_terms
from wheeled_tasks.chassis.standard_getup_reward import (
    contact_reward_contract, dense_terms, guidance, ordered_potential,
    standing_reward_contract, validate,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4.json").read_text())
PREVIOUS = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4_before_standing_bridge_20261103.json").read_text())
SPEC = standing_reward_contract()


def state(*, height=.19, knees=103., error=.35, tilt=0., rate=0., chassis=1., nonwheel=1.,
          clearance=0., wheel_peak=.5):
    angle = math.radians(tilt)
    return dict(height=torch.tensor([height]),
        gravity=torch.tensor([[math.sin(angle), 0., -math.cos(angle)]]),
        velocity=torch.zeros(1, 3), omega=torch.tensor([[0., rate, 0.]]),
        torque=torch.zeros(1, 6), action_delta=torch.zeros(1),
        recovery_support_error_m=torch.tensor([error]),
        recovery_fore_aft_error_m=torch.tensor([error]),
        recovery_knee_deg=torch.tensor([[knees, knees]]), recovery_geometry_valid=torch.tensor([True]),
        recovery_wheel_load_fraction=torch.tensor([[.5, .5]]),
        recovery_wheel_clearance_m=torch.full((1, 2), clearance),
        recovery_chassis_load_fraction=torch.tensor([chassis]),
        recovery_nonwheel_load_fraction=torch.tensor([nonwheel]),
        recovery_peak_wheel_load_fraction=torch.tensor([wheel_peak]))


def test_only_reward_changes_and_all_previous_course_physics_budget_are_preserved():
    restored = deepcopy(PLAN)
    assert restored["multi_pose_recovery"]["dense_reward"] == SPEC
    restored["multi_pose_recovery"]["dense_reward"] = contact_reward_contract()
    assert restored == PREVIOUS
    assert task.validate(PLAN) == PLAN
    assert task.validate(PREVIOUS) == PREVIOUS
    assert validate(SPEC) == SPEC
    bad = dict(SPEC, chassis_contact_weight=6.)
    with pytest.raises(ValueError, match="versioned"):
        validate(bad)


def test_initial_prone_chassis_contact_is_permitted_without_new_cost():
    s = state()
    assert guidance(s, SPEC)["preparation_score"].item() == 0.
    assert dense_terms(s, SPEC)["recovery_chassis_contact"].item() == 0.
    bad = task.failures(s["gravity"], s["height"], torch.tensor([210.]),
        torch.zeros(1), torch.ones(1), torch.tensor([True]), torch.tensor([False]), PLAN)
    assert not any(v.any() for v in bad.values())


def test_contact_cost_is_progressive_and_monotonic_after_both_knees_tuck():
    knees = [66., 65., 64.99, 60., 55., 50., 49.]
    scores = [guidance(state(knees=k, chassis=.17), SPEC)["preparation_score"].item() for k in knees]
    assert scores[:2] == [0., 0.] and scores[-2:] == [1., 1.]
    assert all(a < b for a, b in zip(scores[1:5], scores[2:6]))
    assert scores[2] < .00001
    s = state(knees=45., chassis=.17)
    assert dense_terms(s, SPEC)["recovery_chassis_contact"].item() == pytest.approx(-2.04)
    s["recovery_knee_deg"][0, 1] = 103.
    assert guidance(s, SPEC)["preparation_score"].item() == 0.
    costs = [dense_terms(state(knees=45., chassis=f), SPEC)["recovery_chassis_contact"].item()
             for f in [0., .05, .17, .5, 1.]]
    assert costs[0] == 0. and all(b < a for a, b in zip(costs, costs[1:]))


def test_support_capture_keeps_cost_armed_during_extension_without_refolding():
    for knees in [45., 70., 103.]:
        s = state(height=.25, knees=knees, error=.02, chassis=.1)
        g, t = guidance(s, SPEC), dense_terms(s, SPEC)
        assert g["preparation_score"].item() == g["height_gate"].item() == 1.
        assert t["recovery_chassis_contact"].item() == pytest.approx(-1.2)
        assert t["recovery_fold_error"].item() == 0.


def test_chassis_cost_does_not_deadlock_lift_while_body_is_still_touching():
    grounded = state(height=.22, knees=45., error=.02, chassis=.5)
    airborne_body = dict(grounded, recovery_chassis_load_fraction=torch.zeros(1),
                         recovery_nonwheel_load_fraction=torch.zeros(1))
    for s in [grounded, airborne_body]:
        assert guidance(s, SPEC)["height_gate"].item() == 1.
        assert dense_terms(s, SPEC)["recovery_height"].item() > 0.
    for name in ["recovery_height", "recovery_height_error"]:
        assert torch.equal(dense_terms(grounded, SPEC)[name], dense_terms(airborne_body, SPEC)[name])


def test_measured_low_crouch_cannot_earn_preparation_density_anymore():
    # Representative checkpoint1000 tail; this is arithmetic, not a PPO ablation.
    s = state(height=.2217, knees=42.7, error=.046, tilt=14.5, rate=.017,
              chassis=.17, nonwheel=.19)
    t = recovery_terms(s, PLAN["multi_pose_recovery"], torch.zeros(1))
    assert t["recovery_support"].item() == t["recovery_capture_stability"].item() == 0.
    assert t["recovery_quiet_standing"].item() < 1e-12
    assert sum(t.values()).item() < -2.
    assert ordered_potential(s, SPEC).item() > 0.
    # Progress shaping retains the preparation sequence instead of prescribing actions.
    values = [ordered_potential(v, SPEC).item() for v in [state(),
        state(knees=45., error=.18), state(knees=45., error=.02),
        state(height=.305, knees=70., error=.02)]]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_true_standing_receives_the_same_quiet_reward_before_and_after_completion():
    s = state(height=.305, knees=70., error=.02, chassis=0., nonwheel=0.)
    before = dense_terms(s, SPEC)["recovery_quiet_standing"]
    after = task.reward_terms(s["gravity"], s["height"], s["velocity"], s["omega"],
        torch.tensor([[0., 0., .305]]), s["torque"], torch.zeros(1, 6), s["action_delta"],
        torch.zeros(1), torch.zeros(1), torch.tensor([False]), torch.tensor([True]))["quiet"]
    assert before.item() == after.item() == 4.
    for field, index in [("velocity", 0), ("velocity", 1), ("omega", 1), ("omega", 2)]:
        moving = {k:v.clone() for k, v in s.items()}
        moving[field][0, index] = .3
        assert dense_terms(moving, SPEC)["recovery_quiet_standing"].item() < before.item()
        assert sum(dense_terms(moving, SPEC).values()).item() < sum(dense_terms(s, SPEC).values()).item()
    for field in ["recovery_nonwheel_load_fraction", "recovery_wheel_load_fraction"]:
        v = .2 if field == "recovery_nonwheel_load_fraction" else 0.
        grounded = dict(s, **{field:torch.full_like(s[field], v)})
        assert dense_terms(grounded, SPEC)["recovery_quiet_standing"].item() < 1e-12


def test_shared_quiet_quality_preserves_the_original_standing_expression():
    generator = torch.Generator().manual_seed(107)
    gravity = torch.randn(64, 3, generator=generator)*.03+torch.tensor([0., 0., -1.])
    height = torch.randn(64, generator=generator)*.01+.305
    velocity = torch.randn(64, 3, generator=generator)*.03
    omega = torch.randn(64, 3, generator=generator)*.1
    target = torch.rand(64, generator=generator)*.02+.295
    level = (gravity-gravity.new_tensor([0., 0., -1.])).square().sum(-1)
    original = torch.exp(-((height-target)/.02).square()-level/math.sin(math.radians(5.))**2
        -(velocity[:, :2]/.08).square().sum(-1)-(omega/.5).square().sum(-1))
    assert torch.equal(task.quiet_standing_quality(gravity, height, velocity, omega, target), original)


@pytest.mark.parametrize("field", ["recovery_chassis_load_fraction", "recovery_nonwheel_load_fraction",
                                    "recovery_peak_wheel_load_fraction"])
def test_unknown_load_cannot_award_quiet_standing_or_produce_nonfinite_cost(field):
    s = state(height=.305, knees=70., error=.02, chassis=0., nonwheel=0.)
    s[field][0] = float("nan")
    t = dense_terms(s, SPEC)
    assert all(torch.isfinite(v).all() for v in t.values())
    assert t["recovery_quiet_standing"].item() == 0.
    assert t["recovery_chassis_contact"].item() < 0.


def test_actual_completion_starts_quiet_hold_and_body_contact_refall():
    target = torch.tensor([[1., .5, .305]])
    prone = torch.tensor([True])
    for recovered in [float("nan"), 50.]:
        command = task.commands(target, torch.tensor([70.]), torch.tensor([recovered]), prone, PLAN)
        assert torch.equal(command, torch.tensor([[0., 0., .305]]))
    s = state(height=.305)
    before = task.failures(s["gravity"], s["height"], torch.tensor([30.]), torch.zeros(1),
        torch.ones(1), torch.tensor([True]), torch.tensor([False]), PLAN)
    after = task.failures(s["gravity"], s["height"], torch.tensor([30.]), torch.zeros(1),
        torch.ones(1), torch.tensor([False]), torch.tensor([False]), PLAN)
    assert not before["refall"].item() and after["refall"].item()


def test_guidance_is_idempotent_and_uses_no_elapsed_clock_or_mutation():
    s = state(knees=45.)
    copy = {k:v.clone() for k,v in s.items()}
    first = guidance(s, SPEC)
    s["elapsed"] = torch.tensor([200.])
    second = guidance(s, SPEC)
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert all(torch.equal(s[k], copy[k]) for k in copy)


def test_fold_tilt_and_rate_costs_favor_small_attitude_without_joint_locking():
    for name, values, make in [
            ("recovery_fold_upright", [0., 5., 15., 46.], lambda x:state(tilt=x)),
            ("recovery_fold_body_rate", [0., .3, 1., 5.], lambda x:state(rate=x))]:
        costs = [dense_terms(make(v), SPEC)[name].item() for v in values]
        assert costs[0] == 0. and all(b < a for a,b in zip(costs,costs[1:]))
    # The additional tilt weight is active before support capture, on top of 6.
    s = state(tilt=46.)
    t = dense_terms(s, SPEC)
    assert t["recovery_fold_upright"].item() == pytest.approx(2.*t["recovery_upright"].item())
    s["motor_position"] = torch.tensor([[1., -2., 0., -1., 2., 0.]])
    assert all(torch.equal(v,dense_terms(s,SPEC)[k]) for k,v in t.items())


def test_wheel_skim_targets_world_clearance_and_does_not_reward_airborne_unloading():
    costs = [dense_terms(state(clearance=h, wheel_peak=0.), SPEC)["recovery_ground_skim"].item()
             for h in [0., .001, .003, .01, .05, .12]]
    assert costs[:3] == [0., 0., 0.]
    assert all(b < a for a,b in zip(costs[2:],costs[3:]))
    # A single lifted wheel still incurs a cost. Zero load alone is insufficient.
    s = state(clearance=0., wheel_peak=0.)
    s["recovery_wheel_clearance_m"][0,1] = .1
    assert dense_terms(s,SPEC)["recovery_ground_skim"].item() < -1.


def test_light_load_is_soft_before_capture_and_allows_full_weight_after_capture():
    costs = [dense_terms(state(wheel_peak=f), SPEC)["recovery_fold_wheel_load"].item()
             for f in [0., .05, .1, .2, .5, 1.]]
    assert costs[:3] == [0., 0., 0.]
    assert all(b < a for a,b in zip(costs[2:],costs[3:]))
    # The minimum physical load used by the lift gate must not hide a peak.
    s = state(wheel_peak=1.)
    s["recovery_wheel_load_fraction"].zero_()
    assert dense_terms(s,SPEC)["recovery_fold_wheel_load"].item() < -1.
    # Once wheel support is under the body, even extended legs may carry weight.
    for knees in [45.,70.,103.]:
        s = state(height=.305,knees=knees,error=.02,chassis=0.,nonwheel=0.,wheel_peak=1.)
        g,t = guidance(s,SPEC),dense_terms(s,SPEC)
        assert g["fold_phase"].item() == 0.
        assert t["recovery_fold_wheel_load"].item() == 0.
        assert g["height_gate"].item() == 1.

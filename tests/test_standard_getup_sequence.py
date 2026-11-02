"""Reward semantics for tuck, move support below the body, then extend.

These are state/reward checks, not proofs of a physically executed getup.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.multi_pose_recovery import recovery_terms
from wheeled_tasks.chassis.prone_motion import validate as validate_plan
from wheeled_tasks.chassis.standard_getup_reward import (
    dense_terms, geometry_inputs, guidance, ordered_potential,
    ordered_reward_contract, contact_reward_contract, reward_contract, validate,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4_before_standing_bridge_20261103.json").read_text())
OLD = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4_before_sequence_20261103.json").read_text())
SPEC = contact_reward_contract()


def state(height=.19, knees=103., error=.35, tilt=0., rate=0., fore_aft=None, load=.5, clearance=0.):
    angle = math.radians(tilt)
    return dict(height=torch.tensor([height]),
        gravity=torch.tensor([[math.sin(angle), 0., -math.cos(angle)]]),
        omega=torch.tensor([[0., rate, 0.]]), torque=torch.zeros(1, 6), action_delta=torch.zeros(1),
        recovery_support_error_m=torch.tensor([error]),
        recovery_fore_aft_error_m=torch.tensor([error if fore_aft is None else fore_aft]),
        recovery_knee_deg=torch.tensor([[knees, knees]]), recovery_geometry_valid=torch.tensor([True]),
        recovery_wheel_load_fraction=torch.tensor([[load, load]]),
        recovery_wheel_clearance_m=torch.tensor([[clearance, clearance]]))


def score(s):
    return sum(recovery_terms(s, PLAN["multi_pose_recovery"], torch.zeros(1)).values()).item()


def test_only_versioned_reward_changes_and_old_contract_still_loads():
    assert PLAN["multi_pose_recovery"]["dense_reward"] == SPEC
    restored = deepcopy(PLAN)
    restored["multi_pose_recovery"]["dense_reward"] = OLD["multi_pose_recovery"]["dense_reward"]
    assert restored == OLD
    validate_plan(PLAN)
    validate_plan(OLD)
    assert validate(reward_contract()) == OLD["multi_pose_recovery"]["dense_reward"]
    for config in [SPEC, reward_contract()]:
        bad = deepcopy(config); bad["upright_error_weight"] *= 2
        with pytest.raises(ValueError, match="versioned"):
            validate(bad)
    assert 40. < SPEC["fold_target_max_deg"] < 110.


def test_preparation_has_no_height_pressure_in_dense_or_potential_rewards():
    s = state()
    s["height"].requires_grad_()
    terms = dense_terms(s, SPEC)
    assert terms["recovery_height"].item() == terms["recovery_height_error"].item() == 0.
    total = sum(terms.values())+ordered_potential(s, SPEC)
    total.sum().backward()
    assert s["height"].grad.item() == 0.
    raised = state(height=.28)
    assert score(raised) < score(s)
    assert ordered_potential(raised, SPEC).item() == ordered_potential(s, SPEC).item()


def test_preparing_tucked_support_and_standing_are_incentivized_in_order():
    initial = state()
    tucked = state(knees=45., error=.18)
    captured = state(height=.20, knees=45., error=.025, tilt=5.)
    standing = state(height=.305, knees=65., error=.015, tilt=2.)
    values = [ordered_potential(s, SPEC).item() for s in (initial, tucked, captured, standing)]
    assert all(b > a for a, b in zip(values, values[1:]))
    assert score(tucked) > score(initial)
    assert score(captured) > score(tucked)
    assert score(standing) > score(captured)
    # Long-leg rotation before tucking must be worse than quiet preparation.
    assert score(state(height=.25, knees=103., error=.18, tilt=20., rate=5.)) < score(initial)


def test_support_capture_releases_fold_cost_so_the_legs_can_extend():
    for knees in [45., 65., 103.]:
        s = state(height=.305, knees=knees, error=.02)
        assert guidance(s, SPEC)["height_gate"].item() == 1.
        assert dense_terms(s, SPEC)["recovery_fold_error"].item() == 0.
        assert ordered_potential(s, SPEC).item() == pytest.approx(7.)
    # Before capture, a single extended side still prevents a full tuck score.
    s = state(knees=45., error=.18)
    s["recovery_knee_deg"][0, 1] = 103.
    assert guidance(s, SPEC)["fold_score"].item() == 0.


def test_height_gate_is_continuous_and_rejects_large_tilt():
    gates = [guidance(state(error=e), SPEC)["height_gate"].item() for e in [.13, .12, .10, .08, .06, .04, .03]]
    assert gates[:2] == [0., 0.] and gates[-2:] == [1., 1.]
    assert all(b > a for a, b in zip(gates[1:5], gates[2:6]))
    for tilt in [30., 90., 127., 180.]:
        s = state(height=.305, knees=45., error=.01, tilt=tilt)
        assert guidance(s, SPEC)["height_gate"].item() == 0.
        assert dense_terms(s, SPEC)["recovery_height"].item() == 0.


def test_existing_tilt_and_body_rate_costs_are_stronger_not_deleted():
    s = state(height=.305, knees=45., error=.02, tilt=40., rate=5.)
    new, old = dense_terms(s, SPEC), dense_terms(s, reward_contract())
    assert new["recovery_upright"].item() == pytest.approx(3.*old["recovery_upright"].item())
    assert new["recovery_body_rate"].item() == pytest.approx(5.*old["recovery_body_rate"].item())


def test_invalid_geometry_cannot_earn_capture_or_fold_credit():
    s = state(height=.305, knees=45., error=.01)
    s["recovery_geometry_valid"][0] = False
    s["recovery_support_error_m"][0] = float("nan")
    s["recovery_knee_deg"][0] = float("nan")
    terms = dense_terms(s, SPEC)
    assert all(torch.isfinite(v).all() for v in terms.values())
    g = guidance(s, SPEC)
    assert g["height_gate"].item() == g["fold_score"].item() == 0.
    assert terms["recovery_support"].item() == ordered_potential(s, SPEC).item() == 0.


@pytest.mark.parametrize("heading", [0., math.pi/2., math.pi, -math.pi/2.])
def test_two_wheels_cannot_fake_capture_with_their_midpoint(heading):
    root = torch.tensor([[0., 0., .19, 0., 0., math.sin(heading/2.), math.cos(heading/2.)]])
    # Both wheels have the correct lateral separation, but one is ahead of
    # the COM and the other behind it. Their midpoint alone appears aligned.
    xyz = torch.tensor([[[.15, .22, .06], [-.15, -.22, .06]]])
    c, s = math.cos(heading), math.sin(heading)
    rotation = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    xyz = xyz@rotation.T
    route = dict(encoder_wheel_world_m=xyz, com_world_m=torch.tensor([[0., 0., .19]]),
        knee_deg=torch.tensor([[45., 45.]]), encoder_valid=torch.tensor([[True, True]]))
    inputs = geometry_inputs(route, root)
    assert inputs["recovery_fore_aft_error_m"].item() == pytest.approx(.15)
    sample = state(knees=45., error=0.)
    sample.update(inputs)
    assert guidance(sample, SPEC)["height_gate"].item() == 0.


def test_guidance_does_not_mutate_state_or_depend_on_elapsed_time():
    s = state(knees=45., error=.07, tilt=5.)
    original = {k:v.clone() for k,v in s.items()}
    s["elapsed"] = torch.tensor([.1])
    before = dense_terms(s, SPEC)
    s["elapsed"].fill_(19.)
    after = dense_terms(s, SPEC)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(torch.equal(original[k], s[k]) for k in original)


def test_geometry_alone_cannot_pay_for_airborne_or_single_wheel_lift():
    for load, clearance in [(0., .01), (.5, .01), (0., 0.)]:
        s = state(height=.25, knees=45., error=.02, load=load, clearance=clearance)
        assert guidance(s, SPEC)["height_gate"].item() == 0.
        assert dense_terms(s, SPEC)["recovery_height"].item() == 0.
        assert dense_terms(s, SPEC)["recovery_capture_stability"].item() == 0.
    s = state(height=.25, knees=45., error=.02)
    s["recovery_wheel_load_fraction"][0, 1] = 0.
    assert guidance(s, SPEC)["height_gate"].item() == 0.
    s["recovery_wheel_load_fraction"][0, 1] = float("nan")
    assert all(torch.isfinite(v).all() for v in dense_terms(s, SPEC).values())
    assert guidance(s, SPEC)["height_gate"].item() == 0.


def test_body_rotation_must_be_arrested_before_lift_without_requiring_refolding():
    moving = state(height=.20, knees=65., error=.02, rate=2.9)
    g = guidance(moving, SPEC)
    assert g["dual_wheel_bearing"].item() == g["support_capture"].item() == 1.
    assert g["height_gate"].item() == 0.
    assert dense_terms(moving, SPEC)["recovery_fold_error"].item() == 0.
    scores = [score(state(height=.20, knees=65., error=.02, rate=r)) for r in [2.9, 1.4, .8, .3, 0.]]
    assert all(b > a for a, b in zip(scores, scores[1:]))
    # A bounded quality bonus guides arrest even above the lift-rate cutoff.
    assert dense_terms(moving, SPEC)["recovery_capture_stability"].item() > 0.
    moving["omega"][0] = torch.tensor([2.9, 0., 0.])
    assert guidance(moving, SPEC)["height_gate"].item() == 0.


def test_load_and_rate_gates_are_smooth_and_wheel_spin_is_not_a_reward():
    gates = [guidance(state(error=.02, load=v), SPEC)["height_gate"].item() for v in [0., .03, .075, .12, .15, .5]]
    assert gates[0] == 0. and gates[-2:] == [1., 1.]
    assert all(b > a for a, b in zip(gates[:4], gates[1:5]))
    s = state(height=.20, knees=45., error=.02)
    before = dense_terms(s, SPEC)
    s["dq"] = torch.tensor([[0., 0., 200., 0., 0., -200.]])
    assert all(torch.equal(before[k], dense_terms(s, SPEC)[k]) for k in before)


def test_untrained_v2_candidate_still_has_exact_geometry_only_semantics():
    c = ordered_reward_contract()
    assert validate(c) == c
    s = state(height=.305, knees=65., error=.02, load=0.)
    assert guidance(s, c)["height_gate"].item() == 1.
    assert ordered_potential(s, c).item() == pytest.approx(6.)
    assert "recovery_capture_stability" not in dense_terms(s, c)

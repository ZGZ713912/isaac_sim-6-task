"""Regressions for sample starvation, native angle wrapping and lift guidance."""
from collections import Counter
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.global_curriculum import GlobalCurriculum
from wheeled_tasks.chassis.multi_pose_recovery import recovery_terms
from wheeled_tasks.chassis.native_wheel_encoder import NativeWheelEncoder, encoder_contract
from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery
from wheeled_tasks.chassis.standing_local_odometry import LocalStandingOdometry, observation_contract
from wheeled_tasks.chassis.standard_getup_reward import dense_terms, reward_contract

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "contracts/wheel_leg_prone_motion_v4.json").read_text())
LEGACY_PLAN = json.loads((ROOT / "contracts/wheel_leg_prone_motion_v4_before_sequence_20261103.json").read_text())


def course(plan=PLAN):
    return GlobalCurriculum(plan, RecoveryMastery(plan["multi_pose_recovery"]["mastery_gate"]))


def test_getup_practice_and_cold_tests_survive_absent_or_lost_standing_pass():
    c = course()
    for learned, retained in ((0, 0), (1, 1), (1, 0)):
        c.levels["standing"], c.retained_levels["standing"] = learned, retained
        counts = Counter(row["curriculum_skill"] for row in c.training_descriptors(256, 10))
        assert counts["recovery"] == 77
        assert ("belly", 0) in c.jobs(250)
        assert not c.recovery.unlocked and c.domain_context() == ("nominal", 0.)
        if not retained:
            assert counts == {"standing": 179, "recovery": 77}
            assert c.learning_level("straight") is None


def test_historical_course_still_reproduces_the_observed_starvation():
    old = json.loads((ROOT / "contracts/wheel_leg_prone_motion_v4_before_getup_fix_20261102.json").read_text())
    c = course(old)
    c.levels["standing"] = 1
    assert {r["curriculum_skill"] for r in c.training_descriptors(256, 1)} == {"standing"}
    assert ("belly", 0) not in c.jobs(250)


def probe_fixture(plan):
    c = course(plan)
    rows = c.training_descriptors(256, 0)
    recovery = torch.tensor([r["recovery"] for r in rows])
    bank = torch.tensor([[0., .1, 0., 0., -.1, 0.], [.4, .6, 0., -.4, -.6, 0.]])
    poses = recovery.long()
    return SimpleNamespace(global_curriculum=c, num_envs=256, prone_reset_generation=0,
        ids=list(range(6)), bank_joints=bank, nominal_pose_index=0, pose_index=poses,
        course_roles=[r["role"] for r in rows], recovery_lane=recovery,
        course_targets=torch.tensor([r["target"] for r in rows]),
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=bank[poses].clone()))))


@pytest.mark.parametrize("historical", [False, True])
def test_native_launch_gate_accepts_the_actual_declared_course_including_legacy(historical):
    from evaluate_prone_motion import _global_initial_checks
    plan = (json.loads((ROOT / "contracts/wheel_leg_prone_motion_v4_before_getup_fix_20261102.json").read_text())
        if historical else PLAN)
    env = probe_fixture(plan)
    assert all(_global_initial_checks(env).values())
    assert int(env.recovery_lane.sum()) == (0 if historical else 77)


@pytest.mark.parametrize("fault", ["roles", "mask", "joint"])
def test_native_launch_gate_still_rejects_wrong_population_or_reset_geometry(fault):
    from evaluate_prone_motion import _global_initial_checks
    env = probe_fixture(PLAN)
    if fault == "roles": env.course_roles[-1] = "stand"
    if fault == "mask": env.recovery_lane[-1] = False
    if fault == "joint": env.robot.data.joint_pos.torch[0, 0] += .1
    assert not all(_global_initial_checks(env).values())


def test_two_physical_samples_lift_actual_four_pi_wrap_without_policy_fault():
    raw = torch.tensor([[6.2, -6.2]], dtype=torch.float64)
    encoder = NativeWheelEncoder(raw, encoder_contract())
    observer = LocalStandingOdometry(1, "cpu", .01, observation_contract())
    frame = torch.zeros(1, 35)
    frame[:, 20:22] = torch.tensor([[8., -8.]])
    enabled = torch.ones(1, dtype=torch.bool)
    observer.local_wheel_angles(frame, encoder.angle, enabled, 0)
    for tick in range(1, 101):
        true = raw + torch.tensor([[.4, -.4]]) * tick
        wrapped = torch.remainder(true + 2.*math.pi, 4.*math.pi) - 2.*math.pi
        encoder.update(wrapped)
        if tick % 2 == 0:
            value = observer.local_wheel_angles(frame, encoder.angle, enabled, tick//2)
            assert not observer.source_fault.any()
            torch.testing.assert_close(value.double(), true-raw, atol=2e-6, rtol=1e-6)
    assert encoder.wraps.min() >= 3
    previous = encoder.angle.clone(), encoder.wraps.clone()
    encoder.update(wrapped)
    assert torch.equal(previous[0], encoder.angle) and torch.equal(previous[1], encoder.wraps)


def test_unexplained_two_pi_jump_remains_an_encoder_fault_and_partial_reset_is_local():
    raw = torch.zeros(2, 2, dtype=torch.float64)
    encoder = NativeWheelEncoder(raw, encoder_contract())
    observer = LocalStandingOdometry(2, "cpu", .01, observation_contract())
    frame, enabled = torch.zeros(2, 35), torch.ones(2, dtype=torch.bool)
    observer.local_wheel_angles(frame, encoder.angle, enabled, 0)
    raw[0, 0] = 2.*math.pi
    encoder.update(raw)
    observer.local_wheel_angles(frame, encoder.angle, enabled, 1)
    assert observer.source_fault.tolist() == [True, False]
    encoder.update(torch.tensor([[.1, .2], [.3, .4]]))
    other = encoder.angle[1].clone()
    encoder.reset(torch.tensor([0]), torch.tensor([[0., 0.]]))
    assert not encoder.angle[0].any() and torch.equal(encoder.angle[1], other)


def state(height, gravity, support):
    return dict(height=torch.tensor(height), gravity=torch.tensor(gravity),
        recovery_support_error_m=torch.tensor(support), omega=torch.zeros(len(height), 3),
        torque=torch.zeros(len(height), 6), action_delta=torch.zeros(len(height)))


def test_dense_guidance_orders_lift_and_support_before_any_success_event():
    s = state([.19, .23, .27, .305], [[0., 0., -1.]]*4, [.2, .12, .05, 0.])
    # The first dense-reward contract remains reproducible for saved runs.
    terms = recovery_terms(s, LEGACY_PLAN["multi_pose_recovery"], torch.zeros(4))
    score = sum(terms.values())
    assert (score.diff() > 0).all() and score[0] < 0. < score[-1]
    assert (terms["rise_time"] == -1.).all()
    # Height remains a useful gradient outside the narrow completion band.
    s["height"].requires_grad_()
    sum(dense_terms(s, reward_contract()).values())[:3].sum().backward()
    assert (s["height"].grad[:3] > 0).all()


def test_ballistic_flipped_lift_is_worse_than_grounded_upright_lift():
    s = state([.305, .305, .305], [[0., 0., -1.], [1., 0., 0.], [0., 0., 1.]], [0., 0., 0.])
    terms = dense_terms(s, reward_contract())
    assert sum(terms.values()).tolist() == pytest.approx([3., -4., -8.])
    assert not terms["recovery_height"][1:].any() and not terms["recovery_support"][1:].any()


def test_cold_acceptance_and_mechanics_are_not_relaxed_to_pass_getup():
    old = json.loads((ROOT / "contracts/wheel_leg_prone_motion_v4_before_getup_fix_20261102.json").read_text())
    for name in ("closure_guard", "mechanical_inner_knee_range_deg", "timing", "solver_iterations",
                 "rise_deadline_seconds", "completion_hold_seconds", "pose_bank", "recovery_impact"):
        assert PLAN[name] == old[name]
    assert PLAN["global_curriculum"]["limits"] == old["global_curriculum"]["limits"]
    assert PLAN["multi_pose_recovery"]["mastery_gate"] == old["multi_pose_recovery"]["mastery_gate"]
    assert PLAN["multi_pose_recovery"]["failure_cost"] == 40.

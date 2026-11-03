"""Separate physical integration from the measured 200Hz feedback controller."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train_prone_motion import configuration
from wheeled_tasks.chassis.control_contract import (
    FINE_PD200_CONTROL_FILE, FINE_PD200_SCOPE, FINE_PD200_TIMING,
    load_chassis_control,
)
from wheeled_tasks.chassis.gimbal_motion import GimbalMotion, motion_contract
from wheeled_tasks.chassis.preflight import SubstepControlAudit
from wheeled_tasks.chassis.prone_motion import timing, validate
from wheeled_tasks.chassis.prone_motion_cli import resolve_plan, train_parser

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4_free_translation_v1.json").read_text())


def test_cli_explicit_feedback_clock_preserves_identified_actuators(tmp_path):
    args = train_parser().parse_args(["--physics-hz", "1000", "--pd-hz", "200"])
    base = deepcopy(BASE)
    # Configuration-only test; native admission must be measured separately.
    base["multi_pose_recovery"]["require_native_reset_admission"] = False
    plan = resolve_plan(base, physics_hz=args.physics_hz, pd_hz=args.pd_hz)
    assert timing(plan) == FINE_PD200_TIMING
    assert plan["simulation_timing_approximation"] == FINE_PD200_SCOPE
    assert validate(plan) == plan
    control = load_chassis_control(ROOT/FINE_PD200_CONTROL_FILE)
    recorded = json.loads((ROOT/BASE["wheel_model"]["recorded_control"]["file"]).read_text())
    assert control["actuators"] == recorded["actuators"]
    cfg, settings, _ = configuration(tmp_path, plan)
    assert (cfg["physics_dt"], cfg["policy_dt"], cfg["feedback_control_dt"]) == (.001, .01, .005)
    assert (cfg["v5_control"]["leg_kp"], cfg["v5_control"]["leg_kd"]) == (160., 2.5)
    assert settings["closure_gap_record_only"] and plan["getup_translation"] == BASE["getup_translation"]
    assert plan["multi_pose_recovery"]["dense_reward"] == BASE["multi_pose_recovery"]["dense_reward"]


@pytest.mark.parametrize("overrides", [dict(physics_hz=200, pd_hz=1000),
    dict(physics_hz=1000, pd_hz=201), dict(physics_hz=1000, pd_hz=True)])
def test_incompatible_feedback_rates_are_rejected(overrides):
    with pytest.raises(ValueError, match="PD|pd_hz"):
        resolve_plan(BASE, **overrides)


def test_control_audit_counts_held_steps_and_rejects_wrong_phase():
    audit = SubstepControlAudit(.001, .01, .005)
    audit.begin_policy(); audit.targets_decoded()
    with pytest.raises(RuntimeError, match="Held effort"):
        audit.feedback_held()
    for substep in range(10):
        if substep % 5 == 0:
            audit.feedback_control()
        else:
            with pytest.raises(RuntimeError, match="refreshed physical state"):
                audit.feedback_control()
            audit.feedback_held()
        audit.physics_updated(.001)
    result = audit.end_policy()
    assert (result["pd_updates"], result["feedback_refreshes"], result["target_decodes"]) == (2, 10, 1)
    assert result["pc_control_dt"] == .005 and result["effort_hold_physics_steps"] == 5


def test_production_motor_hold_uses_fresh_feedback_without_mutating_cache():
    tree = ast.parse((ROOT/"src/wheeled_tasks/chassis/robust_getup_env.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RobustGetupEnv")
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    loop = next(n for n in step.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                and n.target.id == "substep")
    stop = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "self.identified_wheels.enabled")
    prefix = ast.Module(body=loop.body[:stop], type_ignores=[])
    code = compile(prefix, "production_motor_hold_prefix", "exec")
    q = torch.zeros(1, 6); feedback = []
    def raw_pd(position, velocity, legs, wheels):
        feedback.append(position.clone())
        return torch.full_like(q, 10.*len(feedback))
    env = SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(
        joint_pos=SimpleNamespace(torch=q), joint_vel=SimpleNamespace(torch=torch.zeros_like(q)))),
        control_audit=SubstepControlAudit(.001, .01, .005),
        v5=SimpleNamespace(motor_efforts=raw_pd, current_motor_bounds=torch.full_like(q, 40.)),
        actuator_response=SimpleNamespace(apply_torque=lambda value: value.mul_(.5)),
        passive=torch.tensor([False]), finished=torch.tensor([False]))
    env.control_audit.begin_policy(); env.control_audit.targets_decoded()
    scope = dict(self=env, motor_ids=list(range(6)), legs=None, wheels=None)
    values = []
    for substep in range(10):
        scope["substep"] = substep
        exec(code, scope)
        values.append(scope["tau"].clone())
        q.add_(1.)
        env.control_audit.physics_updated(.001)
    assert len(feedback) == 2
    torch.testing.assert_close(feedback[0], torch.zeros_like(q))
    torch.testing.assert_close(feedback[1], torch.full_like(q, 5.))
    torch.testing.assert_close(torch.stack(values[:5]), torch.full((5, 1, 6), 5.))
    torch.testing.assert_close(torch.stack(values[5:]), torch.full((5, 1, 6), 10.))
    assert env.control_audit.end_policy()["pd_updates"] == 2


def test_gimbal_keeps_physical_target_clock_and_holds_only_servo_effort():
    gimbal = GimbalMotion(2, "cpu", .001, motion_contract(), 107)
    reference = GimbalMotion(2, "cpu", .001, motion_contract(), 107)
    held = None
    for step in range(10):
        q = torch.full((2, 2), step*.01); dq = torch.full_like(q, step*.1)
        enabled = torch.tensor([True, step < 7])
        raw = reference.step(q, dq, enabled).clone()
        if step % 5 == 0:
            held = raw.clone()
        expected = held*enabled[:, None]
        actual = gimbal.step(q, dq, enabled, feedback=step % 5 == 0)
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        torch.testing.assert_close(gimbal.target, reference.target, rtol=0., atol=0.)
        torch.testing.assert_close(gimbal.time, reference.time, rtol=0., atol=0.)
    gimbal.reset(torch.tensor([0]))
    assert not gimbal.effort[0].any()

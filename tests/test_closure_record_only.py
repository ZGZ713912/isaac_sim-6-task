"""Closure is diagnostic; independent behavior and physical failures still count."""
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluate_global_curriculum import assess_trace
from evaluate_recovery_mastery import evaluate
from train_prone_motion import configuration
from wheeled_tasks.chassis.prone_motion import closure_record_only, validate

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4.json").read_text())
PREVIOUS = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4_before_closure_record_only_20261103.json").read_text())


def test_explicit_monitor_is_the_only_plan_change():
    restored = deepcopy(PLAN)
    restored["closure_guard"] = PREVIOUS["closure_guard"]
    assert restored == PREVIOUS
    assert validate(PLAN) == PLAN and validate(PREVIOUS) == PREVIOUS
    assert closure_record_only(PLAN) and not closure_record_only(PREVIOUS)
    assert not closure_record_only({})


@pytest.mark.parametrize("update", [dict(schema="wheel-leg-closure-guard-v1"),
    dict(mode="off"), dict(mode=False), dict(peak_m=.006), dict(peak_m=float("inf"))])
def test_unknown_or_ambiguous_mode_is_rejected(update):
    bad = deepcopy(PLAN)
    bad["closure_guard"].update(update)
    with pytest.raises(ValueError, match="versioned closure"):
        validate(bad)


def test_runtime_preserves_physics_and_control_but_disables_closure_stop(tmp_path):
    cfg, settings, _ = configuration(tmp_path/"new", PLAN)
    old_cfg, old_settings, _ = configuration(tmp_path/"old", PREVIOUS)
    assert cfg["closure_gap_record_only"] and settings["closure_gap_record_only"]
    assert not old_cfg["closure_gap_record_only"] and not old_settings["closure_gap_record_only"]
    assert cfg["closure_gap_termination_m"] == settings["closure_gap_m"] == .005
    for key in ["v5_control", "position_iterations", "velocity_iterations", "physics_dt", "policy_dt",
                "asset_manifest_sha256", "leg_motor_friction"]:
        if key in cfg:
            assert cfg[key] == old_cfg[key]
    assert PLAN["mechanical_inner_knee_range_deg"] == [40., 110.]


def standing_trace():
    n = 500
    cmd = np.zeros((n, 1, 3));cmd[:, :, 2] = .32
    gravity = np.zeros_like(cmd);gravity[:, :, 2] = -1.
    a = dict(time=(np.arange(1, n+1)*.01)[:, None], command=cmd,
        velocity=np.zeros_like(cmd), omega=np.zeros_like(cmd), position=np.zeros_like(cmd),
        gravity=gravity, height=cmd[:, :, 2].copy(), gap=np.full((n, 1), .012),
        terminated=np.zeros((n, 1), bool), valid=np.ones((n, 1), bool), air_run=np.zeros((n, 1, 2)))
    case = dict(case_id="stand", role="stand", curriculum_skill="standing", curriculum_level=0,
                heading_rad=0., target=[0., 0., .32])
    return a, [case], np.zeros((1, 2))


def test_global_cold_gate_records_large_gap_without_admitting_bad_behavior():
    a, cases, xy = standing_trace()
    limits = PLAN["global_curriculum"]["limits"]
    legacy = assess_trace(a, cases, xy, limits, .01, 500)[0]
    assert not legacy["passed"] and not legacy["checks"]["closure"]
    monitor = assess_trace(a, cases, xy, limits, .01, 500, closure_record_only=True)[0]
    assert monitor["passed"] and "closure" not in monitor["checks"]
    assert monitor["closure_peak_m"] == .012 and monitor["closure_reference_exceeded"]
    a["velocity"][:, :, 0] = .2
    assert not assess_trace(a, cases, xy, limits, .01, 500, closure_record_only=True)[0]["passed"]
    a["velocity"][:] = 0.
    a["terminated"][10] = True
    assert not assess_trace(a, cases, xy, limits, .01, 500, closure_record_only=True)[0]["passed"]


@pytest.mark.parametrize("record_only,terminated,expected", [(False, False, False),
    (True, False, True), (True, True, False)])
def test_recovery_cold_gate_uses_actual_completion_and_retains_other_failure(record_only, terminated, expected):
    env = SimpleNamespace(num_envs=1, device="cpu", dt=.005, policy_dt=.01,
        evaluation_cases=None, fixed_cases=None, course_roles=["prone"],
        getup_cfg=dict(closure_gap_m=.005, closure_gap_record_only=record_only),
        generator=torch.Generator(), domain=SimpleNamespace(rng=torch.Generator()),
        recovery_sampler=SimpleNamespace(rng=random.Random(107), class_cursor={}))
    env.get_observations = lambda: torch.zeros(1, 1)
    env.reset = lambda ids: None
    def step(action):
        d = dict(stable=torch.ones(1, dtype=torch.bool), recovered=torch.ones(1, dtype=torch.bool),
            terminated=torch.tensor([terminated]), gap=torch.tensor([.012]),
            recovery_completed_event=torch.ones(1, dtype=torch.bool), recovery_active_seconds=torch.tensor([.01]))
        return env.get_observations(), torch.zeros(1), torch.tensor([terminated]), dict(diagnostics=d)
    env.step = step
    case = dict(family="belly_down", leg_class="rear_rear", pose_index=3, heading_rad=0., role="prone")
    result = evaluate(env, torch.nn.Identity(), dict(max_active_recovery_seconds=.2,
        post_completion_hold_seconds=.03), cases_override=[case])
    row = result["cases"][0]
    assert row["passed"] == expected and row["closure_reference_exceeded"]
    assert row["closure_peak_m"] == pytest.approx(.012)
    assert env.evaluation_cases is None and env.course_roles == ["prone"]

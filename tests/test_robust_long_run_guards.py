"""Long-run admission/retention must reject absent and nonfinite evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/unified_robust_20261021"
SPEC = importlib.util.spec_from_file_location("long_run_guards070", REPORTS / "long_training070_runner.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def admission():
    return dict(status="simulation_admitted_for_long_training", full_training_admitted=True,
        checkpoint_sha256="candidate", nominal=dict(adequate=True, failures=[]),
        long_standing_failures=[], long_standing={name: dict(episodes=16, survived=16, full_window_within_5cm=16,
            full_window_max_excursion_m=.04, wheel_clearance_after_02s_peak_m=.001)
            for name in ("stand", "stand_low", "stand_high")})


@pytest.mark.parametrize("bad", ["wrong_sha", "not_admitted", "missing_long", "nominal_failure", "long_failure", "incomplete",
    "blank_row", "nan_row", "drift", "termination"])
def test_long_run_rejects_partial_or_other_checkpoint_admission(bad):
    data = admission()
    if bad == "wrong_sha": data["checkpoint_sha256"] = "other"
    elif bad == "not_admitted": data["full_training_admitted"] = False
    elif bad == "missing_long": del data["long_standing"]
    elif bad == "nominal_failure": data["nominal"]["failures"] = ["standing"]
    elif bad == "long_failure": data["long_standing_failures"] = ["drift"]
    elif bad == "blank_row": data["long_standing"]["stand"] = {}
    elif bad == "nan_row": data["long_standing"]["stand"]["full_window_max_excursion_m"] = float("nan")
    elif bad == "drift": data["long_standing"]["stand"]["full_window_max_excursion_m"] = .051
    elif bad == "termination": data["long_standing"]["stand"]["survived"] = 15
    else: data["status"] = "running"
    with pytest.raises(ValueError, match="matching"):
        MODULE.require_admission(data, "candidate")


def test_complete_matching_admission_is_accepted_and_target_is_new_updates():
    MODULE.require_admission(admission(), "candidate")
    plan = json.loads((REPORTS / "long_training070_plan.json").read_text())
    assert sum(plan["block_sizes"]) == plan["new_long_updates_target"] == 5120
    assert plan["block_sizes"][0] == 64 and max(plan["block_sizes"]) <= 256
    assert plan["expected_final_lineage_clock"] == plan["parent_lineage_clock"] + 5120
    assert plan["expected_charged_if_all_completed"] == plan["robust_charged_before"] + 5120


@pytest.mark.parametrize("bad", ["none", "missing_arc", "nan_arc", "missing_fast", "nan_fast", "slow_fast",
    "zero_recovery", "nan_recovery", "no_motion", "nan_motion", "no_moving_recovery", "nan_gap", "passive_effort"])
def test_frozen_full_suite_gates_cover_real_cases_and_fail_closed(bad):
    analysis = dict(actual_settled_attitudes={family: dict(episodes=8, practical_recovery_and_motion_success=3)
        for family in ("left", "right", "inverted")}, motion_episodes=92, motion_survived=86,
        practical_moving_fall_recoveries=1, max_gap_m=.0008, passive_motor_peak_nm=0.,
        cases={name: dict(vx_mae_m_s=.1, yaw_mae_rad_s=.1)
               for name in ("arc_forward_left", "arc_forward_right", "arc_reverse_left", "arc_reverse_right")})
    analysis["cases"].update(forward_fast=dict(vx_mean_m_s=3.18), backward_fast=dict(vx_mean_m_s=-2.48))
    reference = deepcopy(analysis)
    protocol = json.loads((REPORTS / "acceptance_protocol003.json").read_text())
    if bad == "missing_arc": del analysis["cases"]["arc_forward_left"]
    elif bad == "nan_arc": analysis["cases"]["arc_forward_left"]["vx_mae_m_s"] = float("nan")
    elif bad == "missing_fast": del analysis["cases"]["forward_fast"]
    elif bad == "nan_fast": analysis["cases"]["forward_fast"]["vx_mean_m_s"] = float("nan")
    elif bad == "slow_fast": analysis["cases"]["forward_fast"]["vx_mean_m_s"] -= .36
    elif bad == "zero_recovery": analysis["actual_settled_attitudes"]["left"]["practical_recovery_and_motion_success"] = 0
    elif bad == "nan_recovery": analysis["actual_settled_attitudes"]["left"]["practical_recovery_and_motion_success"] = float("nan")
    elif bad == "no_motion": analysis["motion_episodes"] = 0
    elif bad == "nan_motion": analysis["motion_survived"] = float("nan")
    elif bad == "no_moving_recovery": analysis["practical_moving_fall_recoveries"] = 0
    elif bad == "nan_gap": analysis["max_gap_m"] = float("nan")
    elif bad == "passive_effort": analysis["passive_motor_peak_nm"] = .001
    failures = MODULE.full_failures(analysis, protocol, reference)
    assert bool(failures) == (bad != "none")

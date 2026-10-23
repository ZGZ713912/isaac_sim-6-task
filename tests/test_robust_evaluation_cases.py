"""Diagnostic speed lanes must retain their real command and frozen reset."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_speed_seam_cases_keep_pose_and_do_not_mutate_the_full_suite_template():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_robust_getup_batch.py"
    spec = importlib.util.spec_from_file_location("robust_evaluation_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = dict(name="forward", recovery=False, pose_index=17, command=[1., 0., .305])
    cases = module.fixed_forward_cases(template, [1.5, 1.75, 2.], 16)
    assert len(cases) == 48
    assert all(case["pose_index"] == 17 and not case["recovery"] for case in cases)
    for speed in (1.5, 1.75, 2.):
        assert sum(case["command"] == [speed, 0., .305] for case in cases) == 16
    cases[0]["command"][0] = 0.
    assert cases[3]["command"] == [1.5, 0., .305]
    assert template == dict(name="forward", recovery=False, pose_index=17, command=[1., 0., .305])


def test_forward_yaw_seam_records_each_true_two_axis_command():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_robust_getup_batch.py"
    spec = importlib.util.spec_from_file_location("robust_evaluation_yaw_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = dict(name="forward", recovery=False, pose_index=17, command=[1., 0., .305])
    cases = module.fixed_forward_cases(template, [3.], 8, [0., .05, .125, .25, .5, -.125])
    assert len(cases) == 48
    assert len({case["name"] for case in cases}) == 6
    assert all(case["pose_index"] == 17 for case in cases)
    assert sum(case["command"] == [3., -.125, .305] for case in cases) == 8


def test_stationary_push_retains_native_pose_and_push_and_leaves_default_cases_frozen():
    root = Path(__file__).resolve().parents[1]
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    batch = load("stationary_push_batch", root / "scripts/evaluate_robust_getup_batch.py")
    training = load("stationary_push_training", root / "scripts/train_robust_getup.py")
    bank = json.loads((root / "reports/unified_robust_20261021/pose_bank002.json").read_text())
    full = training.fixed_cases(bank, 140, 211322)
    digest = hashlib.sha256(json.dumps(full, sort_keys=True).encode()).hexdigest()
    assert digest == "3fe0212986b85b2add73b96c6676a6427b73d0c9400ab223962915304b7412f5"
    selected = batch.fixed_stationary_push_cases(full)
    original = [case for case in full if case["name"].startswith("moving_push_")]
    assert len(selected) == 8
    for case, old in zip(selected, original, strict=True):
        assert case["name"] == old["name"].replace("moving_", "stationary_", 1)
        assert case["command"] == [0., 0., .305]
        assert {k: v for k, v in case.items() if k not in ("name", "command")} == {
            k: v for k, v in old.items() if k not in ("name", "command")}
    selected[0]["command"][0] = 5.
    assert selected[2]["command"] == [0., 0., .305]
    assert hashlib.sha256(json.dumps(full, sort_keys=True).encode()).hexdigest() == digest


def test_scaled_stationary_push_does_not_change_default_or_duration():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_robust_getup_batch.py"
    spec = importlib.util.spec_from_file_location("stationary_push_scaling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = dict(push_force_n=80., push_torque_nm=32., push_start_seconds=5.,
                    push_duration_seconds=.3, push_torque_direction=-1.)
    assert module.scaled_stationary_push(original, 1.) == original
    scaled = module.scaled_stationary_push(original, .5)
    assert scaled == dict(original, push_force_n=40., push_torque_nm=16.)
    assert original["push_force_n"] == 80.
    for scale in (0., -1., 2., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            module.scaled_stationary_push(original, scale)


@pytest.mark.parametrize("arguments", [["--stationary-push-scale", ".5"],
    ["--stationary-push-cases", "--stationary-push-scale", "nan"],
    ["--stationary-push-cases", "--stationary-push-scale", "0"],
    ["--stationary-push-cases", "--case-names", "stand"]])
def test_stationary_diagnostic_invalid_cli_rejected_before_checkpoint_or_native_launch(tmp_path, arguments):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist"
    command = [sys.executable, str(root / "scripts/evaluate_robust_getup_batch.py"),
               "--checkpoint", str(tmp_path / "missing.pt"), "--output", str(output), *arguments]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2 and "error:" in result.stderr
    assert not output.exists()

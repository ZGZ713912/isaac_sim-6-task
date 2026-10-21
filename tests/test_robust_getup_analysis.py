"""Signal-unit checks and refusal of stale behavior evidence."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("getup_analysis", ROOT / "scripts/analyze_robust_getup.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_high_frequency_measure_keeps_physical_rms_and_rejects_slow_drift():
    time = np.arange(1800) * .005
    fast = 2. * np.sin(2 * np.pi * 40 * time)
    slow = 10. * np.sin(2 * np.pi * 4 * time)
    assert abs(MODULE.hf_rms(fast[:, None], .005) - np.sqrt(2)) < .002
    assert MODULE.hf_rms(slow[:, None], .005) < .0001
    assert MODULE.hf_rms(fast[:100, None], .005) is None
    assert MODULE.hf_rms(fast[:, None], .025) is None


def test_analysis_refuses_a_changed_trajectory(tmp_path):
    import hashlib
    import json
    import pytest
    path = tmp_path / "trajectory.npz"
    path.write_bytes(b"frozen")
    (tmp_path / "report.json").write_text(json.dumps({"trajectory_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="trajectory changed"):
        MODULE.analyze(tmp_path)


def endpoint_trace(directory, *, flicker_every=100, translation=.5, survived=True):
    import hashlib
    import json
    count = 2400
    stable = np.ones((count, 1), bool)
    stable[::flicker_every] = False
    command = np.zeros((count, 1, 3), np.float32)
    command[:, :, 0], command[:, :, 1] = .5, .8
    velocity = np.zeros_like(command)
    velocity[:, :, 0] = translation
    omega = np.zeros_like(command)
    omega[:, :, 2] = .8
    np.savez(directory / "trajectory.npz", time=np.arange(1, count+1, dtype=np.float32)[:, None]*.005,
        alive=np.ones((count, 1), bool), commands=command, velocity=velocity, omega=omega,
        passive=np.zeros((count, 1), bool), stable=stable)
    row = dict(name="right_symmetric", lane=0, recovery=True, recovered=True, survived=survived,
        followup_success=False, post_settle_family="right", max_gap_m=.0002)
    report = dict(lanes=[row], seconds=12., trajectory_sha256=hashlib.sha256((directory/"trajectory.npz").read_bytes()).hexdigest(),
        checkpoint_sha256="test", seed=21, actor_variant="robust", motion_episodes=0, motion_survived=0,
        passive_motor_peak_nm=0., max_gap_m=.0002)
    (directory/"report.json").write_text(json.dumps(report))


def test_practical_endpoint_tolerates_isolated_flicker_and_preserves_strict_result(tmp_path):
    endpoint_trace(tmp_path)
    result = MODULE.analyze(tmp_path)
    row = result["actual_settled_attitudes"]["right"]
    assert row["recovery_and_motion_success"] == 0
    assert row["practical_recovery_and_motion_success"] == 1


def test_practical_endpoint_rejects_missing_translation_instability_and_early_failure(tmp_path):
    for kwargs in ({"translation":0.}, {"flicker_every":10}, {"survived":False}):
        endpoint_trace(tmp_path, **kwargs)
        assert MODULE.analyze(tmp_path)["actual_settled_attitudes"]["right"]["practical_recovery_and_motion_success"] == 0

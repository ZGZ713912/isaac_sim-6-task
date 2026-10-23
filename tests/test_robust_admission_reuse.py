"""Explicit native reuse requires complete matching commands and artifacts."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("admission_reuse067", ROOT / "reports/unified_robust_20261021/admission067_runner.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("bad", [None, "command", "running", "returncode", "native_status", "checkpoint", "trajectory", "analysis", "variant"])
def test_explicit_reuse_rejects_partial_or_changed_native_evidence(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(MODULE, "REPORTS", tmp_path)
    output = tmp_path / "native"
    directory = output / "robust"
    directory.mkdir(parents=True)
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"synthetic-unit-fixture")
    trajectory = directory / "trajectory.npz"
    trajectory.write_bytes(b"synthetic-unit-fixture-not-native")
    command = ["python", "evaluation", "--seed", "211131"]
    receipt = dict(status="completed", returncode=0, native_status="completed", command=command)
    progress = dict(status="completed")
    report = dict(checkpoint_sha256=MODULE.sha(checkpoint), trajectory_sha256=MODULE.sha(trajectory), actor_variant="robust")
    if bad == "command": receipt["command"] = command + ["other"]
    elif bad == "running": progress["status"] = "starting"
    elif bad == "returncode": receipt["returncode"] = 1
    elif bad == "native_status": receipt["native_status"] = "failed"
    elif bad == "checkpoint": report["checkpoint_sha256"] = "other"
    elif bad == "trajectory": report["trajectory_sha256"] = "other"
    elif bad == "variant": report["actor_variant"] = "parent"
    for path, data in ((tmp_path / "native_process.json", receipt), (output / "progress.json", progress),
                       (directory / "report.json", report)):
        path.write_text(json.dumps(data))
    (directory / "analysis.json").write_text(json.dumps(dict(report_sha256="other" if bad == "analysis"
                                                           else MODULE.sha(directory / "report.json"))))
    if bad:
        with pytest.raises(ValueError):
            MODULE.reuse_completed("native", command, output, checkpoint, ("robust",))
    else:
        MODULE.reuse_completed("native", command, output, checkpoint, ("robust",))

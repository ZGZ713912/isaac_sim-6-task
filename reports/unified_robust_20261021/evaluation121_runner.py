"""Refresh exact115 native behavior without inferring motion from its parent."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from iteration045_runner import native, write


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    checkpoint = ROOT / "runs/unified_robust_20261022/composed_getup115/model_migrated.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == "11c4451ab73c4935051fc0187a5fc3a70a3534cee67f22d393500f623bd9f121"
    name = "motion_getup115_refresh121"
    output = ROOT / "runs/unified_robust_20261022" / name
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py",
        "--checkpoint", str(checkpoint), "--output", str(output),
        "--variants", "robust", "--replicas", "4", "--seed", "211322",
        "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12",
        "--isolated-origins"]
    result = dict(schema="exact115-behavior-refresh-v1", status="running",
        checkpoint_sha256=digest, command=command, started_unix=time.time(),
        purpose="Current standing and motion evidence, including actual termination reasons",
        qualification=None, full_training_admitted=False, optimizer_updates=0,
        existing_admission115_unchanged=True, physics_parameters_changed=False,
        impulse_termination_changed=False, hardware_started=False)
    path = REPORTS / "evaluation121_pipeline.json"
    if path.exists():
        raise ValueError("Refuse to overwrite an existing refresh")
    write(path, result)
    try:
        native(name, command, output)
        with (REPORTS / (name + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py",
                str(output / "robust"), "--tensorboard"], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((output / "robust/report.json").read_text())
        result.update(status="completed", report=str(output / "robust/report.json"),
            analysis=str(output / "robust/analysis.json"),
            trajectory_sha256=report["trajectory_sha256"])
    except BaseException:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        result["finished_unix"] = time.time()
        write(path, result)


if __name__ == "__main__":
    main()

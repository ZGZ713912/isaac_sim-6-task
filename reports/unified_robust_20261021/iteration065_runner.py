"""Serial continuous stationary-push pair, optionally followed by fresh admission."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from iteration045_runner import native, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continue-admission", action="store_true")
    args = parser.parse_args()
    gate = json.loads((REPORTS / "iteration066_final.json").read_text())
    assert not gate["static_failures"] and gate["full_suite_complete"]
    # CPU-check the frozen default suite and the explicit diagnostic selector.
    spec = importlib.util.spec_from_file_location("reference065_training", ROOT / "scripts/train_robust_getup.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    bank = json.loads((REPORTS / "pose_bank002.json").read_text())
    cases = training.fixed_cases(bank, 140, 211322)
    digest = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    assert digest == "3fe0212986b85b2add73b96c6676a6427b73d0c9400ab223962915304b7412f5"
    spec = importlib.util.spec_from_file_location("reference065_batch", ROOT / "scripts/evaluate_robust_getup_batch.py")
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    selected = batch.fixed_stationary_push_cases(training.fixed_cases(bank, 560, 211322))
    assert len(selected) == 32 and all(case["command"] == [0., 0., .305] for case in selected)
    pipeline = dict(schema="reference065-pipeline-v1", status="running", phase="migration",
                    started=time.time(), new_macro_updates=0, long_training_admitted=False,
                    frozen_default_suite_sha256=digest)
    write(REPORTS / "iteration065_pipeline.json", pipeline)
    try:
        parent = RUNS / "quiet_high_recenter066/model_migrated.pt"
        subprocess.run([sys.executable, str(REPORTS / "iteration065_migrate.py"), "--parent", str(parent),
                        "--static-result", str(REPORTS / "iteration066_final.json")], cwd=ROOT, check=True)
        candidate = RUNS / "quiet_severe_reference065/model_migrated.pt"
        for label, checkpoint in (("before", parent), ("after", candidate)):
            pipeline["phase"] = "native_" + label
            write(REPORTS / "iteration065_pipeline.json", pipeline)
            name = f"stationary_reference_{label}065"
            output = RUNS / name
            command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(checkpoint),
                "--output", str(output), "--variants", "robust", "--replicas", "16", "--seed", "211322",
                "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins",
                "--stationary-push-cases"]
            native(name, command, output)
            task = json.loads((output / "task_contract.json").read_text())["settings"]
            expected = dict(push_start_seconds=5., push_duration_seconds=.3, push_force_n=80.,
                            push_torque_nm=32., push_torque_direction=-1.)
            assert all(task[key] == value for key, value in expected.items())
        subprocess.run([sys.executable, str(REPORTS / "iteration065_analyze.py")], cwd=ROOT, check=True)
        result = json.loads((REPORTS / "iteration065_final.json").read_text())
        if args.continue_admission and result["native_lifecycle_exercised_and_valid"] and result["mechanical_gate_passed"]:
            pipeline["phase"] = "fresh_admission067"
            write(REPORTS / "iteration065_pipeline.json", pipeline)
            subprocess.run([sys.executable, str(REPORTS / "admission067_runner.py")], cwd=ROOT, check=True)
        pipeline.update(status="completed", phase="lifecycle_exercised" if
            result["native_lifecycle_exercised_and_valid"] else "needs_iteration", finished=time.time())
        write(REPORTS / "iteration065_pipeline.json", pipeline)
    except BaseException:
        pipeline.update(status="failed", error=traceback.format_exc(), finished=time.time())
        write(REPORTS / "iteration065_pipeline.json", pipeline)
        raise


if __name__ == "__main__":
    main()

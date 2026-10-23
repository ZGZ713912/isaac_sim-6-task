"""Controlled-tip reference proof; retain the failed full-strength disturbance."""
import argparse
import hashlib
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


def make_plan():
    strong = json.loads((REPORTS / "iteration065_final.json").read_text())
    assert not strong["native_lifecycle_exercised_and_valid"] and not strong["lifecycle_violations"]
    left = [r for r in strong["after"]["lanes"] if r["case"] == "stationary_push_left"]
    assert len(left) == 16 and all(not r["survived"] and r["termination_causes"] == ["impact_bad"] for r in left)
    return dict(schema="reference069-controlled-tip-plan-v1", status="frozen_before_native",
        reason="Full-strength left push terminated at impact protection before reference recapture; no lifecycle violation was found",
        retained_strong_push_receipt_sha256=hashlib.sha256((REPORTS / "iteration065_final.json").read_bytes()).hexdigest(),
        change="Diagnostic force/torque scale1->.5 only; no protection or default-suite change",
        native=dict(seed=211322, replicas_per_direction=16, seconds=12, domain_scale=.35,
                    ground_friction=[.3, .5], push_force_n=40., push_torque_nm=16.,
                    push_duration_seconds=.3, push_start_seconds=5., push_torque_direction=-1.,
                    policy_hz=200, physics_hz=1000, origin="isolated_overlap_v1"),
        before_checkpoint="quiet_high_recenter066/model_migrated.pt",
        after_checkpoint="quiet_severe_reference065/model_migrated.pt",
        new_macro_updates=0, robust_consumed=1160, long_training_admitted=False,
        interpretation="Functional reset/recapture proof at the declared disturbance; does not repair or qualify the failed80N/32Nm test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continue-admission", action="store_true")
    args = parser.parse_args()
    plan = make_plan()
    existing = json.loads((REPORTS / "iteration069_plan.json").read_text())
    assert existing == plan
    pipeline = dict(schema="reference069-pipeline-v1", status="running", phase="before",
                    started=time.time(), new_macro_updates=0, long_training_admitted=False)
    write(REPORTS / "iteration069_pipeline.json", pipeline)
    try:
        for side, model in (("before", plan["before_checkpoint"]), ("after", plan["after_checkpoint"])):
            pipeline["phase"] = side
            write(REPORTS / "iteration069_pipeline.json", pipeline)
            name = f"stationary_reference_{side}069"
            output = RUNS / name
            command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(RUNS / model),
                "--output", str(output), "--variants", "robust", "--replicas", "16", "--seed", "211322",
                "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins",
                "--stationary-push-cases", "--stationary-push-scale", ".5"]
            native(name, command, output)
            task = json.loads((output / "task_contract.json").read_text())["settings"]
            expected = dict(push_force_n=40., push_torque_nm=16., push_duration_seconds=.3,
                            push_start_seconds=5., push_torque_direction=-1.)
            assert all(task[key] == value for key, value in expected.items())
        subprocess.run([sys.executable, str(REPORTS / "iteration065_analyze.py"), "--iteration", "69"],
                       cwd=ROOT, check=True)
        result_path = REPORTS / "iteration069_final.json"
        result = json.loads(result_path.read_text())
        if args.continue_admission and result["native_lifecycle_exercised_and_valid"] and result["mechanical_gate_passed"]:
            pipeline["phase"] = "fresh_admission067"
            write(REPORTS / "iteration069_pipeline.json", pipeline)
            subprocess.run([sys.executable, str(REPORTS / "admission067_runner.py"),
                            "--lifecycle-result", str(result_path)], cwd=ROOT, check=True)
        pipeline.update(status="completed", phase="controlled_tip_exercised" if
            result["native_lifecycle_exercised_and_valid"] else "needs_iteration", finished=time.time())
        write(REPORTS / "iteration069_pipeline.json", pipeline)
    except BaseException:
        pipeline.update(status="failed", error=traceback.format_exc(), finished=time.time())
        write(REPORTS / "iteration069_pipeline.json", pipeline)
        raise


if __name__ == "__main__":
    main()

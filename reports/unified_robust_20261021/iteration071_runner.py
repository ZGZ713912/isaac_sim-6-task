"""Paired physical strength grid after strong-impact and weak-no-fall results."""
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
from evaluate_robust_getup_batch import fixed_stationary_push_cases
from train_robust_getup import fixed_cases

SCALES = [.65, .75, .85, .95]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan():
    weak = json.loads((REPORTS / "iteration069_final.json").read_text())
    assert not weak["native_lifecycle_exercised_and_valid"] and not weak["lifecycle_violations"]
    assert all(row["actual_falls"] == 0 for row in weak["after"]["summary"].values())
    bank = json.loads((REPORTS / "pose_bank002.json").read_text())
    cases = fixed_stationary_push_cases(fixed_cases(bank, 35 * 8, 211322), SCALES)
    assert len(cases) == 64
    return dict(schema="reference071-strength-grid-plan-v1", status="frozen_before_native",
        reason="80N/32Nm: left impact termination before recapture;40N/16Nm:no severe falls in either direction",
        retained_receipts_sha256={str(i): sha(REPORTS / f"iteration{i:03d}_final.json") for i in (65, 69)},
        native=dict(seed=211322, replicas_per_strength_direction=8, environments=64, seconds=12,
                    domain_scale=.35, ground_friction=[.3, .5], scales=SCALES,
                    force_n=[80. * scale for scale in SCALES], torque_nm=[32. * scale for scale in SCALES],
                    push_duration_seconds=.3, push_start_seconds=5., push_torque_direction=-1.,
                    policy_hz=200, physics_hz=1000, origin="isolated_overlap_v1"),
        before_checkpoint="quiet_high_recenter066/model_migrated.pt",
        after_checkpoint="quiet_severe_reference065/model_migrated.pt",
        checkpoint_sha256={side: sha(RUNS / path) for side, path in (
            ("before", "quiet_high_recenter066/model_migrated.pt"),
            ("after", "quiet_severe_reference065/model_migrated.pt"))},
        cases_sha256=hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        mechanism="Existing per-case floating-point push_sign multiplies force and torque; explicit diagnostic_push_scale metadata",
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in (
            ROOT / "scripts/evaluate_robust_getup_batch.py", REPORTS / "iteration065_analyze.py",
            ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py")},
        native_environment_code_changed=False, protection_thresholds_changed=False,
        new_macro_updates=0, robust_consumed=1160, long_training_admitted=False,
        interpretation="Functional lifecycle proof over an explicit strength grid; default80N suite and its failures remain")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--continue-admission", action="store_true")
    args = parser.parse_args()
    plan = make_plan()
    plan_path = REPORTS / "iteration071_plan.json"
    if args.prepare:
        if plan_path.exists():
            assert json.loads(plan_path.read_text()) == plan
        else:
            write(plan_path, plan)
        print(json.dumps(plan), flush=True)
        return
    assert json.loads(plan_path.read_text()) == plan
    pipeline = dict(schema="reference071-pipeline-v1", status="running", phase="before",
                    started=time.time(), new_macro_updates=0, long_training_admitted=False)
    write(REPORTS / "iteration071_pipeline.json", pipeline)
    try:
        for side in ("before", "after"):
            pipeline["phase"] = side
            write(REPORTS / "iteration071_pipeline.json", pipeline)
            name = f"stationary_reference_{side}071"
            output = RUNS / name
            command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint",
                str(RUNS / plan[side + "_checkpoint"]), "--output", str(output),
                "--variants", "robust", "--replicas", "8", "--seed", "211322",
                "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12",
                "--isolated-origins", "--stationary-push-cases", "--stationary-push-scales", *map(str, SCALES)]
            native(name, command, output)
            task = json.loads((output / "task_contract.json").read_text())["settings"]
            assert all(task[key] == value for key, value in dict(push_force_n=80., push_torque_nm=32.,
                push_duration_seconds=.3, push_start_seconds=5., push_torque_direction=-1.).items())
            report = json.loads((output / "robust/report.json").read_text())
            bank = json.loads((REPORTS / "pose_bank002.json").read_text())
            cases = fixed_stationary_push_cases(fixed_cases(bank, 35 * 8, 211322), SCALES)
            assert report["checkpoint_sha256"] == plan["checkpoint_sha256"][side]
            assert all(all(row[key] == value for key, value in case.items())
                       for row, case in zip(report["lanes"], cases, strict=True))
        subprocess.run([sys.executable, str(REPORTS / "iteration065_analyze.py"), "--iteration", "71"],
                       cwd=ROOT, check=True)
        result_path = REPORTS / "iteration071_final.json"
        result = json.loads(result_path.read_text())
        if args.continue_admission and result["native_lifecycle_exercised_and_valid"] and result["mechanical_gate_passed"]:
            pipeline["phase"] = "fresh_admission067"
            write(REPORTS / "iteration071_pipeline.json", pipeline)
            subprocess.run([sys.executable, str(REPORTS / "admission067_runner.py"),
                            "--lifecycle-result", str(result_path)], cwd=ROOT, check=True)
        pipeline.update(status="completed", phase="lifecycle_exercised" if
            result["native_lifecycle_exercised_and_valid"] else "needs_iteration", finished=time.time())
        write(REPORTS / "iteration071_pipeline.json", pipeline)
    except BaseException:
        pipeline.update(status="failed", error=traceback.format_exc(), finished=time.time())
        write(REPORTS / "iteration071_pipeline.json", pipeline)
        raise


if __name__ == "__main__":
    main()

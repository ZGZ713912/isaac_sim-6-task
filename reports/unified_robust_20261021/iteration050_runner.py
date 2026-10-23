"""Bounded64-update learned rolling feedback, then static/conditional full suite."""
from pathlib import Path
import json
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from iteration045_runner import native, write
from iteration036_compare import read, paired, standing
from iteration048_runner import static_failures


def evaluation(name, replicas, static):
    output = RUNS / name
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(RUNS / "pilot027_rolling_feedback050/model_final.pt"),
        "--output", str(output), "--variants", "robust", "--replicas", str(replicas), "--seed", "211322",
        "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins"]
    if static:
        command += ["--case-names", "stand_low", "stand", "stand_high"]
    native(name, command, output)
    with (REPORTS / (name + "_analysis.log")).open("w") as log:
        subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    pipeline = dict(schema="rolling050-pipeline-v1", status="running", phase="training", started=time.time(),
                    long_training_admitted=False)
    write(REPORTS / "iteration050_pipeline.json", pipeline)
    try:
        plan_path = REPORTS / "iteration050_plan.json"
        plan = json.loads(plan_path.read_text())
        plan["run_started"] = True
        write(plan_path, plan)
        parent = RUNS / "quiet_rolling050/model_migrated.pt"
        train = RUNS / "pilot027_rolling_feedback050"
        command = [sys.executable, "scripts/train_robust_getup.py", "--output", str(train), "--resume", str(parent),
            "--mode", "train", "--updates", "64", "--num-envs", "128", "--rollout-steps", "128",
            "--seed", "211329", "--actor-lr", ".001", "--domain-scale", ".35", "--friction-min", ".3",
            "--isolated-origins", "--quiet-only", "--quiet-feedback-only", "--standing-support-training",
            "--early-standing-position", "--standing-drift-objective", "--eval-seconds", "2"]
        progress = native("iteration050_train", command, train)
        assert progress["new_optimizer_updates"] == 64 and progress["trainable_actor_parameters"] == 12
        subprocess.run([sys.executable, str(REPORTS / "iteration050_native_audit.py")], cwd=ROOT, check=True)
        pipeline["phase"] = "extended_static"
        write(REPORTS / "iteration050_pipeline.json", pipeline)
        evaluation("learned_rolling_static050", 16, True)
        before, after = read("quiet_velocity_baseline036"), read("learned_rolling_static050")
        rows = standing(after)
        failures = static_failures(rows, standing(before))
        result = dict(schema="rolling050-static-results-v1", pairing=paired(before, after),
            before=standing(before), after=rows, static_failures=failures,
            new_macro_updates=64, robust_consumed=1025, lineage_clock=505,
            full_suite_complete=False, long_training_admitted=False, hardware_ready=False)
        if not failures:
            pipeline["phase"] = "full_randomized_suite"
            write(REPORTS / "iteration050_pipeline.json", pipeline)
            evaluation("learned_rolling_full050", 4, False)
            full = read("learned_rolling_full050")
            result.update(full_suite_complete=True, full_suite_pairing=paired(read("side_confirmation032"), full),
                          full_analysis=full[2])
        result["status"] = "static_target_not_met" if failures else "requires_fresh_formal_admission"
        write(REPORTS / "iteration050_final.json", result)
        pipeline.update(status="completed", phase=result["status"], finished=time.time())
        write(REPORTS / "iteration050_pipeline.json", pipeline)
        print(json.dumps(result), flush=True)
    except BaseException:
        pipeline.update(status="failed", error=traceback.format_exc(), finished=time.time())
        write(REPORTS / "iteration050_pipeline.json", pipeline)
        raise


if __name__ == "__main__":
    main()

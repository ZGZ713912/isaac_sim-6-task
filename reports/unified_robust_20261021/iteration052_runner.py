"""Zero-PPO all-height prediction, retaining original randomized delays."""
from pathlib import Path
import hashlib
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
    checkpoint = RUNS / "quiet_all_prediction052/model_migrated.pt"
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(checkpoint),
        "--output", str(output), "--variants", "robust", "--replicas", str(replicas), "--seed", "211322",
        "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins"]
    if static:
        command += ["--case-names", "stand_low", "stand", "stand_high"]
    native(name, command, output)
    with (REPORTS / (name + "_analysis.log")).open("w") as log:
        subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    assert json.loads((REPORTS / "iteration051_pipeline.json").read_text())["status"] == "completed"
    audit = json.loads((REPORTS / "iteration052_audit.json").read_text())
    checkpoint = RUNS / "quiet_all_prediction052/model_migrated.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == audit["checkpoint_sha256"]
    pipeline = dict(schema="all-height-prediction052-pipeline-v1", status="running", phase="extended_static",
        started=time.time(), optimizer_updates=0, robust_consumed=1025, long_training_admitted=False)
    write(REPORTS / "iteration052_pipeline.json", pipeline)
    plan_path = REPORTS / "iteration052_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["run_started"] = True
    write(plan_path, plan)
    try:
        evaluation("all_prediction_static052", 16, True)
        before, after = read("learned_rolling_static050"), read("all_prediction_static052")
        retained = read("quiet_velocity_baseline036")
        rows = standing(after)
        failures = static_failures(rows, standing(retained))
        result = dict(schema="all-height-prediction052-results-v1", pairing_to050=paired(before, after),
            pairing_to_retained=paired(retained, after), before=standing(before), after=rows,
            original_randomized_delays_retained=True, evaluation_overrides=after[1]["evaluation_overrides"],
            optimizer_updates=0, lineage_clock=505, robust_consumed=1025, static_failures=failures,
            full_suite_complete=False, long_training_admitted=False, hardware_ready=False,
            checkpoint_sha256=audit["checkpoint_sha256"], trajectory_sha256=after[1]["trajectory_sha256"])
        assert not result["evaluation_overrides"]
        if not failures:
            pipeline["phase"] = "full_randomized_suite"
            write(REPORTS / "iteration052_pipeline.json", pipeline)
            evaluation("all_prediction_full052", 4, False)
            full = read("all_prediction_full052")
            result.update(full_suite_complete=True, full_suite_pairing=paired(read("side_confirmation032"), full),
                          full_analysis=full[2])
        result["status"] = "static_target_not_met" if failures else "requires_fresh_formal_admission"
        write(REPORTS / "iteration052_final.json", result)
        pipeline.update(status="completed", phase=result["status"], finished=time.time())
        write(REPORTS / "iteration052_pipeline.json", pipeline)
        print(json.dumps(result), flush=True)
    except BaseException:
        pipeline.update(status="failed", finished=time.time(), error=traceback.format_exc())
        write(REPORTS / "iteration052_pipeline.json", pipeline)
        raise


if __name__ == "__main__":
    main()

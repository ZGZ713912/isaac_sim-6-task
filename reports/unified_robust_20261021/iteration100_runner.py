"""Conditional quiet097-policy30s and fresh-seed standing confirmation."""
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

CHECKPOINT = RUNS / "quiet_high_bias097/model_migrated.pt"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan():
    migration = json.loads((REPORTS / "iteration097_migration.json").read_text())
    assert migration["checkpoint_sha256"] == sha(CHECKPOINT)
    return dict(schema="quiet100-static-plan-v1", checkpoint_sha256=sha(CHECKPOINT),
        prerequisite="097 exact48-lane12s static gate passed; quiet-only confirmation independent of rejected098",
        jobs=[dict(name="quiet_standing30s100", seconds=30., seed=211322),
              dict(name="quiet_standing_fresh100", seconds=12., seed=211425)],
        stop_on_static_failure=True, replicas=16, environments=48, cases=["stand_low", "stand", "stand_high"],
        domain_scale=.35, friction_min=.3, friction_max=.5, policy_hz=200, physics_hz=1000,
        native_origin_mode="isolated_overlap_v1", thresholds="036 six actualHF nonregression, allalive,5cm full-window,3mm post0.2s air",
        original12s_prefix="097 all48 obs/actions/q/dq/XY/alive exact for first2400frames of30s",
        new_macro_updates=0, new_offline_steps=0, cumulative_offline_steps=0,
        robust_consumed=1384, full_training_admitted=False,
        further_required="protocol003 matched nominal two-seed and full randomized/lifecycle confirmation",
        implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "scripts/analyze_robust_getup.py",
            ROOT / "src/wheeled_algo/quiet_output_actor.py", ROOT / "src/wheeled_algo/robust_getup.py",
            ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py", Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    value = plan()
    plan_path = REPORTS / "iteration100_plan.json"
    if args.prepare:
        if plan_path.exists():
            assert json.loads(plan_path.read_text()) == value
        else:
            write(plan_path, value)
        print(json.dumps(value), flush=True)
        return
    assert json.loads(plan_path.read_text()) == value
    assert json.loads((REPORTS / "iteration098_pipeline.json").read_text())["status"] == "completed"
    assert json.loads((REPORTS / "iteration097_final.json").read_text())["static_pass"]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *value["implementation_sha256"],
                    str(plan_path.relative_to(ROOT))], cwd=ROOT, check=True)
    pipeline = dict(status="running", phase="long_standing", started=time.time(), new_macro_updates=0)
    write(REPORTS / "iteration100_pipeline.json", pipeline)
    try:
        import numpy as np
        from iteration036_compare import read, standing, paired
        from iteration048_runner import static_failures
        results = []
        for job in value["jobs"]:
            pipeline["phase"] = job["name"]
            write(REPORTS / "iteration100_pipeline.json", pipeline)
            output = RUNS / job["name"]
            native(job["name"], [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(CHECKPOINT),
                "--output", str(output), "--variants", "robust", "--replicas", "16", "--seed", str(job["seed"]),
                "--domain-scale", ".35", "--friction-min", ".3", "--seconds", str(job["seconds"]),
                "--isolated-origins", "--case-names", *value["cases"]], output)
            with (REPORTS / (job["name"] + "_analysis.log")).open("w") as log:
                subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            evidence = read(job["name"])
            rows = standing(evidence)
            failures = static_failures(rows, standing(read("quiet_velocity_baseline036")))
            result = dict(**job, standing=rows, failures=failures,
                          trajectory_sha256=evidence[1]["trajectory_sha256"])
            if job["seconds"] == 30.:
                old = read("high_bias_static097")
                result["original12s_initial_pairing"] = paired(old, evidence)
                result["original12s_trajectory_exact"] = {
                    key: np.array_equal(old[3][key], evidence[3][key][:len(old[3][key])])
                    for key in ("observations", "actions", "motor_position", "motor_velocity", "local_position", "alive")}
                assert all(result["original12s_trajectory_exact"].values())
            results.append(result)
            write(REPORTS / "iteration100_partial.json", dict(status="running", results=results, full_training_admitted=False))
            if failures:
                break
        passed = len(results) == 2 and all(not row["failures"] for row in results)
        result = dict(schema="quiet100-static-result-v1", status="completed", results=results,
            checkpoint_sha256=sha(CHECKPOINT), static_confirmation_pass=passed,
            candidate_adopted=False, full_training_admitted=False,
            new_macro_updates=0, new_offline_steps=0, cumulative_offline_steps=0, robust_consumed=1384)
        write(REPORTS / "iteration100_final.json", result)
        print("100 RESULT", json.dumps(result), flush=True)
        pipeline.update(status="completed", phase="result_sealed", finished=time.time())
    except BaseException:
        pipeline.update(status="failed_no_retry", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration100_pipeline.json", pipeline)


if __name__ == "__main__":
    main()

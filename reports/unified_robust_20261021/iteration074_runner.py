"""Randomized static confirmation and conditional full suite for alpha.25."""
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

CHECKPOINT = RUNS / "quiet_whole_wheel074/model_migrated.pt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan():
    return dict(schema="quiet074-randomized-plan-v1", checkpoint_sha256=sha(CHECKPOINT),
        migration_sha256=sha(REPORTS / "iteration074_migration.json"), static_environments=48,
        conditional_full_environments=140, seed=211322, seconds=12, domain_scale=.35,
        ground_friction=[.3, .5], policy_hz=200, physics_hz=1000, origin="isolated_overlap_v1",
        static_gate="all alive, full5cm, post.2s air3mm, each actual3-95HF<=036",
        sources={str(path.relative_to(ROOT)): sha(path) for path in (
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "scripts/export_robust_getup.py",
            ROOT / "src/wheeled_algo/quiet_output_actor.py", ROOT / "src/wheeled_algo/robust_getup.py")},
        new_macro_updates=0, robust_consumed=1160, selected_lineage=505, full_training_admitted=False)


def evaluate(name, subset):
    output = RUNS / name
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(CHECKPOINT),
        "--output", str(output), "--variants", "robust", "--replicas", "16" if subset else "4",
        "--seed", "211322", "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins"]
    if subset:
        command += ["--case-names", "stand_low", "stand", "stand_high"]
    native(name, command, output)
    with (REPORTS / (name + "_analysis.log")).open("w") as log:
        subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan = make_plan()
    path = REPORTS / "iteration074_plan.json"
    if args.prepare:
        if path.exists():
            assert json.loads(path.read_text()) == plan
        else:
            write(path, plan)
        print(json.dumps(plan), flush=True)
        return
    assert json.loads(path.read_text()) == plan
    pipeline = dict(status="running", phase="static", started=time.time(), new_macro_updates=0)
    write(REPORTS / "iteration074_pipeline.json", pipeline)
    try:
        from iteration036_compare import read, paired, standing
        from iteration048_runner import static_failures
        evaluate("quiet_whole_static074", True)
        before, after = read("high_recenter_static066"), read("quiet_whole_static074")
        pairing = paired(before, after)
        rows = standing(after)
        failures = static_failures(rows, standing(read("quiet_velocity_baseline036")))
        result = dict(schema="quiet074-randomized-result-v1", checkpoint_sha256=sha(CHECKPOINT),
            pairing=pairing, before=standing(before), after=rows, static_failures=failures,
            full_suite_complete=False, full_training_admitted=False, new_macro_updates=0, robust_consumed=1160)
        write(REPORTS / "iteration074_final.json", result)
        print("074STATIC", json.dumps(dict(rows=rows, failures=failures)), flush=True)
        if not failures:
            pipeline["phase"] = "full_suite"
            write(REPORTS / "iteration074_pipeline.json", pipeline)
            evaluate("quiet_whole_full074", False)
            full = read("quiet_whole_full074")
            result.update(full_suite_complete=True, full_pairing=paired(read("high_recenter_full066"), full),
                recovery=full[2]["actual_settled_attitudes"], motion=dict(survived=full[2]["motion_survived"],
                episodes=full[2]["motion_episodes"], practical_moving_fall_recoveries=full[2]["practical_moving_fall_recoveries"]),
                closure_peak_m=full[2]["max_gap_m"], passive_motor_peak_nm=full[2]["passive_motor_peak_nm"])
            write(REPORTS / "iteration074_final.json", result)
        pipeline.update(status="completed", phase="needs_iteration" if failures else "full_suite_sealed", finished=time.time())
    except BaseException:
        pipeline.update(status="failed", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration074_pipeline.json", pipeline)


if __name__ == "__main__":
    main()

"""Serial32-update coupled-state-feedback pilot, matched static test, conditional full suite."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/unified_robust_20261021"
RUNS = ROOT / "runs/unified_robust_20261022"
PARENT = RUNS / "quiet_coupled_state045/model_migrated.pt"
TRAIN = RUNS / "pilot026_coupled_state045"


def write(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def native(name, command, output):
    if output.exists():
        raise ValueError("Refuse to overwrite native output: " + str(output))
    receipt = dict(schema="quiet045-native-process-v1", command=command,
                   status="starting", started=time.time(), returncode=None)
    with (REPORTS / (name + ".log")).open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            receipt.update(pid=child.pid, status="running")
            write(REPORTS / (name + "_process.json"), receipt)
            print(name, "started", child.pid, flush=True)
        finally:
            code = child.wait()
    path = output / "progress.json"
    progress = json.loads(path.read_text()) if path.exists() else {}
    receipt.update(returncode=code, native_status=progress.get("status"), finished=time.time(),
                   status="completed" if code == 0 and progress.get("status") == "completed" else "failed")
    write(REPORTS / (name + "_process.json"), receipt)
    if receipt["status"] != "completed":
        raise RuntimeError(str(receipt))
    return progress


def evaluation(name, replicas, subset):
    output = RUNS / name
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(TRAIN / "model_final.pt"),
        "--output", str(output), "--variants", "robust", "--replicas", str(replicas),
        "--seed", "211322", "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12",
        "--isolated-origins"]
    if subset:
        command += ["--case-names", "stand_low", "stand", "stand_high"]
    native(name, command, output)
    with (REPORTS / (name + "_analysis.log")).open("w") as log:
        subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name, "native evaluation and original-event analysis complete", flush=True)


def main():
    pipeline = dict(schema="quiet-coupled045-pipeline-v1", status="running", started=time.time(), phase="training",
                    long_training_admitted=False)
    write(REPORTS / "iteration045_pipeline.json", pipeline)
    try:
        plan_path = REPORTS / "iteration045_plan.json"
        plan = json.loads(plan_path.read_text())
        plan["run_started"] = True
        write(plan_path, plan)
        command = [sys.executable, "scripts/train_robust_getup.py", "--output", str(TRAIN), "--resume", str(PARENT),
            "--mode", "train", "--updates", "32", "--num-envs", "128", "--rollout-steps", "128",
            "--seed", "211329", "--actor-lr", ".001", "--domain-scale", ".35", "--friction-min", ".3",
            "--isolated-origins", "--quiet-only", "--quiet-feedback-only", "--standing-support-training",
            "--early-standing-position", "--standing-drift-objective", "--eval-seconds", "2"]
        progress = native("iteration045_train", command, TRAIN)
        if progress["new_optimizer_updates"] != 32 or progress["trainable_actor_parameters"] != 204:
            raise ValueError("Unexpected native consumption or trainable parameter scope")
        import torch
        sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
        from export_robust_getup import load_mean
        from iteration036_compare import read, paired, standing
        torch.set_num_threads(1)
        initial = torch.load(TRAIN / "model_initial.pt", map_location="cpu", weights_only=False)
        final = torch.load(TRAIN / "model_final.pt", map_location="cpu", weights_only=False)
        changed = [k for k, v in final["actor_state_dict"].items()
                   if not torch.equal(v, initial["actor_state_dict"][k])]
        if len(changed) != 6 or any(not k.startswith("mlp.mean.quiet_velocity.") for k in changed):
            raise ValueError("Velocity-only training changed frozen actor parameters: " + str(changed))
        load_mean(TRAIN / "model_final.pt")
        sealed = dict(schema="quiet-coupled045-training-v1", new_updates=32,
            base_clock=final["infos"]["completed_updates"], robust_consumed=961, changed_actor=changed,
            all_existing_actor_and_std_exact=True, strict_load_valid=True, full_behavior_verified=False,
            checkpoint_sha256=hashlib.sha256((TRAIN / "model_final.pt").read_bytes()).hexdigest())
        write(REPORTS / "iteration045_training.json", sealed)
        print("045 TRAINING SEALED", json.dumps(sealed), flush=True)
        pipeline["phase"] = "expanded_confirmation"
        write(REPORTS / "iteration045_pipeline.json", pipeline)
        evaluation("coupled_state_static045", 16, True)
        before, after = read("quiet_velocity_baseline036"), read("coupled_state_static045")
        match = paired(before, after)
        counterpart = read("axis_exploration_static039")
        equal_budget_pair = paired(counterpart, after)
        rows = standing(after)
        baseline_rows = standing(before)
        failures = [name for name, row in rows.items() if row["survived"] != row["episodes"]
                    or row["full_window_max_excursion_m"] > .05
                    or row["wheel_clearance_after_02s_peak_m"] > .003
                    or row["leg_motion_fast_rms_rad_3_95"] > baseline_rows[name]["leg_motion_fast_rms_rad_3_95"]
                    or row["wheel_motion_fast_rms_rad_s_3_95"] > baseline_rows[name]["wheel_motion_fast_rms_rad_s_3_95"]]
        summary = dict(schema="quiet045-expanded-comparison-v1", pairing=match,
            baseline_zero_function_equivalence="iteration045_migration.json:15600real observations exact; new coupled constructor consumes no extra RNG",
            before=standing(before), after=rows, static_failures=failures,
            equal_budget039_pairing=equal_budget_pair, equal_budget039=standing(counterpart),
            training=sealed, long_training_admitted=False, full_suite_complete=False)
        if failures:
            pipeline.update(status="static_target_not_met", phase="extended_complete", finished=time.time())
            summary["full_suite_omission_reason"] = "Inspect static learning before spending a full-suite native evaluation"
            write(REPORTS / "iteration045_final.json", summary)
            print("045 STATIC TARGET NOT MET", json.dumps(rows), flush=True)
            return
        pipeline["phase"] = "full_confirmation"
        write(REPORTS / "iteration045_pipeline.json", pipeline)
        evaluation("coupled_state_full045", 4, False)
        original, candidate = read("side_confirmation032"), read("coupled_state_full045")
        summary.update(full_suite_complete=True, full_pairing=paired(original, candidate),
            full_standing=standing(candidate), recovery=candidate[2]["actual_settled_attitudes"],
            motion=dict(survived=candidate[2]["motion_survived"], episodes=candidate[2]["motion_episodes"],
                practical_moving_fall_recoveries=candidate[2]["practical_moving_fall_recoveries"]))
        write(REPORTS / "iteration045_final.json", summary)
        pipeline.update(status="completed", finished=time.time(), phase="all_complete")
    except BaseException:
        pipeline.update(status="failed", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration045_pipeline.json", pipeline)


if __name__ == "__main__":
    main()

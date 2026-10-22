"""Bounded serial native baseline,64-update pilot, and two complete confirmations."""
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
PARENT = RUNS / "quiet_velocity036/model_migrated.pt"
TRAIN = RUNS / "pilot021_velocity_condition036"


def write(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def native(name, command, output):
    if output.exists():
        raise ValueError("Refuse to overwrite native output: " + str(output))
    receipt = dict(schema="robust036-native-process-v1", command=command,
                   status="starting", started=time.time(), returncode=None)
    with (REPORTS / (name + ".log")).open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            receipt.update(pid=child.pid, status="running")
            write(REPORTS / (name + "_process.json"), receipt)
            print(name, "started", child.pid, flush=True)
        finally:
            code = child.wait()
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    receipt.update(returncode=code, native_status=progress.get("status"), finished=time.time(),
                   status="completed" if code == 0 and progress.get("status") == "completed" else "failed")
    write(REPORTS / (name + "_process.json"), receipt)
    if receipt["status"] != "completed":
        raise RuntimeError(str(receipt))
    return progress


def evaluation(name, checkpoint, replicas, subset):
    output = RUNS / name
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(checkpoint),
        "--output", str(output), "--variants", "robust", "--replicas", str(replicas),
        "--seed", "211322", "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12",
        "--isolated-origins"]
    if subset:
        command += ["--case-names", "stand_low", "stand", "stand_high"]
    native(name, command, output)
    with (REPORTS / (name + "_analysis.log")).open("w") as log:
        subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name, "complete, native report and original-event analysis saved", flush=True)


def main():
    pipeline = dict(schema="quiet-velocity036-pipeline-v1", status="running", started=time.time(), phase="baseline")
    write(REPORTS / "iteration036_pipeline.json", pipeline)
    try:
        evaluation("quiet_velocity_baseline036", PARENT, 16, True)
        pipeline["phase"] = "training"
        write(REPORTS / "iteration036_pipeline.json", pipeline)
        plan_path = REPORTS / "iteration036_plan.json"
        plan = json.loads(plan_path.read_text())
        plan.update(run_started=True, training_output=str(TRAIN.relative_to(ROOT)))
        write(plan_path, plan)
        command = [sys.executable, "scripts/train_robust_getup.py", "--output", str(TRAIN), "--resume", str(PARENT),
            "--mode", "train", "--updates", "64", "--num-envs", "128", "--rollout-steps", "128",
            "--seed", "211329", "--actor-lr", "1e-5", "--domain-scale", ".35", "--friction-min", ".3",
            "--isolated-origins", "--quiet-only", "--standing-support-training", "--early-standing-position",
            "--standing-drift-objective", "--quiet-exploration-fraction", ".1", "--eval-seconds", "2"]
        progress = native("iteration036_train", command, TRAIN)
        if progress["new_optimizer_updates"] != 64:
            raise ValueError("Unexpected completed native update count")
        import torch
        sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
        from export_robust_getup import load_mean
        torch.set_num_threads(1)
        initial = torch.load(TRAIN / "model_initial.pt", map_location="cpu", weights_only=False)
        final = torch.load(TRAIN / "model_final.pt", map_location="cpu", weights_only=False)
        changed = [k for k, v in final["actor_state_dict"].items()
                   if not torch.equal(v, initial["actor_state_dict"][k])]
        prefixes = ("mlp.mean.quiet.", "mlp.mean.quiet_low.", "mlp.mean.quiet_high.")
        if not changed or any(not k.startswith(prefixes) for k in changed):
            raise ValueError("A frozen recovery/motion/std parameter changed: " + str(changed))
        load_mean(TRAIN / "model_final.pt")
        sealed = dict(schema="quiet-velocity036-training-v1", new_updates=64,
            base_clock=final["infos"]["completed_updates"], robust_consumed=801, changed_actor=changed,
            frozen_recovery_motion_std_exact=True, strict_load_valid=True, full_behavior_verified=False,
            checkpoint_sha256=hashlib.sha256((TRAIN / "model_final.pt").read_bytes()).hexdigest())
        write(REPORTS / "iteration036_training.json", sealed)
        print("036 TRAINING SEALED", json.dumps(sealed), flush=True)
        pipeline["phase"] = "expanded_confirmation"
        write(REPORTS / "iteration036_pipeline.json", pipeline)
        evaluation("quiet_velocity_confirmation036", TRAIN / "model_final.pt", 16, True)
        pipeline["phase"] = "full_confirmation"
        write(REPORTS / "iteration036_pipeline.json", pipeline)
        evaluation("velocity_full_confirmation036", TRAIN / "model_final.pt", 4, False)
        pipeline.update(status="completed", finished=time.time(), phase="all_complete")
    except BaseException:
        pipeline.update(status="failed", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration036_pipeline.json", pipeline)


if __name__ == "__main__":
    main()

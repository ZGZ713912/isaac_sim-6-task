#!/usr/bin/env python3
"""One bounded training pilot followed by a serial, fixed native review.

No repeated PPO or model adoption is triggered by a reward/score. TensorBoard
is an independent event consumer and is never started by this supervisor.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from train_robust_getup import digest, write_json
from run_gap_long_training import verify_restore


def run(plan_file):
    plan = json.loads(plan_file.read_text())
    output = Path(plan["supervisor_output"])
    if output.exists():
        raise ValueError("A bounded round is never started twice")
    output.mkdir(parents=True)
    command = plan["training_command"]
    updates = int(command[command.index("--updates") + 1])
    if not 1 <= updates <= 64:
        raise ValueError("This supervisor permits only a 1..64 update pilot")
    progress_path = output / "progress.json"
    state = dict(schema="bounded-gap-round-v1", status="starting", pid=os.getpid(),
                 plan=str(plan_file), new_optimizer_updates=0, new_optimizer_attempts=0,
                 actual_experiment_consumed=plan["prior_experiment_consumed"],
                 active_job=None, qualification=None, hardware_ready=False)
    child = None
    stop = False

    def save():
        state["updated_unix"] = time.time()
        write_json(progress_path, state)

    def interrupt(number, frame):
        nonlocal stop
        stop = True
        state["stop_requested_signal"] = number

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def verify_sources():
        for name, expected in plan["source_sha256"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Round source changed: " + name)

    def job(name, args, training=False):
        nonlocal child
        verify_sources()
        if stop:
            raise InterruptedError("Stop requested before " + name)
        with (output / (name + ".log")).open("x") as log:
            child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", OPENBLAS_NUM_THREADS="1"))
            state.update(status=name, active_job=dict(pid=child.pid, name=name, command=args))
            save()
            signalled = restored = False
            while True:
                rc = child.poll()
                if training:
                    path = Path(plan["training_output"]) / "progress.json"
                    if path.exists():
                        row = json.loads(path.read_text())
                        attempted = row.get("new_optimizer_attempts", row.get("new_optimizer_updates", 0))
                        state.update(new_optimizer_updates=row.get("new_optimizer_updates", 0),
                                     new_optimizer_attempts=attempted,
                                     actual_experiment_consumed=plan["prior_experiment_consumed"] + attempted,
                                     current_learning_clock=row.get("completed_updates"),
                                     optimizer_update_in_flight=row.get("optimizer_update_in_flight", False))
                        initial = path.parent / "model_initial.pt"
                        if initial.exists() and not restored:
                            verify_restore(Path(plan["source_checkpoint"]), initial, output / "initial_restore.json")
                            restored = True
                        save()
                if stop and rc is None and not signalled:
                    child.send_signal(signal.SIGTERM)
                    signalled = True
                if rc is not None:
                    break
                time.sleep(5.)
            child = None
            state["active_job"] = None
            save()
            if stop:
                raise InterruptedError("Bounded round stopped")
            if rc:
                raise RuntimeError(f"{name} exited with code {rc}")
            if name == "fixed_evaluation":
                evaluation = Path(plan["evaluation_output"])
                receipt = evaluation / "progress.json"
                completion = json.loads(receipt.read_text()) if receipt.exists() else {}
                if completion.get("status") != "completed" or (evaluation / "failure.json").exists():
                    raise RuntimeError("Native evaluation returned zero without a completed artifact")

    save()
    try:
        job("training", command, training=True)
        train = Path(plan["training_output"])
        report = json.loads((train / "progress.json").read_text())
        if report["status"] != "completed" or report["new_optimizer_updates"] != updates:
            raise ValueError("Training did not finish its bounded pilot")
        checkpoint = train / "model_final.pt"
        state.update(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
        python = command[0]
        evaluation = Path(plan["evaluation_output"])
        job("fixed_evaluation", [python, "scripts/evaluate_gap_robustness.py", "--checkpoint", str(checkpoint),
            "--output", str(evaluation), "--bank", plan["bank"], "--actuator-timing", "native",
            "--seconds", "8", "--seed", "241201"])
        job("standing_analysis", [python, "scripts/analyze_gap_standing.py", "--run", str(evaluation),
            "--output", str(output / "standing.json")])
        job("paired_comparison", [python, "scripts/compare_gap_robustness.py", "--before", plan["baseline_evaluation"],
            "--after", str(evaluation), "--output", str(output / "paired.json")])
        standing = json.loads((output / "standing.json").read_text())
        paired = json.loads((output / "paired.json").read_text())
        state.update(status="review_ready", standing_pass=standing["pose_and_position_pass"],
                     motion_vx_mae_m_s=paired["groups"]["motion"]["after"]["vx_mae_m_s"]["mean"],
                     review_requires_behavior_and_mechanical_assessment=True,
                     further_PPO_started=False)
        save()
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait()
        # A learner can finish one in-flight optimizer update while sealing on
        # SIGTERM. Refresh its final counter after wait; never refund that work.
        final_progress = Path(plan["training_output"]) / "progress.json"
        if final_progress.exists():
            row = json.loads(final_progress.read_text())
            attempted = row.get("new_optimizer_attempts", row.get("new_optimizer_updates", 0))
            state.update(new_optimizer_updates=row.get("new_optimizer_updates", 0),
                         new_optimizer_attempts=attempted,
                         actual_experiment_consumed=plan["prior_experiment_consumed"] + attempted,
                         current_learning_clock=row.get("completed_updates"),
                         optimizer_update_in_flight=row.get("optimizer_update_in_flight", False),
                         stopped_checkpoint=row.get("stopped_checkpoint"))
        state.update(status="stopped" if isinstance(error, InterruptedError) else "failed",
                     active_job=None, error=traceback.format_exc())
        save()
        if not isinstance(error, InterruptedError):
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    run(parser.parse_args().plan.resolve())

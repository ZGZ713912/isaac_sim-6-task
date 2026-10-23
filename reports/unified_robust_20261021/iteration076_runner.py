"""Bounded32-update quiet adaptation after complete-output filtering failed074."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from iteration045_runner import native, write

PARENT = RUNS / "quiet_whole_wheel074/model_migrated.pt"
TRAIN = RUNS / "pilot032_filtered_quiet076"
PREFIXES = tuple("mlp.mean." + name for name in ("quiet.", "quiet_low.", "quiet_high.", "quiet_velocity."))
UPDATES = 32


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight():
    from train_robust_getup import prepare
    with tempfile.TemporaryDirectory(prefix="quiet076-task-") as folder:
        args = SimpleNamespace(output=Path(folder) / "preflight", resume=PARENT,
            bank=REPORTS / "pose_bank002.json", no_domains=False, curriculum_updates=128,
            quiet_leg_alpha=.25, quiet_wheel_alpha=.65, domain_scale=.35,
            quiet_only=True, standing_support_training=True, early_standing_position=True,
            standing_drift_objective=True, quiet_exploration_fraction=.06,
            quiet_wheel_exploration_fraction=.35, num_envs=128, rollout_steps=128,
            mode="train", friction="high", friction_min=.3, seed=211329, isolated_origins=True)
        task = prepare(args)[1]
    assert task["quiet_output_wheel_alpha"] == .25
    assert task["standing_position_reference"] == dict(capture_delay_seconds=.05, reset_on_severe_tilt=True)
    return task


def make_plan():
    import torch
    from export_robust_getup import load_mean
    torch.set_num_threads(1)
    mean, info = load_mean(PARENT)
    task = preflight()
    expected = sum(p.numel() for name, p in mean.named_parameters() if ("mlp.mean." + name).startswith(PREFIXES))
    assert expected == 8106 and info["completed_updates"] == 505
    failed = json.loads((REPORTS / "iteration074_final.json").read_text())
    assert failed["static_failures"] and failed["checkpoint_sha256"] == sha(PARENT)
    return dict(schema="quiet076-bounded-plan-v1", parent_sha256=sha(PARENT), expected_settings=task,
        new_macro_updates_target=UPDATES, num_envs=128, rollout_steps=128, actor_lr=1e-5, critic_lr=1e-3,
        expected_trainable_values=expected, trained="quiet MLPs and24 position weights only",
        frozen="locomotion parent, arc, recovery, std, geometry, filter and reference lifecycle",
        reason="074 survives48/48, reduces wheel vibration, but high start drift5.408cm and low legHF+.53% fail",
        objective="Existing support, full-window position deadband5cm, .02m/s drift and HF costs retained",
        lineage_before=505, lineage_after_if_complete=537, robust_charged_before=1160,
        robust_charged_if_complete=1192, evaluation="48 randomized static; conditional140 full suite",
        sources={str(path.relative_to(ROOT)): sha(path) for path in (
            ROOT / "scripts/train_robust_getup.py", ROOT / "scripts/export_robust_getup.py",
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "src/wheeled_algo/quiet_output_actor.py",
            ROOT / "src/wheeled_algo/robust_getup.py", ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py")},
        full_training_admitted=False, new_macro_updates=0)


def audit(final=False):
    import torch
    from export_robust_getup import load_mean
    from iteration044_selective_memory import equal
    from iteration045_native_state_audit import named_slots, finite
    torch.set_num_threads(1)
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    initial_path = TRAIN / "model_initial.pt"
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    before, _ = load_mean(PARENT)
    mean, info = load_mean(initial_path)
    expected = json.loads((REPORTS / "iteration076_plan.json").read_text())["expected_settings"]
    assert info["settings"] == expected and info["quiet_only_tuning"] and not info["quiet_velocity_only_tuning"]
    for key in ("actor_state_dict", "critic_state_dict"):
        assert equal(parent[key], initial[key]), key
    old_slots, slots = named_slots(parent, before), named_slots(initial, mean)
    assert equal(old_slots, slots)
    assert all(equal(parent["infos"][key], info[key]) for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"))
    rates = {g["name"]: g["lr"] for g in initial["optimizer_state_dict"]["param_groups"]}
    assert rates == dict(actor=1e-5, critic=1e-3)
    progress = json.loads((TRAIN / "progress.json").read_text())
    if "trainable_actor_parameters" in progress:
        assert progress["trainable_actor_parameters"] == 8106
    receipt = dict(schema="quiet076-native-state-v1", initial_sha256=sha(initial_path),
        parent_actor_critic_named_adam_rng_exact=True, task_exact=True, rates=rates,
        initial_clock=info["completed_updates"], updates_seen_at_initial_audit=progress["new_optimizer_updates"])
    assert info["completed_updates"] == 505
    if final:
        final_path = TRAIN / "model_final.pt"
        state = torch.load(final_path, map_location="cpu", weights_only=False)
        after, final_info = load_mean(final_path)
        assert progress["status"] == "completed" and progress["new_optimizer_updates"] == UPDATES
        assert progress["trainable_actor_parameters"] == 8106
        assert all(finite(state[key]) for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"))
        changed = [key for key, value in state["actor_state_dict"].items() if not equal(value, initial["actor_state_dict"][key])]
        assert changed and all(key.startswith(PREFIXES) for key in changed)
        final_slots = named_slots(state, after)
        assert all(equal(slots["actor"][key], final_slots["actor"][key])
                   for key in slots["actor"] if not key.startswith(PREFIXES))
        steps = {key: int(final_slots["actor"][key]["step"].item()) - int(value.get("step", torch.tensor(0.)).item())
                 for key, value in slots["actor"].items() if key.startswith(PREFIXES)}
        assert set(steps.values()) == {384}
        assert final_info["settings"] == expected and final_info["completed_updates"] == 537
        receipt.update(final_sha256=sha(final_path), changed_actor=changed, adam_step_deltas=steps,
            frozen_actor_std_and_named_adam_exact=True, new_macro_updates=UPDATES,
            final_clock=537, robust_consumed=1192, finite=True)
    return receipt


def train(expected):
    if TRAIN.exists():
        raise ValueError("Refuse to overwrite training output")
    command = [sys.executable, "scripts/train_robust_getup.py", "--output", str(TRAIN), "--resume", str(PARENT),
        "--mode", "train", "--updates", "32", "--num-envs", "128", "--rollout-steps", "128",
        "--seed", "211329", "--actor-lr", ".00001", "--domain-scale", ".35", "--friction-min", ".3",
        "--isolated-origins", "--quiet-only", "--standing-support-training", "--early-standing-position",
        "--standing-drift-objective", "--eval-seconds", "2", "--quiet-exploration-fraction", ".06",
        "--quiet-wheel-exploration-fraction", ".35"]
    receipt = dict(schema="quiet076-checked-native-process-v1", status="starting", command=command, started=time.time())
    with (REPORTS / "iteration076_train.log").open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        receipt.update(pid=child.pid, status="running")
        write(REPORTS / "iteration076_train_process.json", receipt)
        print("076 training started", child.pid, flush=True)
        try:
            while child.poll() is None:
                contract = TRAIN / "task_contract.json"
                if not receipt.get("task_exact") and contract.exists():
                    assert json.loads(contract.read_text())["settings"] == expected
                    receipt.update(task_exact=True, initial_exists_at_task_check=(TRAIN / "model_initial.pt").exists())
                    write(REPORTS / "iteration076_train_process.json", receipt)
                if not receipt.get("initial_audited") and zipfile.is_zipfile(TRAIN / "model_initial.pt"):
                    write(REPORTS / "iteration076_initial.json", audit())
                    receipt["initial_audited"] = True
                    write(REPORTS / "iteration076_train_process.json", receipt)
                time.sleep(.5)
            progress = json.loads((TRAIN / "progress.json").read_text())
            assert child.returncode == 0 and progress["status"] == "completed"
            assert receipt.get("task_exact") and receipt.get("initial_audited")
            write(REPORTS / "iteration076_state.json", audit(final=True))
            receipt.update(status="completed", returncode=child.returncode, known_completed=UPDATES, finished=time.time())
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            path = TRAIN / "progress.json"
            progress = json.loads(path.read_text()) if path.exists() else {}
            receipt.update(status="failed_no_retry", returncode=child.returncode,
                known_completed=progress.get("new_optimizer_updates", 0),
                conservative_inflight_charge=int(progress.get("status") == "training"),
                error=traceback.format_exc(), finished=time.time())
            raise
        finally:
            write(REPORTS / "iteration076_train_process.json", receipt)


def evaluate(name, static):
    output = RUNS / name
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(TRAIN / "model_final.pt"),
        "--output", str(output), "--variants", "robust", "--replicas", "16" if static else "4",
        "--seed", "211322", "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins"]
    if static:
        command += ["--case-names", "stand_low", "stand", "stand_high"]
    native(name, command, output)
    with (REPORTS / (name + "_analysis.log")).open("w") as log:
        subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan_path = REPORTS / "iteration076_plan.json"
    plan = make_plan()
    if args.prepare:
        if plan_path.exists():
            assert json.loads(plan_path.read_text()) == plan
        else:
            write(plan_path, plan)
        print(json.dumps({key: value for key, value in plan.items() if key != "expected_settings"}), flush=True)
        return
    assert json.loads(plan_path.read_text()) == plan
    pipeline = dict(status="running", phase="quiet_training", started=time.time(), full_training_admitted=False)
    write(REPORTS / "iteration076_pipeline.json", pipeline)
    try:
        train(plan["expected_settings"])
        pipeline["phase"] = "static_confirmation"
        write(REPORTS / "iteration076_pipeline.json", pipeline)
        evaluate("filtered_quiet_static076", True)
        from iteration036_compare import read, paired, standing
        from iteration048_runner import static_failures
        after, before = read("filtered_quiet_static076"), read("quiet_whole_static074")
        rows = standing(after)
        failures = static_failures(rows, standing(read("quiet_velocity_baseline036")))
        result = dict(schema="quiet076-bounded-result-v1", checkpoint_sha256=sha(TRAIN / "model_final.pt"),
            initial_pairing=paired(before, after), before=standing(before), after=rows, static_failures=failures,
            new_macro_updates=UPDATES, robust_consumed=1192, lineage_clock=537,
            full_suite_complete=False, full_training_admitted=False)
        write(REPORTS / "iteration076_final.json", result)
        print("076STATIC", json.dumps(dict(rows=rows, failures=failures)), flush=True)
        if not failures:
            pipeline["phase"] = "full_suite"
            write(REPORTS / "iteration076_pipeline.json", pipeline)
            evaluate("filtered_quiet_full076", False)
            full = read("filtered_quiet_full076")
            result.update(full_suite_complete=True, full_pairing=paired(read("high_recenter_full066"), full),
                recovery=full[2]["actual_settled_attitudes"], motion=dict(survived=full[2]["motion_survived"],
                episodes=full[2]["motion_episodes"], practical_moving_fall_recoveries=full[2]["practical_moving_fall_recoveries"]))
            write(REPORTS / "iteration076_final.json", result)
        pipeline.update(status="completed", phase="needs_iteration" if failures else "full_suite_sealed", finished=time.time())
    except BaseException:
        pipeline.update(status="failed", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration076_pipeline.json", pipeline)


if __name__ == "__main__":
    main()

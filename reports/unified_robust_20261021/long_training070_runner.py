"""Admission-gated serial recovery PPO; no hardware access or automatic retry."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from iteration045_runner import native, write
from long_training070_plan import make_plan, task_settings, training_arguments


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_admission(admission, checkpoint_sha256):
    if (admission.get("status") != "simulation_admitted_for_long_training"
            or admission.get("full_training_admitted") is not True
            or admission.get("checkpoint_sha256") != checkpoint_sha256
            or admission.get("nominal", {}).get("adequate") is not True
            or admission.get("nominal", {}).get("failures") != []
            or admission.get("long_standing_failures") != []
            or set(admission.get("long_standing", {})) != {"stand", "stand_low", "stand_high"}):
        raise ValueError("Long PPO requires complete matching nominal and long-standing admission")
    for row in admission["long_standing"].values():
        if (row.get("episodes") != 16 or row.get("survived") != 16 or row.get("full_window_within_5cm") != 16
                or not 0. <= row.get("full_window_max_excursion_m", math.inf) <= .05
                or not 0. <= row.get("wheel_clearance_after_02s_peak_m", math.inf) <= .003):
            raise ValueError("Long PPO requires complete matching nominal and long-standing admission")


def full_failures(analysis, protocol, reference):
    failures = []
    for family, rate in protocol["family_min_complete_rate"].items():
        row = analysis["actual_settled_attitudes"].get(family, {})
        count = row.get("practical_recovery_and_motion_success", 0)
        if (not row.get("episodes", 0) > 0 or not count >= max(
                protocol["minimum_successful_episodes_per_family"], rate * row.get("episodes", 0))):
            failures.append(family + " recovery")
    if not (analysis["motion_episodes"] > 0 and analysis["motion_survived"] >=
            protocol["ordinary_motion_survival_rate_min"] * analysis["motion_episodes"]):
        failures.append("ordinary motion survival")
    for name in ("arc_forward_left", "arc_forward_right", "arc_reverse_left", "arc_reverse_right"):
        row = analysis["cases"].get(name, {})
        values = [row.get(key) for key in ("vx_mae_m_s", "yaw_mae_rad_s")]
        if (any(value is None or not math.isfinite(value) for value in values)
                or values[0] > protocol["ordinary_arc_vx_mae_max_m_s"]
                or values[1] > protocol["ordinary_arc_yaw_mae_max_rad_s"]):
            failures.append(name + " tracking")
    for name in ("forward_fast", "backward_fast"):
        speed = analysis["cases"].get(name, {}).get("vx_mean_m_s")
        if speed is None or not math.isfinite(speed) or abs(speed - reference["cases"][name]["vx_mean_m_s"]) > .35:
            failures.append(name + " speed retention")
    if not analysis["practical_moving_fall_recoveries"] >= 1:
        failures.append("moving fall recovery")
    if not 0. <= analysis["max_gap_m"] <= protocol["closure_gap_max_m"]:
        failures.append("closure gap")
    if analysis["passive_motor_peak_nm"] != 0.:
        failures.append("nonzero passive effort")
    return failures


def audit(parent_path, output, expected, updates, final=False):
    import torch
    from export_robust_getup import load_mean
    from iteration044_selective_memory import equal
    from iteration045_native_state_audit import named_slots, finite, RNG_KEYS
    torch.set_num_threads(1)
    initial_path = output / "model_initial.pt"
    parent, initial = [torch.load(path, map_location="cpu", weights_only=False) for path in (parent_path, initial_path)]
    before, _ = load_mean(parent_path)
    mean, info = load_mean(initial_path)
    assert all(equal(parent[key], initial[key]) for key in ("actor_state_dict", "critic_state_dict"))
    old_slots, slots = named_slots(parent, before), named_slots(initial, mean)
    assert equal(old_slots, slots)
    assert all(equal(parent["infos"][key], info[key]) for key in RNG_KEYS)
    assert info["settings"] == expected and info["recovery_only_tuning"]
    assert not any(info.get(key) for key in ("quiet_only_tuning", "quiet_velocity_only_tuning", "arc_only_tuning", "focus_recovery_quiet"))
    assert info["completed_updates"] == parent["infos"]["completed_updates"]
    assert {g["name"]: g["lr"] for g in initial["optimizer_state_dict"]["param_groups"]} == dict(actor=1e-5, critic=1e-3)
    progress = json.loads((output / "progress.json").read_text())
    if "trainable_actor_parameters" in progress:
        assert progress["trainable_actor_parameters"] == 21894
    result = dict(schema="recovery070-state-audit-v1", initial_sha256=sha(initial_path),
        parent_sha256=sha(parent_path), initial_actor_critic_named_Adam_RNG_exact=True,
        task_exact=True, initial_clock=info["completed_updates"],
        updates_seen_at_initial_audit=progress["new_optimizer_updates"])
    if final:
        assert progress["status"] == "completed" and progress["new_optimizer_updates"] == updates
        assert progress["trainable_actor_parameters"] == 21894
        final_path = output / "model_final.pt"
        state = torch.load(final_path, map_location="cpu", weights_only=False)
        after, final_info = load_mean(final_path)
        assert all(finite(state[key]) for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"))
        changed = [key for key, value in state["actor_state_dict"].items() if not equal(value, initial["actor_state_dict"][key])]
        assert changed and all(key.startswith("mlp.mean.recovery.") for key in changed)
        after_slots = named_slots(state, after)
        frozen = [key for key in slots["actor"] if not key.startswith("mlp.mean.recovery.")]
        assert all(equal(slots["actor"][key], after_slots["actor"][key]) for key in frozen)
        steps = {key: int(after_slots["actor"][key]["step"].item()) - int(value.get("step", torch.tensor(0.)).item())
                 for key, value in slots["actor"].items() if key.startswith("mlp.mean.recovery.")}
        assert set(steps.values()) == {12 * updates}
        assert final_info["completed_updates"] == info["completed_updates"] + updates
        assert final_info["settings"] == expected
        result.update(final_sha256=sha(final_path), new_macro_updates=updates,
            final_clock=final_info["completed_updates"], changed_actor=changed,
            frozen_actor_and_named_Adam_exact=True, recovery_Adam_minibatch_step_deltas=steps, finite=True)
    return result


def checked_training(name, command, output, parent, expected, updates):
    if output.exists():
        raise ValueError("Refuse to overwrite training output: " + str(output))
    receipt = dict(schema="recovery070-native-process-v1", command=command, status="starting", started=time.time())
    with (REPORTS / (name + ".log")).open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        receipt.update(pid=child.pid, status="running")
        write(REPORTS / (name + "_process.json"), receipt)
        try:
            while child.poll() is None:
                contract = output / "task_contract.json"
                if not receipt.get("task_exact") and contract.exists():
                    assert json.loads(contract.read_text())["settings"] == expected
                    receipt.update(task_exact=True, initial_exists_at_task_check=(output / "model_initial.pt").exists())
                    write(REPORTS / (name + "_process.json"), receipt)
                if not receipt.get("initial_audited") and zipfile.is_zipfile(output / "model_initial.pt"):
                    write(REPORTS / (name + "_initial.json"), audit(parent, output, expected, updates))
                    receipt["initial_audited"] = True
                    write(REPORTS / (name + "_process.json"), receipt)
                time.sleep(.5)
            code = child.wait()
            progress = json.loads((output / "progress.json").read_text())
            assert code == 0 and progress["status"] == "completed" and receipt.get("task_exact") and receipt.get("initial_audited")
            write(REPORTS / (name + "_state.json"), audit(parent, output, expected, updates, final=True))
            receipt.update(status="completed", returncode=code, finished=time.time(), known_completed=updates)
            write(REPORTS / (name + "_process.json"), receipt)
            return progress
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            progress_path = output / "progress.json"
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            completed = progress.get("new_optimizer_updates", 0)
            receipt.update(status="failed_no_retry", returncode=child.returncode, finished=time.time(),
                known_completed=completed, conservative_in_flight_charge=int(progress.get("status") == "training"),
                error=traceback.format_exc())
            write(REPORTS / (name + "_process.json"), receipt)
            raise


def evaluate(name, checkpoint, *, static=False, seconds=12, seed=211322, nominal=False):
    output = RUNS / name
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(checkpoint),
        "--output", str(output), "--variants", *( ["parent", "robust"] if nominal else ["robust"]),
        "--replicas", "3" if nominal else "16" if static else "4", "--seed", str(seed),
        "--domain-scale", "0" if nominal else ".35", "--seconds", str(seconds), "--isolated-origins"]
    if not nominal:
        command += ["--friction-min", ".3"]
    if static:
        command += ["--case-names", "stand_low", "stand", "stand_high"]
    native(name, command, output)
    for variant in (["parent", "robust"] if nominal else ["robust"]):
        with (REPORTS / (name + "_" + variant + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / variant), "--tensorboard"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return output


def fresh_admission(prefix, checkpoint, protocol):
    from export_robust_getup import assess
    from iteration036_compare import read, standing
    from iteration048_runner import static_failures
    candidates, baselines, extra = [], [], []
    for seed in protocol["required_confirmation_seeds"]:
        name = f"{prefix}_nominal_{seed}"
        output = evaluate(name, checkpoint, seed=seed, nominal=True)
        candidates.append(output / "robust")
        baselines.append(output / "parent")
        for case, row in standing(read(name)).items():
            if row["full_window_max_excursion_m"] > .05 or row["wheel_clearance_after_02s_peak_m"] > .003:
                extra.append(f"{seed} {case} full-window5cm or support3mm")
    result = assess(candidates, baselines, checkpoint, protocol)
    result["failures"].extend(extra)
    result["adequate"] = not result["failures"]
    if result["adequate"]:
        name = prefix + "_standing30"
        evaluate(name, checkpoint, static=True, seconds=30)
        rows = standing(read(name))
        result["long_standing"] = rows
        result["failures"].extend(static_failures(rows, standing(read("quiet_velocity_baseline036"))))
        result["adequate"] = not result["failures"]
    return result


def commit_evidence(paths, message):
    subprocess.run(["git", "add", "--", *map(str, paths)], cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--cached", "--check", "--", *map(str, paths)], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "--only", "-m", message, "--", *map(str, paths)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    args = parser.parse_args()
    plan_path = REPORTS / "long_training070_plan.json"
    plan = json.loads(plan_path.read_text())
    assert plan == make_plan()
    admission_path = REPORTS / "admission067_final.json"
    admission = json.loads(admission_path.read_text())
    require_admission(admission, plan["parent_sha256"])
    state_path = REPORTS / "long_training070_pipeline.json"
    if state_path.exists():
        raise ValueError("Existing long pipeline requires an explicit stopped-state audit; no automatic retry")
    import torch
    from iteration036_compare import read, standing, paired
    from iteration048_runner import static_failures
    torch.set_num_threads(1)
    protocol = json.loads((REPORTS / "acceptance_protocol003.json").read_text())
    parent = ROOT / plan["parent_checkpoint"]
    pipeline = dict(schema="recovery070-long-pipeline-v1", status="running", phase="initializing", blocks=[],
        admission_sha256=sha(admission_path), plan_sha256=sha(plan_path),
        actual_new_long_updates=0, robust_charged=1160, selected_lineage_clock=505, started=time.time(),
        selected_checkpoint=str(parent), selected_checkpoint_sha256=sha(parent),
        hardware_ready=False, installed_in_rmcs=False)
    write(state_path, pipeline)
    next_admission = 1024
    try:
        for index, updates in enumerate(plan["block_sizes"]):
            prefix = f"long_recovery070_b{index:02d}"
            output = RUNS / prefix
            pipeline.update(phase="training", active_block=index, active_updates=updates)
            write(state_path, pipeline)
            import tempfile
            with tempfile.TemporaryDirectory(prefix="recovery070-task-") as folder:
                expected = task_settings(parent, Path(folder) / "preflight", 211431 + index)
            checked_training(prefix, training_arguments(parent, output, updates, 211431 + index),
                             output, parent, expected, updates)
            pipeline["actual_new_long_updates"] += updates
            pipeline["robust_charged"] += updates
            checkpoint = output / "model_final.pt"
            pipeline["phase"] = "behavior_review"
            write(state_path, pipeline)
            static_name = prefix + "_static"
            evaluate(static_name, checkpoint, static=True)
            evidence = read(static_name)
            rows = standing(evidence)
            failures = static_failures(rows, standing(read("quiet_velocity_baseline036")))
            result = dict(schema="recovery070-block-v1", block=index, updates=updates,
                cumulative_new_long_updates=pipeline["actual_new_long_updates"], checkpoint_sha256=sha(checkpoint),
                static=rows, static_pairing=paired(read("high_recenter_static066"), evidence),
                failures=failures, full_suite_complete=False, hardware_ready=False)
            if not failures:
                full_name = prefix + "_full"
                evaluate(full_name, checkpoint)
                full = read(full_name)
                result.update(full_suite_complete=True, full_analysis=full[2],
                    full_pairing=paired(read("high_recenter_full066"), full))
                failures.extend(full_failures(full[2], protocol, read("high_recenter_full066")[2]))
            if not failures and (pipeline["actual_new_long_updates"] >= next_admission or index == len(plan["block_sizes"]) - 1):
                result["fresh_admission"] = fresh_admission(prefix, checkpoint, protocol)
                failures.extend(result["fresh_admission"]["failures"])
                next_admission = (pipeline["actual_new_long_updates"] // 1024 + 1) * 1024
            result["accepted"] = not failures
            result_path = REPORTS / (prefix + "_final.json")
            write(result_path, result)
            pipeline["blocks"].append(dict(block=index, result_sha256=sha(result_path), accepted=not failures,
                result=str(result_path.relative_to(ROOT))))
            if not failures:
                parent = checkpoint
                pipeline.update(selected_checkpoint=str(checkpoint), selected_checkpoint_sha256=sha(checkpoint),
                    selected_lineage_clock=505 + pipeline["actual_new_long_updates"])
            else:
                pipeline.update(status="needs_iteration", phase="behavior_failed", finished=time.time())
            write(state_path, pipeline)
            paths = [result_path, state_path, REPORTS / (prefix + "_process.json"),
                     REPORTS / (prefix + "_initial.json"), REPORTS / (prefix + "_state.json")]
            paths += sorted(REPORTS.glob(prefix + "*_process.json"))
            commit_evidence(list(dict.fromkeys(paths)), f"Seal recovery long-run block{index:02d}: {updates} updates and behavior review")
            if failures:
                print(json.dumps(dict(status="needs_iteration", failures=failures)), flush=True)
                return
        assert pipeline["actual_new_long_updates"] == 5120
        pipeline.update(status="simulation_training_completed", phase="requires_deployment_contract_and_export",
                        finished=time.time())
        write(state_path, pipeline)
        commit_evidence([state_path], "Seal 5120 new recovery updates and final simulation acceptance")
    except BaseException:
        index = pipeline.get("active_block", 0)
        receipt_path = REPORTS / f"long_recovery070_b{index:02d}_process.json"
        if pipeline["phase"] == "training" and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            charged = receipt.get("known_completed", 0) + receipt.get("conservative_in_flight_charge", 0)
            pipeline["actual_new_long_updates"] += receipt.get("known_completed", 0)
            pipeline["robust_charged"] += charged
            pipeline["conservative_in_flight_charge"] = receipt.get("conservative_in_flight_charge", 0)
        pipeline.update(status="failed_no_retry", error=traceback.format_exc(), finished=time.time())
        write(state_path, pipeline)
        raise


if __name__ == "__main__":
    main()

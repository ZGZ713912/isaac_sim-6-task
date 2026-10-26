#!/usr/bin/env python3
"""Read-only CPU integrity seal for a completed from-zero standing experiment.

This does not certify standing, robustness, or deployment. It never creates a
policy, optimizer, simulator, TensorBoard server, or training process.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo.scratch_standing import tensor_hash


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_process_live(identity):
    path = Path("/proc") / str(identity["pid"]) / "stat"
    try:
        # Account for spaces and parentheses in the kernel process name.
        return int(path.read_text().rsplit(") ", 1)[1].split()[19]) == identity["start_ticks"]
    except FileNotFoundError:
        return False


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def seal(run, identity, *, require_current_source=False):
    run = run.resolve()
    if Path(identity.get("run", identity.get("output", ""))).resolve() != run:
        raise ValueError("Process identity belongs to a different experiment")
    if original_process_live(identity):
        raise ValueError("Wait for the original native process to actually exit")
    torch.set_num_threads(2)
    progress = read(run / "progress.json")
    if progress["status"] != "completed_unqualified":
        raise ValueError("Only a completely ended experiment may receive this integrity seal")
    initial, final = [torch.load(run / name, map_location="cpu", weights_only=False)
                      for name in ("model_initial.pt", "model_final.pt")]
    infos, origin = final["infos"], initial["infos"]
    init = read(run / "initialization.json")
    audit = read(run / "first_update_audit.json")
    manifest = read(run / "source_hashes.json")
    task = read(run / "task_contract.json")
    config = read(run / "config/contract.json")
    count = progress["completed_updates"]
    optimizer = final["optimizer_state_dict"]
    parameter_ids = [key for group in optimizer["param_groups"] for key in group["params"]]
    trainable = {role: {name: value for name, value in final[role + "_state_dict"].items()
                       if (name.startswith("mlp.") and name.endswith((".weight", ".bias")))
                       or name == "distribution.log_std_param"} for role in ("actor", "critic")}
    expected_parameters = sum(len(values) for values in trainable.values())
    steps = sorted({int(value["step"].item()) for value in optimizer["state"].values()})
    frozen_mismatch = [name for name, digest in manifest.items()
                       if sha(run / "source_snapshot" / name) != digest]
    live_mismatch = [name for name, digest in manifest.items() if sha(ROOT / name) != digest]
    event, = list(run.glob("events.out.tfevents.*"))
    raw = event.read_bytes()
    with tempfile.TemporaryDirectory() as directory:
        prefix = Path(directory) / event.name
        prefix.write_bytes(raw)
        events = EventAccumulator(str(prefix), size_guidance={"scalars": 0}).Reload()
        tags = ["Train/mean_episode_length", "ScratchStanding/dual_support_physical_fraction",
                "ScratchStanding/leg_speed_rms_rad_s"]
        if progress["args"]["precise_objective"]:
            tags += ["PrecisionStanding/height_mae_m", "PrecisionStanding/tilt_mean_deg",
                     "PrecisionStanding/planar_speed_rms_m_s"]
        event_checks = {}
        for tag in tags:
            rows = events.Scalars(tag)
            event_checks[tag] = len(rows) == count and [row.step for row in rows] == list(range(count)) \
                and all(math.isfinite(row.value) for row in rows)
    descriptor = dict(schema="scratch-standing-checkpoint-v1", training_origin="scratch",
                      continuation_forbidden=True)
    checks = dict(
        original_process_exited=not original_process_live(identity),
        recognized_scratch_checkpoint=all(infos.get(key) == value and origin.get(key) == value
                                         for key, value in descriptor.items()),
        initial_clock_zero=initial["iter"] == origin["completed_updates"] == init["actor_updates"]
            == init["critic_updates"] == 0,
        initial_Adam_empty=not initial["optimizer_state_dict"]["state"] and init["optimizer_states"] == 0,
        no_checkpoint_or_teacher_loaded=init["checkpoint_loaded"] is False and init["teacher_loaded"] is False
            and progress["checkpoint_reads"] == [],
        initialization_receipts_agree=origin["initialization"] == infos["initialization"]
            == progress["initialization"] == init,
        initial_tensor_hashes_match=all(tensor_hash(initial[role + "_state_dict"]) == init[role + "_tensor_sha256"]
                                        for role in ("actor", "critic")),
        completed_update_clocks_agree=count > 0 and count == progress["requested_updates"]
            == progress["optimizer_attempts"] == final["iter"] == infos["completed_updates"]
            and not progress["optimizer_update_in_flight"],
        final_checkpoint_receipt_matches=sha(run / "model_final.pt") == progress["last_checkpoint"]["sha256"]
            and Path(progress["last_checkpoint"]["path"]).resolve() == run / "model_final.pt"
            and progress["last_checkpoint"]["updates"] == count,
        Adam_covers_every_parameter=len(parameter_ids) == len(set(parameter_ids))
            and set(parameter_ids) == set(optimizer["state"]) and len(parameter_ids) == expected_parameters
            and all(sum(value.numel() for value in trainable[role].values())
                    == init[role + "_trainable_parameters"] for role in trainable),
        Adam_steps_match=steps == [12 * count],
        checkpoint_and_Adam_finite=finite(initial) and finite(final),
        first_update_audited=audit["closed"] and audit["all_recorded_values_finite"]
            and audit["committed_optimizer_calls"] == 12 and audit["optimizer_calls"] == 12
            and len(audit["rows"]) == 12 and audit["rows"][0]["adam_step_before_max"] == 0.
            and audit["rows"][-1]["adam_step_after_min"] == 12.
            and audit["actor_changed_parameter_tensors"] > 0 and audit["critic_changed_parameter_tensors"] > 0,
        source_manifest_and_archive_match=manifest == infos["source_hashes"] == origin["source_hashes"]
            and not frozen_mismatch,
        bank_matches=sha(run / "pose_bank.json") == infos["bank_sha256"] == origin["bank_sha256"]
            == task["bank_sha256"],
        mechanism_matches=sha(run / "config/contract.json") == infos["mechanism_config_sha256"]
            == origin["mechanism_config_sha256"] == task["mechanism_config_sha256"],
        design_matches=sha(config["design_preflight"]["file"]) == config["design_preflight"]["sha256"],
        agent_matches=sha(run / "agent_config.json") == infos["agent_config_sha256"]
            == origin["agent_config_sha256"],
        complete_native_RSL_updates=all(event_checks.values()),
    )
    if require_current_source:
        checks["current_source_matches"] = not live_mismatch
    return dict(schema="scratch-standing-integrity-seal-v2",
        observed_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
        pid=identity["pid"], start_ticks=identity["start_ticks"], status=progress["status"],
        completed_updates=count, requested_updates=progress["requested_updates"],
        native_Adam_calls_per_update=12, Adam_parameter_states=len(optimizer["state"]), Adam_step_values=steps,
        source_count=len(manifest), checks=checks, all_checks_pass=all(checks.values()),
        initial_checkpoint_sha256=sha(run / "model_initial.pt"),
        final_checkpoint_sha256=sha(run / "model_final.pt"), frozen_source_mismatches=frozen_mismatch,
        current_source_mismatches=live_mismatch, RSL_event_checks=event_checks,
        RSL_event=dict(path=str(event), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
        new_PPO=0, optimizer_constructed=False, native_steps=0, hardware_ready=False,
        behavior_qualification=False, validator_sha256=sha(Path(__file__)),
        scope="CPU process/checkpoint/source/event integrity only. No standing, robustness, long-training "
              "or deployment qualification. Read-only checkpoint loading outside the exited learner.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-current-source", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".source.py").exists():
        raise ValueError("Preserve the earlier seal and its actual validator")
    result = seal(args.run, read(args.identity), require_current_source=args.require_current_source)
    result["identity_file"] = str(args.identity.resolve())
    result["identity_file_sha256"] = sha(args.identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "completed_updates", "checks", "all_checks_pass")},
                     ensure_ascii=False))
    if not result["all_checks_pass"]:
        raise SystemExit(1)

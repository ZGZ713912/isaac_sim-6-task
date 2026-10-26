#!/usr/bin/env python3
"""Read-only integrity seal after a specific fresh-training process exits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def seal(run, pid, start_ticks):
    process = Path(f"/proc/{pid}/stat")
    original_process_exited = True
    if process.exists():
        data = process.read_text()
        fields = data[data.rfind(")") + 2:].split()
        original_process_exited = int(fields[19]) != start_ticks
    if not original_process_exited:
        raise ValueError("The original native process is still present; do not seal it as exited")
    progress_file, agent_file = run / "progress.json", run / "agent_config.json"
    progress, agent = json.loads(progress_file.read_text()), json.loads(agent_file.read_text())
    if progress["status"] not in ("completed_unqualified", "stopped") or progress.get("optimizer_update_in_flight"):
        raise ValueError("Need a terminal run at a complete optimizer boundary")
    final_file = run / ("model_stopped.pt" if progress["status"] == "stopped" else "model_final.pt")
    initial_file = run / "model_initial.pt"
    initial = torch.load(initial_file, map_location="cpu", weights_only=False)
    final = torch.load(final_file, map_location="cpu", weights_only=False)
    infos = final["infos"]
    initial_optimizer, final_optimizer = initial["optimizer_state_dict"], final["optimizer_state_dict"]
    parameter_ids = [i for group in final_optimizer["param_groups"] for i in group["params"]]
    completed = progress["completed_updates"]
    calls_per_update = agent["algorithm"]["num_learning_epochs"] * agent["algorithm"]["num_mini_batches"]
    adam_steps = sorted({int(value["step"].item()) for value in final_optimizer["state"].values()})
    source_hashes = json.loads((run / "source_hashes.json").read_text())
    frozen_bad = [name for name, expected in source_hashes.items()
                  if sha(run / "source_snapshot" / name) != expected]
    archive_sources_match = not frozen_bad and source_hashes == infos["source_hashes"]
    audit = json.loads((run / "first_update_audit.json").read_text()) if completed else None
    checks = dict(
        original_process_exited=original_process_exited,
        recognized_scratch_checkpoint=infos.get("schema") == "scratch-standing-checkpoint-v1",
        continuation_forbidden=infos.get("continuation_forbidden") is True,
        fresh_initialization=all(infos["initialization"].get(name) == value for name, value in
            (("checkpoint_loaded", False), ("teacher_loaded", False), ("actor_updates", 0),
             ("critic_updates", 0), ("optimizer_states", 0))),
        empty_initial_Adam=not initial_optimizer["state"],
        initial_learning_clock_zero=initial["infos"]["completed_updates"] == 0,
        training_checkpoint_reads_zero=progress["checkpoint_reads"] == [],
        completed_clock_matches=infos["completed_updates"] == completed == progress["optimizer_attempts"],
        checkpoint_matches_last_receipt=sha(final_file) == progress["last_checkpoint"]["sha256"],
        Adam_contains_every_parameter=set(final_optimizer["state"]) == set(parameter_ids),
        Adam_steps_match_complete_updates=adam_steps == ([completed * calls_per_update] if completed else []),
        parameters_and_Adam_finite=all(finite(final[name]) for name in
            ("actor_state_dict", "critic_state_dict", "optimizer_state_dict")),
        frozen_source_archive_matches=archive_sources_match,
        bank_matches=sha(run / "pose_bank.json") == infos["bank_sha256"],
        mechanism_matches=sha(run / "config/contract.json") == infos["mechanism_config_sha256"],
        agent_matches=infos.get("agent_config_sha256") in (None, sha(agent_file)),
        first_update_audited=(completed == 0 or (audit["all_recorded_values_finite"]
            and audit["committed_optimizer_calls"] == calls_per_update
            and audit["actor_changed_parameter_tensors"] > 0 and audit["critic_changed_parameter_tensors"] > 0)))
    if not all(checks.values()):
        raise ValueError("Integrity seal failed: " + ", ".join(name for name, passed in checks.items() if not passed))
    return dict(schema="scratch-standing-integrity-seal-v1", observed_utc=datetime.now(timezone.utc).isoformat(),
        run=str(run), pid=pid, start_ticks=start_ticks, status=progress["status"], completed_updates=completed,
        requested_updates=progress["requested_updates"], native_Adam_calls_per_update=calls_per_update,
        Adam_parameter_states=len(final_optimizer["state"]), Adam_step_values=adam_steps,
        source_count=len(source_hashes), checks=checks, all_checks_pass=True,
        initial_checkpoint_sha256=sha(initial_file), final_checkpoint_sha256=sha(final_file),
        new_PPO=0, optimizer_constructed=False, hardware_ready=False,
        scope="CPU checkpoint/source/process integrity only. Does not certify standing, robustness or deployment.",
        validator_sha256=sha(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start-ticks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = seal(args.run.resolve(), args.pid, args.start_ticks)
    if args.output.exists():
        raise ValueError("Preserve old seals; use a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report))

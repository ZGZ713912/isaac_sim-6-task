#!/usr/bin/env python3
"""Authenticate a specific live scratch job and preserve a read-only observation.

No checkpoint load, policy, optimizer, simulator, lifecycle action or training
admission is performed. Historical observations are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observe(identity_path, comparison_path, plot_path, output, campaign_path):
    archived_reader = output.with_suffix(".reader.py")
    if output.exists() or archived_reader.exists():
        raise ValueError("Preserve the previous live observation and its exact reader")
    identity = read(identity_path)
    run = Path(identity["run"])
    process = Path("/proc") / str(identity["pid"])
    stat = process.joinpath("stat").read_text().rsplit(") ", 1)[1].split()
    if int(stat[19]) != identity["start_ticks"] or stat[0] == "Z":
        raise ValueError("The original native worker is no longer live; inspect terminal state and seal, never restart here")
    command = [part.decode() for part in process.joinpath("cmdline").read_bytes().split(b"\0") if part]
    if str(run) not in command or str(ROOT / "scripts/train_scratch_standing.py") not in command:
        raise ValueError("The live native command does not match this run")
    progress = read(run / "progress.json")
    manifest = read(run / "source_hashes.json")
    if not all(sha(ROOT / name) == expected == sha(run / "source_snapshot" / name)
               for name, expected in manifest.items()):
        raise ValueError("Current or frozen training source changed")
    gpu_pids = [int(line.strip()) for line in subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).splitlines() if line.strip()]
    if identity["pid"] not in gpu_pids:
        raise ValueError("The original process is live but not present on the GPU; inspect this same worker")
    comparison = read(comparison_path)
    if run.name not in comparison["sources"]:
        raise ValueError("Comparison does not include this live training run")
    for source in comparison["sources"].values():
        with Path(source["path"]).open("rb") as stream:
            prefix = stream.read(source["observed_prefix_bytes"])
        if hashlib.sha256(prefix).hexdigest() != source["observed_prefix_sha256"]:
            raise ValueError("Original RSL event prefix changed")
    plot_receipt = read(plot_path.with_suffix(".receipt.json"))
    if (sha(plot_path) != plot_receipt["plot_sha256"]
            or sha(comparison_path) != plot_receipt["comparison_sha256"]
            or not plot_receipt["every_plotted_window_verified_from_original_events"]):
        raise ValueError("Plot does not authenticate this original event comparison")
    count = progress["completed_updates"]
    if not 0 <= count <= identity["requested_updates"] or progress["checkpoint_reads"]:
        raise ValueError("Invalid scratch update clock or forbidden training checkpoint reads")
    record = dict(schema="scratch-pilot-live-observation-v2", observed_utc=datetime.now(timezone.utc).isoformat(),
        classification="verified_wait",
        original_pid=identity["pid"], original_start_ticks=identity["start_ticks"],
        original_process_live=True, original_GPU_present=True, status=progress["status"],
        completed_updates=count, requested_updates=identity["requested_updates"], training_checkpoint_reads=[],
        prior_completed_fresh_updates=identity["prior_from_zero_completed"],
        total_completed_fresh_updates=identity["prior_from_zero_completed"] + count,
        source_count=len(manifest), frozen_source_bytes_unchanged=True, source_commit=identity["commit"],
        latest_complete_comparison=str(comparison_path), comparison_sha256=sha(comparison_path),
        common_logged_updates=comparison["common_complete_updates"], complete_windows=len(comparison["windows"]),
        last_complete_window=comparison["windows"][-1], plot=str(plot_path), plot_sha256=sha(plot_path),
        qualification=False, hardware_ready=False, new_native_steps_by_observer=0, new_PPO_by_observer=0,
        checkpoint_reads_by_observer=0, reader_sha256=sha(__file__),
        next_action="Observe the same native worker through its fixed budget and original evaluation. "
                    "Verify actual process/GPU exit and seal before another native job. No hotpatch, restart, resume or average-training admission.")
    output.parent.mkdir(parents=True, exist_ok=True)
    archived_reader.write_bytes(Path(__file__).read_bytes())
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    if campaign_path is not None:
        campaign = read(campaign_path)
        campaign.update(status=record["status"], active_run=str(run), active_pid=identity["pid"],
            active_start_ticks=identity["start_ticks"], completed_updates_observed=count,
            total_from_zero_completed=record["total_completed_fresh_updates"],
            total_from_zero_completed_observed=record["total_completed_fresh_updates"],
            latest_live_read=str(output), latest_event_comparison=str(comparison_path),
            latest_active_run_event_comparison=str(comparison_path), latest_training_plot=str(plot_path),
            latest_training_plot_runs=list(comparison["sources"]),
            last_observed_utc=record["observed_utc"], next_action=record["next_action"])
        temporary = campaign_path.with_suffix(campaign_path.suffix + ".tmp")
        temporary.write_text(json.dumps(campaign, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(campaign_path)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    record = observe(args.identity.resolve(), args.comparison.resolve(), args.plot.resolve(),
                     args.output.resolve(), args.campaign.resolve() if args.campaign else None)
    print(json.dumps({key: record[key] for key in ("original_pid", "original_process_live", "original_GPU_present",
        "status", "completed_updates", "complete_windows", "source_count", "qualification")}))

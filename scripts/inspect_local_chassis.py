#!/usr/bin/env python3
"""Read a local learner's existing receipts and TensorBoard events without a server."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def inspect(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    directory = Path(directory).resolve()
    contract = json.loads((directory / "contract.json").read_text())
    completed = directory / "completion.json"
    progress = json.loads((directory / ("completion.json" if completed.exists() else "progress.json")).read_text())
    updates = progress.get("parent_updates", 0) + progress["successful_updates"]
    prior = progress.get("prior_local_experiment_consumption", 0)
    result = dict(run_dir=str(directory), inspected_at=datetime.now(timezone.utc).isoformat(),
        status=progress.get("status", "in_progress"), successful_local_updates=updates,
        prior_experiment_consumption=prior, total_consumed_updates=prior + updates,
        policy_hz=1 / contract["policy_dt"], physics_hz=1 / contract["physics_dt"],
        rollout_seconds=contract["policy_dt"] * contract["num_steps_per_env"])
    if not completed.exists():
        # A live PID alone is insufficient: verify the actual learner's run argument.
        pid = progress["pid"]
        try:
            command = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().split("\0")
        except OSError:
            command = []
        pairs = zip(command, command[1:])
        result.update(pid=pid, learner_process_matches=any(
            flag == "--run-dir" and Path(value).resolve() == directory for flag, value in pairs),
            last_progress_at=progress.get("updated_at"))
    events = EventAccumulator(str(directory), size_guidance={"scalars": 200})
    events.Reload()
    wanted = ("Loss/value", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate",
              "Policy/mean_std", "Train/mean_reward", "Train/mean_episode_length")
    result["tensorboard"] = {}
    for tag in wanted:
        if tag not in events.Tags()["scalars"]:
            continue
        rows = events.Scalars(tag)
        last = rows[-1]
        finite = math.isfinite(last.value)
        item = dict(step=last.step, value=last.value if finite else None,
            latest_sample_finite=finite,
            finite_in_retained_samples=all(math.isfinite(row.value) for row in rows),
            retained_samples=len(rows))
        if tag == "Train/mean_episode_length":
            item["seconds"] = last.value * contract["policy_dt"] if finite else None
        result["tensorboard"][tag] = item
    result["finite_check_scope"] = "at most 200 retained event samples per selected tag; not a convergence check"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.run_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

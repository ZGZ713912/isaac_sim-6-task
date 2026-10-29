#!/usr/bin/env python3
"""Change the declared quiet wheel sampling fraction, preserving every mean."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    import argparse
    import math
    import torch
    from export_robust_getup import load_mean
    from train_robust_getup import digest, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--experiment-consumed", type=int, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.fraction) or not .02 <= args.fraction <= .5:
        parser.error("Use a finite quiet-wheel fraction within [.02,.5]")
    if args.output.exists():
        raise ValueError("Never overwrite a checkpoint")
    load_mean(args.source)
    original = torch.load(args.source, map_location="cpu", weights_only=False)
    state = deepcopy(original)
    task = state["infos"]["settings"]
    before = task.get("quiet_wheel_exploration_fraction")
    if task.get("gap_entry_handoff_version") != "orbit-v2" or before == args.fraction:
        raise ValueError("Require an orbit-v2 policy and a changed wheel fraction")
    task["quiet_wheel_exploration_fraction"] = args.fraction
    revision = dict(source=str(args.source.resolve()), source_sha256=digest(args.source),
                    previous_fraction=before, fraction=args.fraction, optimizer_updates=0,
                    hypothesis="Reduce wheel sampling disturbance in quiet exposure while retaining exploration elsewhere")
    state["infos"]["quiet_wheel_exploration_migration"] = revision
    state["infos"]["hardware_ready"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    load_mean(args.output)
    write_json(args.output.with_suffix(".json"), dict(
        schema="quiet-wheel-exploration-migration-v1", **revision,
        checkpoint_sha256=digest(args.output), learning_clock=state["infos"]["completed_updates"],
        mean_actor_critic_optimizer_rng_unchanged=True, new_PPO=0,
        experiment_consumed=args.experiment_consumed, qualification=None, hardware_ready=False))
    print(args.output)


if __name__ == "__main__":
    main()

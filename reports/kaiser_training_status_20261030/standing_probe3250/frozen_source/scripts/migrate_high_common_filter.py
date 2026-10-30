#!/usr/bin/env python3
"""Declare a single zero-PPO probe of the existing high-stance common filter."""
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
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--experiment-consumed", type=int, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or not 0. < args.alpha <= 1.:
        parser.error("Require a finite common filter alpha in (0, 1]")
    if args.output.exists():
        raise ValueError("Never overwrite an experiment checkpoint")
    load_mean(args.source)
    original = torch.load(args.source, map_location="cpu", weights_only=False)
    state = deepcopy(original)
    task = state["infos"]["settings"]
    before = task.get("quiet_high_common_alpha")
    if task.get("quiet_velocity_feedback_mode") != "standing_displacement" or before == args.alpha:
        raise ValueError("Require a changed filter option and displacement feedback")
    task["quiet_high_common_alpha"] = args.alpha
    revision = dict(source=str(args.source.resolve()), source_sha256=digest(args.source),
                    previous_alpha=before, alpha=args.alpha, optimizer_updates=0,
                    hypothesis="Test the declared high-stance paired common-angle interpolation in matched native rollouts",
                    physical_changes=False, wheel_filter_changes=False, knee_filter_changes=False)
    state["infos"]["high_common_filter_probe"] = revision
    state["infos"]["hardware_ready"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    load_mean(args.output)
    write_json(args.output.with_suffix(".json"), dict(
        schema="high-common-filter-probe-v1", **revision,
        checkpoint_sha256=digest(args.output), learning_clock=state["infos"]["completed_updates"],
        actor_critic_optimizer_rng_exact=True, new_PPO=0,
        experiment_consumed=args.experiment_consumed, qualification=None, hardware_ready=False))
    print(args.output)


if __name__ == "__main__":
    main()

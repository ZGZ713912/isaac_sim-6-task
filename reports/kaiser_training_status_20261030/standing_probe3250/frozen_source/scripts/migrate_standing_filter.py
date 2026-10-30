#!/usr/bin/env python3
"""Declare one zero-PPO standing observation filter-time experiment."""
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
    from wheeled_tasks.chassis.standing_recovery_reference import validate_observation_contract

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--experiment-consumed", type=int, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not .02 <= args.seconds <= .2:
        parser.error("Use a finite observation filter duration within .02.. .2 seconds")
    if args.output.exists():
        raise ValueError("Never overwrite an experiment checkpoint")
    load_mean(args.source)
    original = torch.load(args.source, map_location="cpu", weights_only=False)
    state = deepcopy(original)
    task = state["infos"]["settings"]
    if task.get("quiet_velocity_feedback_mode") != "standing_displacement":
        raise ValueError("Require the explicit standing displacement observation")
    before = task["standing_position_filter_seconds"]
    if before == args.seconds:
        raise ValueError("Requested observation filter is already active")
    task["standing_position_filter_seconds"] = args.seconds
    task["standing_position_observation"]["filter_seconds"] = args.seconds
    validate_observation_contract(task)

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return a == b

    checks = {key: equal(value, state[key]) for key, value in original.items() if key != "infos"}
    checks.update({key: equal(value, state["infos"][key]) for key, value in original["infos"].items()
                   if key not in ("settings", "observation_filter_migration", "hardware_ready")})
    if not all(checks.values()):
        raise ValueError("Filter migration changed learned, optimizer, random or ancestry state")
    revision = dict(source=str(args.source.resolve()), source_sha256=digest(args.source),
                    previous_filter_seconds=before, filter_seconds=args.seconds,
                    hypothesis="Reduce displacement feedback lag before further PPO",
                    new_optimizer_updates=0, physical_changes=False)
    state["infos"]["observation_filter_migration"] = revision
    state["infos"]["hardware_ready"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    load_mean(args.output)
    write_json(args.output.with_suffix(".json"), dict(
        schema="standing-observation-filter-migration-v1", **revision, checks=checks,
        checkpoint_sha256=digest(args.output), completed_updates=state["infos"]["completed_updates"],
        experiment_consumed_at_preparation=args.experiment_consumed,
        qualification=None, hardware_ready=False,
    ))
    print(args.output)


if __name__ == "__main__":
    main()

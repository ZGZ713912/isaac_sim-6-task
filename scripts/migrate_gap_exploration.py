#!/usr/bin/env python3
"""Change one conditional training-noise scale, preserving the complete mean."""
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
    from migrate_gap_pose_bank import recursive_equal
    from train_robust_getup import digest, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--experiment-consumed", type=int, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.fraction) or not .1 <= args.fraction < 1.:
        parser.error("Use one finite fallen leg-noise fraction within [.1,1)")
    if args.output.exists():
        raise ValueError("Never overwrite an exploration migration")
    load_mean(args.source)
    original = torch.load(args.source, map_location="cpu", weights_only=False)
    state = deepcopy(original)
    info = state["infos"]
    task = info["settings"]
    if task.get("gap_entry_handoff_version") != "orbit-v2":
        raise ValueError("Exploration migration requires authenticated orbit-v2 authority")
    key = "gap_recovery_leg_exploration_fraction"
    previous = task.get(key, 1.)
    if previous == args.fraction:
        raise ValueError("Requested exploration scale is already active")
    task[key] = args.fraction
    source = "src/wheeled_algo/gap_recovery_exploration.py"
    info["source_sha256"][source] = digest(ROOT / source)
    revision = dict(source=str(args.source.resolve()), source_sha256=digest(args.source),
        previous_fraction=previous, recovery_leg_fraction=args.fraction,
        source_module=source, source_module_sha256=info["source_sha256"][source],
        scope="four leg log-std only, fallen tilt authority times one minus upright-entry authority",
        deterministic_mean_changed=False, wheel_distribution_changed=False,
        Gaussian_likelihood="the same modified log-std is used for sampling and native PPO likelihood",
        new_optimizer_updates=0, new_tensors=0)
    info["recovery_exploration_migration"] = revision
    info["hardware_ready"] = False
    checks = {k: recursive_equal(v, state[k]) for k, v in original.items() if k != "infos"}
    checks.update({k: recursive_equal(v, info[k]) for k, v in original["infos"].items()
                   if k not in ("settings", "source_sha256", "recovery_exploration_migration", "hardware_ready")})
    changed_settings = [k for k in set(original["infos"]["settings"]) | set(task)
                        if not recursive_equal(original["infos"]["settings"].get(k), task.get(k))]
    if not all(checks.values()) or changed_settings != [key]:
        raise ValueError("Exploration migration altered learned, Adam, RNG or unrelated task state")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    load_mean(args.output)
    write_json(args.output.with_suffix(".json"), dict(schema="gap-fallen-exploration-migration-v1",
        **revision, checkpoint_sha256=digest(args.output), checks=checks,
        completed_updates=info["completed_updates"],
        experiment_consumed_at_preparation=args.experiment_consumed,
        qualification=None, hardware_ready=False))
    print(args.output)


if __name__ == "__main__":
    main()

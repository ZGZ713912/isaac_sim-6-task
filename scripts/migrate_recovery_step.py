#!/usr/bin/env python3
"""Make one explicit, zero-PPO change to the learned recovery action space."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    import argparse
    import torch
    from export_robust_getup import load_mean
    from train_robust_getup import digest, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--experiment-consumed", type=int, required=True)
    args = parser.parse_args()
    if not .05 <= args.step <= .3:
        parser.error("Common-leg raw-action step must be within .05.. .3")
    if args.output.exists():
        raise ValueError("Never overwrite an experiment checkpoint")
    load_mean(args.source)
    original = torch.load(args.source, map_location="cpu", weights_only=False)
    migrated = deepcopy(original)
    info = migrated["infos"]
    previous = info["settings"]["recovery_leg_step"]
    if previous == args.step:
        raise ValueError("Requested recovery action space is already active")
    if not info["settings"].get("smooth_knee_targets"):
        raise ValueError("This diagnostic requires the existing smooth-knee action coordinates")
    info["settings"]["recovery_leg_step"] = args.step
    info["recovery_step_migration"] = dict(
        source=str(args.source.resolve()), source_sha256=digest(args.source),
        previous_step=previous, step=args.step,
        coordinate="learned same-leg common rotation, raw action per policy tick",
        changes="recovery_leg_step only; existing knee map, mean weights, sensors and guards retained",
        new_optimizer_updates=0, tensor_changes=0, diagnostic_only=True,
    )
    info["hardware_ready"] = False

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return a == b

    settings_changes = [key for key, value in original["infos"]["settings"].items()
                        if not equal(value, info["settings"][key])]
    if settings_changes != ["recovery_leg_step"]:
        raise ValueError("Unrelated action or sensor contract changed")
    checks = {key: equal(value, migrated[key]) for key, value in original.items() if key != "infos"}
    checks.update({key: equal(value, info[key]) for key, value in original["infos"].items()
                   if key not in ("settings", "hardware_ready", "recovery_step_migration")})
    if not all(checks.values()):
        raise ValueError("Zero-PPO migration changed learned, optimizer, random or lineage state")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, args.output)
    load_mean(args.output)
    write_json(args.output.with_suffix(".json"), dict(
        schema="single-recovery-step-migration-v1", source=str(args.source.resolve()),
        source_sha256=digest(args.source), checkpoint_sha256=digest(args.output),
        checks=checks, settings_changes=settings_changes,
        previous_step=previous, step=args.step,
        completed_updates=info["completed_updates"],
        experiment_consumed_at_preparation=args.experiment_consumed,
        new_optimizer_updates=0, qualification=None, hardware_ready=False,
    ))
    print(args.output)


if __name__ == "__main__":
    main()

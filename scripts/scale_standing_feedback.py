#!/usr/bin/env python3
"""Prepare an explicit zero-PPO scalar probe of the 24 learned feedback weights."""
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
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--heights", choices=("low", "mid", "high"), nargs="+", default=["low", "mid", "high"])
    parser.add_argument("--experiment-consumed", type=int, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.scale) or not .1 <= args.scale <= 16. or args.scale == 1.:
        parser.error("Use a finite diagnostic scale within .1..16, different from one")
    if len(set(args.heights)) != len(args.heights):
        parser.error("Select each height expert at most once")
    if args.output.exists():
        raise ValueError("Never overwrite an experiment checkpoint")
    load_mean(args.source)
    original = torch.load(args.source, map_location="cpu", weights_only=False)
    state = deepcopy(original)
    if state["infos"]["settings"].get("quiet_velocity_feedback_mode") != "standing_displacement":
        raise ValueError("This probe is only for the explicit displacement feedback path")
    keys = ["mlp.mean.quiet_velocity." + height + ".weight" for height in args.heights]
    for key in keys:
        if state["actor_state_dict"][key].shape != (4, 2):
            raise ValueError("Expected the three bias-free common-leg/wheel feedback heads")
        state["actor_state_dict"][key] *= args.scale
    for key, value in original["actor_state_dict"].items():
        if key not in keys and not torch.equal(value, state["actor_state_dict"][key]):
            raise ValueError("Probe changed an unrelated actor tensor")
    state["infos"]["feedback_scale_probe"] = dict(
        source=str(args.source.resolve()), source_sha256=digest(args.source),
        scale=args.scale, keys=keys, changed_parameters=8 * len(keys), optimizer_updates=0,
        optimizer="unchanged, not an equivalent coordinate transformation",
        hypothesis="Test the response to displacement gain before spending more PPO on drift credit",
    )
    state["infos"]["hardware_ready"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    load_mean(args.output)
    write_json(args.output.with_suffix(".json"), dict(
        schema="standing-feedback-scalar-probe-v1", source_sha256=digest(args.source),
        checkpoint_sha256=digest(args.output), scale=args.scale,
        changed_actor_tensors=keys, changed_parameters=8 * len(keys),
        all_other_actor_tensors_unchanged=True, critic_optimizer_rng_unchanged=True,
        new_optimizer_updates=0, completed_updates=state["infos"]["completed_updates"],
        experiment_consumed_at_preparation=args.experiment_consumed,
        qualified=False, hardware_ready=False,
    ))
    print(args.output)


if __name__ == "__main__":
    main()

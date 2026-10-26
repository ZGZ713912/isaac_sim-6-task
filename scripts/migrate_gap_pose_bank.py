#!/usr/bin/env python3
"""Bind an append-only reset-bank revision without changing learned state."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def validate_append_only_bank(before, after, source_sha256):
    old, new = before["poses"], after["poses"]
    extension = after.get("extension", {})
    if (not old or len(new) <= len(old) or new[:len(old)] != old
            or extension.get("source_sha256") != source_sha256
            or extension.get("original_pose_count") != len(old)
            or extension.get("appended_pose_count") != len(new) - len(old)
            or extension.get("original_poses_unchanged") is not True):
        raise ValueError("Reset-bank revision must preserve every old pose and index")
    prior = {k: v for k, v in before.items() if k not in ("poses", "extension")}
    revised = {k: v for k, v in after.items() if k not in ("poses", "extension")}
    if prior != revised:
        raise ValueError("Reset-bank revision changed the asset or reset contract")
    names = [p["name"] for p in new]
    if len(names) != len(set(names)):
        raise ValueError("Reset-bank pose names must remain unique")
    joint_names = set(old[0]["full_joint_positions"])
    for pose in new[len(old):]:
        if (set(pose["full_joint_positions"]) != joint_names
                or not all(math.isfinite(x) for x in pose["full_joint_positions"].values())
                or pose.get("rmcs_reference") is not True):
            raise ValueError("Appended poses require finite, complete RMCS joint states")
    return dict(original_pose_count=len(old), appended_pose_count=len(new) - len(old),
                original_poses_and_indices_exact=True, reset_contract_exact=True)


def recursive_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(recursive_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(recursive_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def main():
    import argparse
    import torch
    from build_robust_getup_bank import Builder
    from export_robust_getup import load_mean
    from train_robust_getup import digest, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--target-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-consumed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never overwrite a checkpoint migration")
    load_mean(args.source)
    original = torch.load(args.source, map_location="cpu", weights_only=False)
    source_sha, target_sha = digest(args.source_bank), digest(args.target_bank)
    if original["infos"]["bank_sha256"] != source_sha:
        raise ValueError("The source bank does not authenticate this checkpoint")
    before = json.loads(args.source_bank.read_text())
    after = json.loads(args.target_bank.read_text())
    validation = validate_append_only_bank(before, after, source_sha)
    bundle = ROOT / after["asset_directory"]
    if (digest(bundle / "manifest.json") != after["asset_manifest_sha256"]
            or digest(bundle / "model_spec.json") != after["model_spec_sha256"]):
        raise ValueError("Reset bank differs from the current native mechanism")
    builder = Builder(bundle, after["asset_manifest_sha256"])
    physical = [builder.validate_q(p["full_joint_positions"])
                for p in after["poses"][len(before["poses"]):]]
    state = deepcopy(original)
    state["infos"]["bank_sha256"] = target_sha
    revision = dict(source=str(args.source.resolve()), source_sha256=digest(args.source),
        source_bank=str(args.source_bank.resolve()), source_bank_sha256=source_sha,
        target_bank=str(args.target_bank.resolve()), target_bank_sha256=target_sha,
        **validation, appended_closure_gap_max_m=max(p["gap_m"] for p in physical),
        new_optimizer_updates=0, sampling_distribution_changed=True,
        actuator_or_reward_changes=False, dynamic_qualification=False)
    state["infos"]["pose_bank_migration"] = revision
    state["infos"]["hardware_ready"] = False
    checks = {k: recursive_equal(v, state[k]) for k, v in original.items() if k != "infos"}
    checks.update({k: recursive_equal(v, state["infos"][k]) for k, v in original["infos"].items()
                   if k not in ("bank_sha256", "pose_bank_migration", "hardware_ready")})
    if not all(checks.values()):
        raise ValueError("Reset-bank migration changed learned, optimizer, RNG or lineage state")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    load_mean(args.output)
    write_json(args.output.with_suffix(".json"), dict(schema="append-only-gap-pose-bank-migration-v1",
        **revision, checkpoint_sha256=digest(args.output), checks=checks,
        completed_updates=state["infos"]["completed_updates"],
        experiment_consumed_at_preparation=args.experiment_consumed,
        qualification=None, hardware_ready=False))
    print(args.output)


if __name__ == "__main__":
    main()

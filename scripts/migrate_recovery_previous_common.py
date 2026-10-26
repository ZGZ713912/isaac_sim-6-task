#!/usr/bin/env python3
"""Declare circular previous-target coordinates without taking an optimizer step."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
SOURCES = ("src/wheeled_algo/robust_getup.py", "src/wheeled_algo/recovery_coordinates.py",
           "src/wheeled_algo/gap_entry_handoff.py", "src/wheeled_algo/gap_entry_preservation.py")


def migrate(source, source_root, output, version, experiment_consumed):
    import torch
    from export_robust_getup import load_mean
    from migrate_gap_pose_bank import recursive_equal
    from train_robust_getup import digest, write_json

    source, source_root, output = map(Path, (source, source_root, output))
    if output.exists():
        raise ValueError("Never overwrite a migration checkpoint")
    if version not in (None, "circular-v1"):
        raise ValueError("Unknown previous recovery common-angle version")
    if isinstance(experiment_consumed, bool) or not isinstance(experiment_consumed, int) or experiment_consumed < 0:
        raise ValueError("Require the actual nonnegative experiment consumption")
    original = torch.load(source, map_location="cpu", weights_only=False)
    for name in SOURCES:
        expected = original["infos"]["source_sha256"].get(name)
        if expected is not None and digest(source_root / name) != expected:
            raise ValueError("Archived actor source differs: " + name)
    load_mean(source, migration_source_root=source_root)
    state = deepcopy(original)
    info = state["infos"]
    task = info["settings"]
    if not task.get("smooth_knee_targets") or not task.get("cyclic_leg_targets"):
        raise ValueError("This correction requires the existing cyclic smooth-knee action space")
    old_version = task.get("recovery_previous_common_version")
    if version is None:
        task.pop("recovery_previous_common_version", None)
    else:
        task["recovery_previous_common_version"] = version
    for name in SOURCES:
        info["source_sha256"][name] = digest(ROOT / name)
    receipt = dict(schema="recovery-previous-common-migration-v1", source=str(source.resolve()),
        source_sha256=digest(source), archived_source_root=str(source_root.resolve()),
        previous_version=old_version, version=version, new_PPO=0,
        scope="Only circular common phase of previous paired leg targets, in fallen and upright entry paths",
        hard_limit_or_torque_changes=False, runtime_fsm_or_trajectory=False)
    info["recovery_previous_common_migration"] = receipt
    info["hardware_ready"] = False
    checks = {key: recursive_equal(value, state[key]) for key, value in original.items() if key != "infos"}
    changes = [key for key in set(task) | set(original["infos"]["settings"])
               if not recursive_equal(task.get(key), original["infos"]["settings"].get(key))]
    if not all(checks.values()) or set(changes) - {"recovery_previous_common_version"}:
        raise ValueError("Migration changed learned, optimizer, RNG, clock or unrelated task state")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    load_mean(output)
    write_json(output.with_suffix(".json"), dict(**receipt, checkpoint_sha256=digest(output),
        checks=checks, changed_settings=changes, completed_updates=info["completed_updates"],
        experiment_consumed=experiment_consumed, qualification=None, hardware_ready=False))
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", choices=("none", "circular-v1"), required=True)
    parser.add_argument("--experiment-consumed", type=int, required=True)
    args = parser.parse_args()
    print(migrate(args.source, args.source_root, args.output,
                  None if args.version == "none" else args.version, args.experiment_consumed))

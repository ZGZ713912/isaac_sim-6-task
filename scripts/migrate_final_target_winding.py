#!/usr/bin/env python3
"""Bind a declared circular final-pair representation without optimizer work."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
SOURCES = ("src/wheeled_algo/upright_target_limit.py", "src/wheeled_algo/gap_entry_preservation.py")


def migrate(source, source_root, output, version, experiment_consumed):
    import torch
    from export_robust_getup import load_mean
    from migrate_gap_pose_bank import recursive_equal
    from train_robust_getup import digest, write_json
    if output.exists():
        raise ValueError("Never overwrite a migration checkpoint")
    if version not in (None, "paired-v1"):
        raise ValueError("Unknown final target winding version")
    if isinstance(experiment_consumed, bool) or not isinstance(experiment_consumed, int) or experiment_consumed < 0:
        raise ValueError("Require actual nonnegative experiment consumption")
    original = torch.load(source, map_location="cpu", weights_only=False)
    for name in SOURCES:
        expected = original["infos"]["source_sha256"].get(name)
        if expected is not None and digest(source_root / name) != expected:
            raise ValueError("Archived actor source differs: " + name)
    load_mean(source, migration_source_root=source_root)
    state = deepcopy(original)
    info, task = state["infos"], state["infos"]["settings"]
    if task.get("gap_entry_handoff_version") != "orbit-v2" or task.get("gap_upright_target_limit_version") != "error-v1":
        raise ValueError("This migration requires orbit-v2 and error-v1")
    if version is None:
        task.pop("gap_final_target_winding_version", None)
    else:
        task["gap_final_target_winding_version"] = version
    for name in SOURCES:
        info["source_sha256"][name] = digest(ROOT / name)
    checks = {key: recursive_equal(value, state[key]) for key, value in original.items() if key != "infos"}
    changed = [key for key in set(task) | set(original["infos"]["settings"])
               if not recursive_equal(task.get(key), original["infos"]["settings"].get(key))]
    if not all(checks.values()) or set(changed) - {"gap_final_target_winding_version"}:
        raise ValueError("Migration changed learned, optimizer, RNG, clock or unrelated settings")
    info["hardware_ready"] = False
    info["final_target_winding_migration"] = dict(schema="final-pair-winding-migration-v1",
        source=str(source.resolve()), source_sha256=digest(source), version=version,
        source_root=str(source_root.resolve()), new_PPO=0,
        scope="Rebase independently wrapped final hip/aux targets together before the unchanged paired_branch107_v1 decoder",
        requested_physical_angles_preserved_for_feasible_targets=True,
        physics_or_torque_limit_changed=False, runtime_fsm_or_trajectory=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    load_mean(output)
    write_json(output.with_suffix(".json"), dict(**info["final_target_winding_migration"],
        checkpoint=str(output), checkpoint_sha256=digest(output), checks=checks, changed_settings=changed,
        completed_updates=info["completed_updates"], experiment_consumed=experiment_consumed,
        qualification=None, hardware_ready=False))
    return output


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--version", choices=("none", "paired-v1"), required=True)
    p.add_argument("--experiment-consumed", type=int, required=True)
    args = p.parse_args()
    print(migrate(args.source, args.source_root, args.output,
                  None if args.version == "none" else args.version, args.experiment_consumed))

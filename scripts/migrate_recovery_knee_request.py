#!/usr/bin/env python3
"""Declare a fallen-only knee-request change with exact learned-state retention."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
ACTOR_SOURCE = "src/wheeled_algo/robust_getup.py"


def main():
    import argparse
    import inspect
    import math
    import torch
    from export_robust_getup import load_mean
    from migrate_gap_pose_bank import recursive_equal
    from train_robust_getup import digest, write_json
    from wheeled_algo.robust_getup import RobustGetupActor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Archived old source tree proving the precise actor revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--experiment-consumed", type=int, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.step) or not .05 <= args.step <= .45:
        parser.error("Use finite measured-relative knee lead within .05.. .45 rad")
    if args.output.exists():
        raise ValueError("Never overwrite a migration checkpoint")
    if "recovery_knee_step" not in inspect.signature(RobustGetupActor).parameters:
        raise ValueError("The explicit, backwards-compatible knee-request actor extension is not installed")
    original = torch.load(args.source, map_location="cpu", weights_only=False)
    previous_source_sha = original["infos"]["source_sha256"][ACTOR_SOURCE]
    if digest(args.source_root / ACTOR_SOURCE) != previous_source_sha:
        raise ValueError("Archived actor source does not match the source checkpoint")
    load_mean(args.source, migration_source_root=args.source_root)
    state = deepcopy(original)
    info = state["infos"]
    task = info["settings"]
    if not task.get("smooth_knee_targets") or task.get("recovery_leg_step") != .3:
        raise ValueError("This diagnostic preserves the existing smooth-knee/common-step.3 coordinates")
    previous = task.get("recovery_knee_step", .15)
    task["recovery_knee_step"] = args.step
    info["source_sha256"][ACTOR_SOURCE] = digest(ROOT / ACTOR_SOURCE)
    revision = dict(source=str(args.source.resolve()), source_sha256=digest(args.source),
        archived_source_root=str(args.source_root.resolve()),
        previous_actor_source_sha256=previous_source_sha,
        actor_source_sha256=info["source_sha256"][ACTOR_SOURCE],
        previous_knee_step=previous, knee_step=args.step,
        scope="primary fallen-recovery smooth knee target only; orbit-v2 upright-entry path keeps.15",
        common_step=.3, new_optimizer_updates=0, new_tensors=0,
        changes="Explicit measured-relative knee request lead, not a prescribed posture or trajectory",
        hard_limit_or_torque_changes=False, diagnostic_only=True)
    info["recovery_knee_request_migration"] = revision
    info["hardware_ready"] = False
    checks = {k: recursive_equal(v, state[k]) for k, v in original.items() if k != "infos"}
    checks.update({k: recursive_equal(v, info[k]) for k, v in original["infos"].items()
                   if k not in ("settings", "source_sha256", "recovery_knee_request_migration", "hardware_ready")})
    setting_changes = [k for k in set(original["infos"]["settings"]) | set(task)
                       if not recursive_equal(original["infos"]["settings"].get(k), task.get(k))]
    source_changes = [k for k, v in original["infos"]["source_sha256"].items()
                      if info["source_sha256"][k] != v]
    if (not all(checks.values()) or setting_changes != ["recovery_knee_step"]
            or source_changes != [ACTOR_SOURCE]):
        raise ValueError("Knee-request migration altered unrelated learned, optimizer, RNG or contract state")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    load_mean(args.output)
    write_json(args.output.with_suffix(".json"), dict(schema="fallen-knee-request-migration-v1",
        **revision, checkpoint_sha256=digest(args.output), checks=checks,
        setting_changes=setting_changes, source_changes=source_changes,
        completed_updates=info["completed_updates"],
        experiment_consumed_at_preparation=args.experiment_consumed,
        qualification=None, hardware_ready=False))
    print(args.output)


if __name__ == "__main__":
    main()

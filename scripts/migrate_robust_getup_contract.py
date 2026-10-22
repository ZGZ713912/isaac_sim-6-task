#!/usr/bin/env python3
"""Bind a zero-PPO feedback correction and an authenticated quiet-head restore."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    from export_robust_getup import load_mean
    from wheeled_algo.robust_getup import SensorConditionedMean
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quiet-reference", type=Path)
    parser.add_argument("--restore-quiet-contract", action="store_true")
    parser.add_argument("--stance-aware-recovery", action="store_true")
    parser.add_argument("--shape-aware-recovery", action="store_true")
    parser.add_argument("--quiet-high-common-alpha", type=float)
    parser.add_argument("--quiet-high-leg-prediction-ms", type=float)
    parser.add_argument("--motion-residual-fraction", type=float,
                        help="Explicit zero-PPO revision of straight/spin compensation; mixed arc authority stays intact")
    parser.add_argument("--fast-forward-reference", action="store_true",
                        help="Use authenticated original feedback for high straight-forward commands")
    parser.add_argument("--fast-forward-reference-range", type=float, nargs=2, metavar=("START", "FULL"),
                        help="Explicit command-speed transition in m/s; old checkpoints retain 1.5..2")
    parser.add_argument("--fast-forward-reference-all-yaws", action="store_true",
                        help="Retain original high-forward feedback during turns; old checkpoints retain .25 rad/s yaw fade")
    parser.add_argument("--reason", default="restore paired continuous-feedback PD equivalence across encoder reporting seams")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shape_aware_recovery and not args.stance_aware_recovery:
        parser.error("Shape-aware capture requires --stance-aware-recovery")
    if args.quiet_high_common_alpha is not None and not 0. < args.quiet_high_common_alpha <= 1.:
        parser.error("High-height common-angle interpolation must be in (0,1]")
    if args.quiet_high_leg_prediction_ms is not None and not 0. <= args.quiet_high_leg_prediction_ms <= 10.:
        parser.error("High-height leg prediction must be 0..10 ms")
    if args.motion_residual_fraction is not None and not 0. <= args.motion_residual_fraction <= 1.:
        parser.error("Motion residual fraction must be 0..1")
    if args.fast_forward_reference_range is not None and not 0. <= args.fast_forward_reference_range[0] < args.fast_forward_reference_range[1] <= 3.:
        parser.error("Forward reference transition must be ordered within 0..3 m/s")
    mean, old_info = load_mean(args.checkpoint, migration_source_root=args.checkpoint.parent / "source")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    original_actor = {k: v.clone() for k, v in state["actor_state_dict"].items()}
    info = deepcopy(old_info)
    info["settings"].update(version="robust-v2.9-feedback-winding", paired_feedback_winding=True)
    info["settings"]["stance_aware_recovery"] = args.stance_aware_recovery
    info["settings"]["recovery_shape_aware"] = args.shape_aware_recovery
    if args.shape_aware_recovery:
        info["settings"]["version"] = "robust-v2.10-knee-shape"
    if args.quiet_high_common_alpha is not None:
        info["settings"].update(version="robust-v2.12-high-common-filter",
                                quiet_high_common_alpha=args.quiet_high_common_alpha)
    if args.quiet_high_leg_prediction_ms is not None:
        info["settings"].update(version="robust-v2.13-high-leg-prediction",
                                quiet_high_leg_prediction_s=args.quiet_high_leg_prediction_ms * .001)
    if args.motion_residual_fraction is not None:
        info["settings"].update(version="robust-v2.15-motion-residual",
                                motion_residual_fraction=args.motion_residual_fraction)
    if args.fast_forward_reference:
        info["settings"].update(version="robust-v2.16-reference-forward", fast_forward_reference=True)
    if args.fast_forward_reference_range is not None:
        if not info["settings"].get("fast_forward_reference", False):
            parser.error("Reference transition requires the reference feedback path")
        info["settings"].update(version="robust-v2.17-reference-range",
                                fast_forward_reference_range=args.fast_forward_reference_range)
    if args.fast_forward_reference_all_yaws:
        if not info["settings"].get("fast_forward_reference", False):
            parser.error("All-yaw reference requires the reference feedback path")
        info["settings"].update(version="robust-v2.18-reference-forward-turns",
                                fast_forward_reference_yaw_fade=None)
    revision = dict(source_checkpoint=str(args.checkpoint.resolve()), source_sha256=digest(args.checkpoint),
                    optimizer_updates=0, reason=args.reason)
    if args.quiet_reference:
        reference, ref_info = load_mean(args.quiet_reference, migration_source_root=args.quiet_reference.parent / "source")
        for key in ("quiet_actor_alpha", "quiet_velocity_feature_scale", "quiet_wheel_bound"):
            if args.restore_quiet_contract:
                info["settings"][key] = ref_info["settings"].get(key, .5 if key == "quiet_wheel_bound" else None)
            elif info["settings"].get(key) != ref_info["settings"].get(key):
                raise ValueError("Quiet reference coordinates differ: " + key)
        for key, value in reference.state_dict().items():
            if key.startswith("quiet.") or (args.restore_quiet_contract and key in ("quiet_alpha", "quiet_bound")):
                state["actor_state_dict"]["mlp.mean." + key] = value.clone()
        # Native actor has only this SensorConditionedMean as trainable state.
        names = [name for name, p in SensorConditionedMean(mean).named_parameters() if p.requires_grad]
        group = next(g for g in state["optimizer_state_dict"]["param_groups"] if g["name"] == "actor")
        if len(names) != len(group["params"]):
            raise ValueError("Cannot authenticate quiet-head optimizer parameter mapping")
        for name, identifier in zip(names, group["params"], strict=True):
            if name.startswith("mean.quiet."):
                state["optimizer_state_dict"]["state"].pop(identifier, None)
        revision["quiet_reference"] = dict(path=str(args.quiet_reference.resolve()), sha256=digest(args.quiet_reference),
                                            restored_feature_and_bound_contract=args.restore_quiet_contract,
                                            optimizer="only quiet-head moments cleared")
    revision["stance_aware_recovery"] = args.stance_aware_recovery
    revision["recovery_shape_aware"] = args.shape_aware_recovery
    revision["quiet_high_common_alpha"] = info["settings"].get("quiet_high_common_alpha")
    revision["quiet_high_leg_prediction_s"] = info["settings"].get("quiet_high_leg_prediction_s", 0.)
    revision["motion_residual_fraction"] = info["settings"].get("motion_residual_fraction", 0.)
    revision["fast_forward_reference"] = info["settings"].get("fast_forward_reference", False)
    revision["fast_forward_reference_range"] = info["settings"].get("fast_forward_reference_range", [1.5, 2.])
    revision["fast_forward_reference_yaw_fade"] = info["settings"].get("fast_forward_reference_yaw_fade", .25)
    allowed = (("mlp.mean.quiet.", "mlp.mean.quiet_alpha", "mlp.mean.quiet_bound") if args.quiet_reference else ())
    if any(not torch.equal(value, state["actor_state_dict"][key])
           for key, value in original_actor.items() if not key.startswith(allowed)):
        raise ValueError("Migration changed unrelated actor parameters")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in sorted(set((ROOT / "src/wheeled_algo").glob("*.py"))
                       | set((ROOT / "src/wheeled_tasks/chassis").glob("*.py"))
                       | {Path(__file__), ROOT / "scripts/train_robust_getup.py"}):
        relative = path.resolve().relative_to(ROOT)
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        sources[str(relative)] = digest(path)
    task = json.loads((args.checkpoint.parent / "task_contract.json").read_text())
    task["settings"] = info["settings"]
    task["control_revision"] = revision
    task_file = args.output / "task_contract.json"
    task_file.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n")
    info.update(source_sha256=sources, task_contract_sha256=digest(task_file),
                migrated_from_sha256=digest(args.checkpoint), control_revision=revision, hardware_ready=False)
    state["infos"] = info
    target = args.output / "model_migrated.pt"
    torch.save(state, target)
    actual, _ = load_mean(target)
    expected = {k.removeprefix("mlp.mean."): v for k, v in state["actor_state_dict"].items() if k.startswith("mlp.mean.")}
    if any(not torch.equal(value, actual.state_dict()[key]) for key, value in expected.items()):
        raise ValueError("Saved actor differs from the intended migration")
    receipt = dict(schema="robust-contract-migration-v1", **revision, checkpoint=str(target.resolve()),
        checkpoint_sha256=digest(target), completed_updates=info["completed_updates"],
        actor_other_than_quiet_unchanged=True, feedback_winding="paired_circular_feedback_v1", hardware_tested=False)
    (args.output / "migration.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

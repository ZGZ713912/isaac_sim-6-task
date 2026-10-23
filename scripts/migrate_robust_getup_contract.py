#!/usr/bin/env python3
"""Bind explicit zero-PPO corrections and authenticated quiet expert restores."""
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
    parser.add_argument("--all-height-leg-prediction", action="store_true",
                        help="Explicitly apply the existing positive prediction duration at every quiet standing height")
    parser.add_argument("--recovery-leg-step", type=float,
                        help="Diagnostic common-leg increment in raw action per 5 ms; .2 means .05 rad, not a scripted angle")
    parser.add_argument("--quiet-height-experts", action="store_true",
                        help="Explicitly clone the quiet head into three continuous height experts, including Adam moments")
    parser.add_argument("--native-height-anchors", action="store_true",
                        help="Match height expert anchors to native float32(command)*5; requires existing height experts")
    parser.add_argument("--quiet-exploration-fraction", type=float,
                        help="Explicit training-distribution revision, .05...35; deterministic actor is unchanged")
    parser.add_argument("--quiet-wheel-exploration-fraction", type=float,
                        help="Explicit wheel-only quiet Gaussian fraction .05...35; old checkpoints keep the shared fraction")
    parser.add_argument("--low-quiet-reference", type=Path,
                        help="Optional authenticated low-height quiet head; requires --quiet-height-experts")
    parser.add_argument("--high-quiet-reference", type=Path,
                        help="Restore only an existing high-height expert and its named Adam moments; retain recovery")
    parser.add_argument("--quiet-velocity-feature-gain", type=float,
                        help="Explicit1...30 post-tanh feature conditioning; inverse-scale quiet weights and named Adam moments")
    parser.add_argument("--quiet-velocity-feedback", action="store_true",
                        help="Append three zero-initialized feedback paths; preserve all previous tensors/moments/RNG")
    parser.add_argument("--quiet-velocity-feedback-mode", choices=("all", "wheels", "wheel_memory", "coupled_state", "rolling_velocity", "standing_displacement"),
                        help="Explicit new path: legacy velocity modes or bounded full-sensor common-leg/wheel corrections")
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
    if args.recovery_leg_step is not None and not .05 <= args.recovery_leg_step <= .3:
        parser.error("Recovery common-leg increment must be within .05...3 raw action per policy tick")
    if args.quiet_exploration_fraction is not None and not .05 <= args.quiet_exploration_fraction <= .35:
        parser.error("Quiet exploration fraction must be within .05...35")
    if args.quiet_wheel_exploration_fraction is not None and not .05 <= args.quiet_wheel_exploration_fraction <= .35:
        parser.error("Quiet wheel exploration fraction must be within .05...35")
    if args.low_quiet_reference and not args.quiet_height_experts:
        parser.error("Low-height reference requires explicit quiet height experts")
    if args.quiet_height_experts and args.quiet_reference:
        parser.error("Migrate height experts and restore a quiet reference in separate explicit revisions")
    if args.high_quiet_reference and (args.quiet_reference or args.quiet_height_experts):
        parser.error("Restore an existing high-height expert in a separate explicit revision")
    if args.quiet_velocity_feature_gain is not None:
        if not 1. <= args.quiet_velocity_feature_gain <= 30.:
            parser.error("Quiet velocity feature gain must be within1...30")
        if args.quiet_reference or args.high_quiet_reference or args.quiet_height_experts:
            parser.error("Condition existing quiet experts in a separate explicit revision")
    if args.quiet_velocity_feedback and (args.quiet_reference or args.high_quiet_reference
            or args.quiet_height_experts or args.quiet_velocity_feature_gain is not None):
        parser.error("Add zero velocity feedback to existing experts in a separate explicit revision")
    if args.quiet_velocity_feedback_mode is not None and not args.quiet_velocity_feedback:
        parser.error("Velocity mode requires explicit --quiet-velocity-feedback migration")
    if args.motion_residual_fraction is not None and not 0. <= args.motion_residual_fraction <= 1.:
        parser.error("Motion residual fraction must be 0..1")
    if args.fast_forward_reference_range is not None and not 0. <= args.fast_forward_reference_range[0] < args.fast_forward_reference_range[1] <= 3.:
        parser.error("Forward reference transition must be ordered within 0..3 m/s")
    mean, old_info = load_mean(args.checkpoint, migration_source_root=args.checkpoint.parent / "source")
    if old_info["settings"].get("quiet_velocity_feedback", False) and (args.quiet_reference
            or args.high_quiet_reference or args.quiet_height_experts):
        parser.error("Restoring quiet MLPs independently of existing learned velocity feedback is unsupported")
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
    if args.all_height_leg_prediction:
        if info["settings"].get("quiet_leg_prediction_all_heights", False):
            parser.error("All-height leg prediction is already enabled")
        if not info["settings"].get("quiet_high_leg_prediction_s", 0.):
            parser.error("All-height leg prediction requires a positive existing or explicit prediction duration")
        info["settings"].update(version="robust-v2.31-all-height-leg-prediction",
                                quiet_leg_prediction_all_heights=True)
    if args.recovery_leg_step is not None:
        info["settings"].update(version="robust-v2.19-recovery-common-step",
                                recovery_leg_step=args.recovery_leg_step)
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
    if args.quiet_velocity_feedback:
        if not old_info["settings"].get("quiet_height_experts", False) or old_info["settings"].get("quiet_velocity_feedback", False):
            parser.error("Add velocity feedback once to an existing three-height-expert checkpoint")
        names = [name for name, p in SensorConditionedMean(mean).named_parameters() if p.requires_grad]
        group = next(g for g in state["optimizer_state_dict"]["param_groups"] if g["name"] == "actor")
        if len(names) != len(group["params"]):
            raise ValueError("Cannot authenticate velocity-feedback actor optimizer order")
        next_id = max(p for g in state["optimizer_state_dict"]["param_groups"] for p in g["params"]) + 1
        mode = args.quiet_velocity_feedback_mode or "all"
        size = 4 if mode in ("coupled_state", "rolling_velocity", "standing_displacement") else 6 if mode == "all" else 2
        inputs = 2 if mode == "standing_displacement" else 1 if mode == "rolling_velocity" else 16 if mode == "coupled_state" else 4 if mode == "wheel_memory" else size
        for name in ("mid", "low", "high"):
            key = "mlp.mean.quiet_velocity." + name + ".weight"
            state["actor_state_dict"][key] = state["actor_state_dict"]["mlp.mean.quiet.0.weight"].new_zeros((size, inputs))
            group["params"].append(next_id)
            next_id += 1
            if mode == "coupled_state":
                state["actor_state_dict"][key.removesuffix("weight") + "bias"] = state["actor_state_dict"][key].new_zeros(size)
                group["params"].append(next_id)
                next_id += 1
        if mode == "rolling_velocity":
            from wheeled_algo.rolling_velocity import frozen_rolling_estimator, TABLE_RELATIVE, TABLE_SHA256
            for name, value in frozen_rolling_estimator().state_dict().items():
                state["actor_state_dict"]["mlp.mean.rolling_velocity." + name] = value.clone()
            revision["rolling_geometry"] = dict(file=TABLE_RELATIVE, sha256=TABLE_SHA256, trainable=False)
        if mode == "standing_displacement":
            from wheeled_algo.standing_displacement import observation_contract
            info["settings"]["standing_position_observation"] = observation_contract()
            revision["observation_abi_changed"] = True
            revision["rmcs_preprocessor_installed"] = False
        info["settings"].update(version=({"all": "robust-v2.27-quiet-velocity-feedback",
                                         "wheels": "robust-v2.28-quiet-wheel-velocity",
                                         "wheel_memory": "robust-v2.30-quiet-wheel-memory",
                                         "coupled_state": "robust-v2.31-quiet-coupled-state",
                                         "rolling_velocity": "robust-v2.32-quiet-rolling-velocity",
                                         "standing_displacement": "robust-v2.33-standing-displacement"}[mode]),
                                quiet_velocity_feedback=True, quiet_velocity_feedback_mode=mode)
        revision["quiet_velocity_feedback"] = dict(new_parameters=3 * size * (inputs + int(mode == "coupled_state")), mode=mode, initial_weights="zero",
            input=({"all": "tanh(physical six encoder velocities in rad/s)",
                    "wheels": "tanh(physical two wheel encoder velocities in rad/s)",
                    "wheel_memory": "tanh(two wheel encoder velocities) and tanh(previous two wheel raw commands times10); original35D fields only",
                    "coupled_state": "tanh(physical gyro), gravity, tanh(leg positions), tanh(six physical velocities); original35D only, no previous actions",
                    "rolling_velocity": "tanh(50 * model-derived body forward velocity in m/s); existing35D encoders/IMU only",
                    "standing_displacement": "20 * filtered relative encoder rolling coordinate in metres, slots14:16; old networks force these slots to zero"}[mode]),
            output=("bounded additive common-leg and wheel corrections after the old quiet tanh" if mode in ("coupled_state", "rolling_velocity", "standing_displacement")
                    else "six axis logits" if mode == "all" else "two wheel logits; four leg logits exactly unchanged"),
            bias=mode == "coupled_state",
            optimizer="all old moments retained; only new parameters start with empty Adam state",
            deterministic_function="preserved; no random initialization draws")
    if args.quiet_velocity_feature_gain is not None:
        if not old_info["settings"].get("quiet_height_experts", False):
            raise ValueError("Quiet velocity conditioning requires existing three-height experts")
        old_gain = old_info["settings"].get("quiet_velocity_feature_gain", 1.)
        ratio = args.quiet_velocity_feature_gain / old_gain
        names = [name for name, p in SensorConditionedMean(mean).named_parameters() if p.requires_grad]
        group = next(g for g in state["optimizer_state_dict"]["param_groups"] if g["name"] == "actor")
        if len(names) != len(group["params"]):
            raise ValueError("Cannot authenticate conditioned quiet actor optimizer order")
        ids = dict(zip(names, group["params"], strict=True))
        for head in ("quiet", "quiet_low", "quiet_high"):
            name = "mean." + head + ".0.weight"
            weight = state["actor_state_dict"]["mlp." + name]
            weight[:, 16:22].div_(ratio)
            moments = state["optimizer_state_dict"]["state"].get(ids[name], {})
            for key, factor in (("exp_avg", ratio), ("exp_avg_sq", ratio ** 2), ("max_exp_avg_sq", ratio ** 2)):
                if key in moments:
                    if moments[key].shape != weight.shape:
                        raise ValueError("Conditioned quiet optimizer tensor shape differs")
                    moments[key][:, 16:22].mul_(factor)
        info["settings"].update(version="robust-v2.26-quiet-velocity-conditioning",
                                quiet_velocity_feature_gain=args.quiet_velocity_feature_gain)
        revision["quiet_velocity_conditioning"] = dict(old_gain=old_gain,
            new_gain=args.quiet_velocity_feature_gain, weight_ratio=1. / ratio,
            first_moment_ratio=ratio, second_moment_ratio=ratio ** 2,
            scope="three quiet first-layer velocity columns only; other tensors, Adam steps and RNG retained",
            deterministic_function="preserved up to float32 rounding, verify on real observations")
    if args.quiet_exploration_fraction is not None:
        info["settings"].update(version="robust-v2.24-quiet-exploration",
                                quiet_exploration_fraction=args.quiet_exploration_fraction)
        revision["quiet_exploration_fraction"] = args.quiet_exploration_fraction
        revision["deterministic_mean_unchanged"] = True
    if args.quiet_wheel_exploration_fraction is not None:
        info["settings"].update(version="robust-v2.29-axis-quiet-exploration",
                                quiet_wheel_exploration_fraction=args.quiet_wheel_exploration_fraction)
        revision["quiet_wheel_exploration_fraction"] = args.quiet_wheel_exploration_fraction
        revision["deterministic_mean_unchanged"] = True
    if args.native_height_anchors:
        if not old_info["settings"].get("quiet_height_experts", False):
            parser.error("Native height anchors require an existing height-expert checkpoint")
        info["settings"].update(version="robust-v2.23-native-height-anchors", quiet_native_height_anchors=True)
        revision["quiet_native_height_anchors"] = True
    if args.quiet_height_experts:
        if old_info["settings"].get("quiet_height_experts", False):
            parser.error("Quiet height experts are already present")
        names = [name for name, p in SensorConditionedMean(mean).named_parameters() if p.requires_grad]
        group = next(g for g in state["optimizer_state_dict"]["param_groups"] if g["name"] == "actor")
        if len(names) != len(group["params"]):
            raise ValueError("Cannot authenticate original actor optimizer order")
        original_ids = dict(zip(names, group["params"], strict=True))
        low_state, low_ids = state, original_ids
        if args.low_quiet_reference:
            low_mean, low_info = load_mean(args.low_quiet_reference,
                migration_source_root=args.low_quiet_reference.parent / "source")
            if low_info["settings"].get("quiet_velocity_feedback", False):
                raise ValueError("Copying a quiet MLP without its learned velocity feedback is unsupported")
            low_state = torch.load(args.low_quiet_reference, map_location="cpu", weights_only=False)
            low_names = [name for name, p in SensorConditionedMean(low_mean).named_parameters() if p.requires_grad]
            low_group = next(g for g in low_state["optimizer_state_dict"]["param_groups"] if g["name"] == "actor")
            if len(low_names) != len(low_group["params"]):
                raise ValueError("Cannot authenticate low-reference actor optimizer order")
            low_ids = dict(zip(low_names, low_group["params"], strict=True))
            for key in ("base_checkpoint_sha256", "bank_sha256"):
                if low_info[key] != old_info[key]:
                    raise ValueError("Low-height reference mechanism or locomotion parent differs")
            for key in ("quiet_actor_alpha", "quiet_velocity_feature_scale", "quiet_wheel_bound",
                        "quiet_high_leg_prediction_s", "quiet_leg_prediction_all_heights", "recovery_shape_aware", "fast_forward_reference_range",
                        "fast_forward_reference_yaw_fade"):
                if low_info["settings"].get(key) != old_info["settings"].get(key):
                    raise ValueError("Low-height reference controller contract differs: " + key)
            if any(not torch.equal(value, low_state["actor_state_dict"][key])
                   for key, value in original_actor.items() if not key.startswith("mlp.mean.quiet.")):
                raise ValueError("Low-height reference changed a nonquiet actor tensor")
            revision["low_quiet_reference"] = dict(path=str(args.low_quiet_reference.resolve()),
                sha256=digest(args.low_quiet_reference))
        next_id = max(p for g in state["optimizer_state_dict"]["param_groups"] for p in g["params"]) + 1
        for target_name, source_state, ids in (("quiet_low", low_state, low_ids),
                                               ("quiet_high", state, original_ids)):
            for name in names:
                if not name.startswith("mean.quiet."):
                    continue
                target = name.replace("mean.quiet.", "mean." + target_name + ".", 1)
                state["actor_state_dict"]["mlp." + target] = source_state["actor_state_dict"]["mlp." + name].clone()
                if ids[name] in source_state["optimizer_state_dict"]["state"]:
                    state["optimizer_state_dict"]["state"][next_id] = deepcopy(
                        source_state["optimizer_state_dict"]["state"][ids[name]])
                group["params"].append(next_id)
                next_id += 1
        info["settings"].update(version="robust-v2.20-quiet-height-experts", quiet_height_experts=True)
        revision["quiet_height_experts"] = dict(anchors_m=[.23, .305, .42],
            interpolation="continuous existing commanded height; exact expert endpoints",
            optimizer="old parameter moments retained; low/high moments copied from their authenticated source quiet head")
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
    if args.high_quiet_reference:
        reference, ref_info = load_mean(args.high_quiet_reference,
            migration_source_root=args.high_quiet_reference.parent / "source")
        if ref_info["settings"].get("quiet_velocity_feedback", False):
            raise ValueError("Copying a high quiet MLP without its learned velocity feedback is unsupported")
        if not info["settings"].get("quiet_height_experts") or not ref_info["settings"].get("quiet_height_experts"):
            raise ValueError("High-height restore requires existing height experts in both checkpoints")
        for key in ("base_checkpoint_sha256", "bank_sha256"):
            if info[key] != ref_info[key]:
                raise ValueError("High-height reference mechanism or locomotion parent differs: " + key)
        if info["settings"].get("quiet_velocity_feature_gain", 1.) != ref_info["settings"].get("quiet_velocity_feature_gain", 1.):
            raise ValueError("High-height reference controller contract differs: quiet_velocity_feature_gain")
        for key in ("quiet_actor_alpha", "quiet_velocity_feature_scale", "quiet_wheel_bound",
                    "quiet_high_common_alpha", "quiet_high_leg_prediction_s", "quiet_leg_prediction_all_heights", "quiet_native_height_anchors",
                    "leg_action_clip", "wheel_action_clip", "fast_forward_reference",
                    "fast_forward_reference_range", "fast_forward_reference_yaw_fade"):
            if info["settings"].get(key) != ref_info["settings"].get(key):
                raise ValueError("High-height reference controller contract differs: " + key)
        ref_state = torch.load(args.high_quiet_reference, map_location="cpu", weights_only=False)
        id_maps = []
        for model, checkpoint_state in ((mean, state), (reference, ref_state)):
            names = [name for name, p in SensorConditionedMean(model).named_parameters() if p.requires_grad]
            group = next(g for g in checkpoint_state["optimizer_state_dict"]["param_groups"] if g["name"] == "actor")
            if len(names) != len(group["params"]):
                raise ValueError("Cannot authenticate high-height actor optimizer parameter order")
            id_maps.append(dict(zip(names, group["params"], strict=True)))
        restored = []
        for name, target_id in id_maps[0].items():
            if not name.startswith("mean.quiet_high."):
                continue
            state["actor_state_dict"]["mlp." + name] = ref_state["actor_state_dict"]["mlp." + name].clone()
            source_moments = ref_state["optimizer_state_dict"]["state"].get(id_maps[1][name])
            if source_moments is None:
                state["optimizer_state_dict"]["state"].pop(target_id, None)
            else:
                state["optimizer_state_dict"]["state"][target_id] = deepcopy(source_moments)
            restored.append(name)
        if len(restored) != 4:
            raise ValueError("Expected the four authenticated high-height expert tensors")
        info["settings"]["version"] = "robust-v2.25-high-quiet-restore"
        revision["high_quiet_reference"] = dict(path=str(args.high_quiet_reference.resolve()),
            sha256=digest(args.high_quiet_reference), source_clock=ref_info["completed_updates"],
            base_clock_retained=info["completed_updates"], optimizer="only high expert moments copied by parameter name",
            restored_parameters=restored, source_clocks_are_not_added=True)
    revision["stance_aware_recovery"] = args.stance_aware_recovery
    revision["recovery_shape_aware"] = args.shape_aware_recovery
    revision["quiet_high_common_alpha"] = info["settings"].get("quiet_high_common_alpha")
    revision["quiet_high_leg_prediction_s"] = info["settings"].get("quiet_high_leg_prediction_s", 0.)
    revision["quiet_leg_prediction_all_heights"] = info["settings"].get("quiet_leg_prediction_all_heights", False)
    revision["recovery_leg_step"] = info["settings"]["recovery_leg_step"]
    revision["motion_residual_fraction"] = info["settings"].get("motion_residual_fraction", 0.)
    revision["fast_forward_reference"] = info["settings"].get("fast_forward_reference", False)
    revision["fast_forward_reference_range"] = info["settings"].get("fast_forward_reference_range", [1.5, 2.])
    revision["fast_forward_reference_yaw_fade"] = info["settings"].get("fast_forward_reference_yaw_fade", .25)
    allowed = (("mlp.mean.quiet.", "mlp.mean.quiet_alpha", "mlp.mean.quiet_bound") if args.quiet_reference else ())
    if args.high_quiet_reference:
        allowed = ("mlp.mean.quiet_high.",)
    if args.quiet_velocity_feature_gain is not None:
        allowed = tuple("mlp.mean." + name + ".0.weight" for name in ("quiet", "quiet_low", "quiet_high"))
    if any(not torch.equal(value, state["actor_state_dict"][key])
           for key, value in original_actor.items() if not key.startswith(allowed)):
        raise ValueError("Migration changed unrelated actor parameters")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    source_paths = (set((ROOT / "src/wheeled_algo").glob("*.py"))
                    | set((ROOT / "src/wheeled_tasks/chassis").glob("*.py"))
                    | {Path(__file__), ROOT / "scripts/train_robust_getup.py"})
    if info["settings"].get("quiet_velocity_feedback_mode") in ("rolling_velocity", "standing_displacement"):
        from wheeled_algo.rolling_velocity import frozen_table_path
        source_paths.add(frozen_table_path())
    for path in sorted(source_paths):
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

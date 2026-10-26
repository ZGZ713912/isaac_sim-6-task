#!/usr/bin/env python3
"""Clone an authenticated learned entry head without another optimizer update."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

# Non-tensor options read by the authenticated policy factory.
POLICY_SETTINGS = (
    "canonical_recovery", "motion_residual_fraction", "cyclic_leg_targets", "leg_action_clip",
    "wheel_action_clip", "recovery_leg_step", "smooth_knee_targets", "recovery_knee_center",
    "recovery_knee_step", "quiet_actor_alpha", "quiet_velocity_feature_scale", "quiet_velocity_feature_gain",
    "quiet_velocity_feedback", "quiet_velocity_feedback_mode", "quiet_wheel_bound", "stance_aware_recovery",
    "recovery_shape_aware", "quiet_high_common_alpha", "quiet_high_leg_prediction_s",
    "quiet_leg_prediction_all_heights", "quiet_height_experts", "quiet_native_height_anchors",
    "fast_forward_reference", "fast_forward_reference_range", "fast_forward_reference_yaw_fade",
    "quiet_output_wheel_alpha", "standing_position_observation", "standing_position_filter_seconds",
    "standing_position_reference", "gap_entry_handoff_version", "gap_upright_target_limit_version",
    "upright_entry_recovery", "recovery_previous_common_version", "gap_final_target_winding_version",
)


def migrate(source, reference, output, experiment_consumed):
    import torch
    from export_robust_getup import load_mean
    from train_robust_getup import digest, write_json
    load_mean(source)
    load_mean(reference)
    original = torch.load(source, map_location="cpu", weights_only=False)
    teacher = torch.load(reference, map_location="cpu", weights_only=False)
    settings, reference_settings = original["infos"]["settings"], teacher["infos"]["settings"]
    if (settings.get("gap_entry_handoff_version") != "orbit-v2"
            or settings.get("gap_upright_target_limit_version") != "error-v1"
            or settings.get("gap_entry_recovery_split_version") is not None):
        raise ValueError("Migrate an unsplit orbit-v2 error-v1 mean exactly once")
    if any(settings.get(k) != reference_settings.get(k) for k in POLICY_SETTINGS):
        raise ValueError("Entry and fallen sources use different policy coordinates or options")
    if original["infos"]["base_checkpoint_sha256"] != teacher["infos"]["base_checkpoint_sha256"]:
        raise ValueError("Entry reference has a different locomotion parent")
    if original["actor_state_dict"].keys() != teacher["actor_state_dict"].keys():
        raise ValueError("Entry reference actor structure differs")
    if any(not torch.equal(v, teacher["actor_state_dict"][k]) for k, v in original["actor_state_dict"].items()
           if not k.startswith("mlp.mean.recovery.")):
        raise ValueError("This counterfactual requires identical non-recovery weights and std")
    if output.exists():
        raise ValueError("Never overwrite a checkpoint")
    state = deepcopy(original)
    prefix = "mlp.mean.recovery."
    for key, value in teacher["actor_state_dict"].items():
        if key.startswith(prefix):
            state["actor_state_dict"]["mlp.mean.entry_recovery." + key.removeprefix(prefix)] = value.clone()
    state["infos"]["settings"]["gap_entry_recovery_split_version"] = "frozen-v1"
    module = "src/wheeled_algo/gap_entry_preservation.py"
    state["infos"]["source_sha256"][module] = digest(ROOT / module)
    state["infos"]["entry_recovery_preservation"] = dict(
        schema="gap-entry-recovery-preservation-v1", checkpoint=str(reference.resolve()), sha256=digest(reference),
        entry_learning_clock=teacher["infos"]["completed_updates"],
        fallen_checkpoint=str(source.resolve()), fallen_sha256=digest(source),
        fallen_learning_clock=original["infos"]["completed_updates"],
        role="frozen learned upright-entry network; same measured continuous authority", optimizer_updates=0,
        optimizer="Existing groups/moments unchanged; the copied entry tensors are frozen and have no Adam slots")
    state["infos"]["hardware_ready"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    mean, _ = load_mean(output)
    assert not any(p.requires_grad for p in mean.entry_recovery.parameters())
    write_json(output.with_suffix(".json"), dict(
        schema="gap-entry-head-migration-receipt-v1", checkpoint=str(output), sha256=digest(output),
        source_sha256=digest(source), reference_sha256=digest(reference),
        original_actor_tensors_exact=True, new_frozen_entry_parameters=sum(p.numel() for p in mean.entry_recovery.parameters()),
        critic_optimizer_rng_clock_unchanged=True, new_PPO=0, experiment_consumed=experiment_consumed,
        learning_clock=state["infos"]["completed_updates"],
        neutral_copy=(digest(source) == digest(reference)), qualification=None, hardware_ready=False))
    return mean


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--entry-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-consumed", type=int, required=True)
    args = parser.parse_args()
    migrate(args.source, args.entry_reference, args.output, args.experiment_consumed)
    print(args.output)

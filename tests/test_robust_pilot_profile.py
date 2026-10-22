"""A reward-only standing pilot must preserve the resumed recovery function."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


@pytest.mark.parametrize("feedback_mode", ["wheels", "wheel_memory"])
def test_standing_reward_revision_preserves_recovery_authority_flags(tmp_path, feedback_mode):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("robust_profile_training", root / "scripts/train_robust_getup.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    checkpoint = tmp_path / "prior.pt"
    torch.save({"infos": {"settings": {"stance_aware_recovery": True, "recovery_shape_aware": True,
                                        "leg_action_clip": 16., "motion_residual_fraction": .25,
                                        "quiet_high_leg_prediction_s": .005, "fast_forward_reference": True,
                                        "fast_forward_reference_range": [1., 1.5],
                                        "fast_forward_reference_yaw_fade": None,
                                        "quiet_height_experts": True,
                                        "quiet_native_height_anchors": True,
                                        "quiet_exploration_fraction": .1,
                                        "quiet_wheel_exploration_fraction": .35,
                                        "quiet_velocity_feature_gain": 10.,
                                        "quiet_velocity_feedback": True,
                                        "quiet_velocity_feedback_mode": feedback_mode,
                                        "quiet_critic_anchor": True, "quiet_spectral_cost": True,
                                        "standing_support_training": True, "standing_drift_weight": 2.,
                                        "standing_drift_speed_scale_m_s": .01, "quiet_sway_m": .05,
                                        "standing_early_position_weight": 6., "quiet_position_weight": 0.,
                                        "native_origin_mode": "isolated_overlap_v1"}}}, checkpoint)
    bank = tmp_path / "bank.json"
    bank.write_text("{}")
    args = SimpleNamespace(output=tmp_path / "run", bank=bank, resume=checkpoint, no_domains=False,
        curriculum_updates=128, quiet_leg_alpha=.25, quiet_wheel_alpha=.65, domain_scale=.35,
        quiet_only=True, standing_support_training=True, num_envs=32, rollout_steps=128,
        mode="train", friction="high", friction_min=.3, seed=123)
    _, task = training.prepare(args)
    assert task["stance_aware_recovery"] and task["recovery_shape_aware"]
    assert task["quiet_high_leg_prediction_s"] == .005 and task["fast_forward_reference"]
    assert task["fast_forward_reference_range"] == [1., 1.5]
    assert task["fast_forward_reference_yaw_fade"] is None
    assert task["native_origin_mode"] == "isolated_overlap_v1"
    assert task["quiet_height_experts"]
    assert task["quiet_native_height_anchors"]
    assert task["quiet_exploration_fraction"] == .1
    assert task["quiet_wheel_exploration_fraction"] == .35
    assert task["quiet_velocity_feature_gain"] == 10.
    assert task["quiet_velocity_feedback"]
    assert task["quiet_velocity_feedback_mode"] == feedback_mode
    assert json.loads((args.output / "config/contract.json").read_text())["full_rl_isolated_origins"]
    assert task["standing_support_training"] and task["quiet_critic_anchor"]
    assert task["quiet_actor_alpha"] == [.25] * 4 + [.65] * 2
    assert task["sensor_delay_policy_ticks"] == [0, 1]
    assert task["quiet_height_probabilities"] == [.25, .25, .5]
    assert task["quiet_position_weight"] == 2.
    args.early_standing_position = True
    _, revised = training.prepare(args)
    assert revised["quiet_position_weight"] == 0.
    assert revised["standing_early_position_weight"] == 6.
    assert revised["standing_drift_weight"] == 0.
    args.standing_drift_objective = True
    _, drift = training.prepare(args)
    assert drift["standing_drift_weight"] == 2.
    assert drift["standing_early_position_weight"] == 6.
    assert drift["quiet_sway_m"] == .05
    assert drift["quiet_high_leg_prediction_s"] == .005
    assert drift["fast_forward_reference_yaw_fade"] is None
    assert drift["standing_drift_speed_scale_m_s"] == .02
    args.standing_drift_speed_scale = .01
    _, fine = training.prepare(args)
    assert fine["standing_drift_speed_scale_m_s"] == .01
    assert fine["standing_drift_weight"] == drift["standing_drift_weight"]
    assert fine["quiet_sway_m"] == drift["quiet_sway_m"]
    args.high_standing_only = True
    _, high = training.prepare(args)
    assert high["quiet_height_probabilities"] == [0., 0., 1.]
    args.friction_min = args.friction_max = .3
    cfg, fixed = training.prepare(args)
    assert fixed["active_friction_range"] == [.3, .3]
    from wheeled_tasks.chassis.contact_material import ContactMaterialDomain
    material = ContactMaterialDomain(4, cfg["contact_material"], seed=123)
    summary = material.summary()
    assert summary["sampled_ground_static_min_max"] == [float(torch.tensor(.3))] * 2
    assert summary["effective_wheel_static_min_max"] == [float(torch.tensor(.2))] * 2
    args.quiet_only = args.standing_support_training = args.early_standing_position = False
    args.standing_drift_objective = args.high_standing_only = False
    args.recovery_only = args.side_recovery_focus = True
    _, side = training.prepare(args)
    assert side["recovery_family_weights"] == [0., .05, .05, .35, .35, .15, .05]
    assert side["motion_fraction"] == .2 and side["motion_push_fraction"] == 1.
    assert side["side_recovery_birth_focus"] and not task["side_recovery_birth_focus"]
    assert side["quiet_height_experts"] and side["quiet_high_leg_prediction_s"] == .005
    assert side["quiet_actor_alpha"] == task["quiet_actor_alpha"]
    assert side["recovery_leg_step"] == task["recovery_leg_step"]
    assert side["reward_gamma"] == task["reward_gamma"]
    assert side["quiet_critic_anchor"] and side["quiet_spectral_cost"] and side["standing_support_training"]
    assert side["standing_drift_weight"] == 2. and side["standing_drift_speed_scale_m_s"] == .01
    assert side["quiet_position_weight"] == 0. and side["standing_early_position_weight"] == 6.
    assert side["quiet_sway_m"] == .05
    assert side["quiet_height_probabilities"] == [.5, .25, .25]
    assert not side["quiet_training"]
    assert side["quiet_exploration_fraction"] == .1
    assert side["quiet_wheel_exploration_fraction"] == .35
    assert side["quiet_velocity_feature_gain"] == 10.
    assert side["quiet_velocity_feedback"]
    assert side["quiet_velocity_feedback_mode"] == feedback_mode

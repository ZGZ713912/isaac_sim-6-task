"""A reward-only standing pilot must preserve the resumed recovery function."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def test_standing_reward_revision_preserves_recovery_authority_flags(tmp_path):
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
    args.friction_min = args.friction_max = .3
    cfg, fixed = training.prepare(args)
    assert fixed["active_friction_range"] == [.3, .3]
    from wheeled_tasks.chassis.contact_material import ContactMaterialDomain
    material = ContactMaterialDomain(4, cfg["contact_material"], seed=123)
    summary = material.summary()
    assert summary["sampled_ground_static_min_max"] == [float(torch.tensor(.3))] * 2
    assert summary["effective_wheel_static_min_max"] == [float(torch.tensor(.2))] * 2

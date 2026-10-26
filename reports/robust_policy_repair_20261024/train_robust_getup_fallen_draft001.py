#!/usr/bin/env python3
"""Bounded native Isaac Sim pilots for one self-righting/locomotion policy.

Uses installed RSL-RL PPO and its TensorBoard logger, with the existing passive
interval mask. No training server, GUI, vehicle process or remote is started.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PARENT = ROOT / "runs/flat_base_20261020/pilot034_directional200_1346/model_001346.pt"
BASE_CONFIG = PARENT.parent / "config/contract.json"


class TrainingStopped(Exception):
    """A requested stop has been sealed after a complete PPO update."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def prepare(args):
    from wheeled_tasks.chassis.getup_ground import require_bounded_world
    if not getattr(args, "allow_large_world_coordinates", False) and not getattr(args, "isolated_origins", False):
        require_bounded_world(args.num_envs)
    from wheeled_tasks.chassis.robust_getup import settings, rigid_body_domain
    cfg = json.loads(BASE_CONFIG.read_text())
    design = json.loads(Path(cfg["design_preflight"]["file"]).read_text())
    row = design["stages"]["speed"]["roles"]["train"]
    task = settings(domains_enabled=not args.no_domains, curriculum_updates=args.curriculum_updates,
                    origin_updates=getattr(args, "curriculum_origin", 0),
                    quiet_actor_alpha=[args.quiet_leg_alpha] * 4 + [args.quiet_wheel_alpha] * 2,
                    quiet_velocity_feature_scale=getattr(args, "quiet_velocity_feature_scale", None),
                    quiet_wheel_bound=getattr(args, "quiet_wheel_bound", .5),
                    domain_scale=getattr(args, "domain_scale", 1.), quiet_frozen=getattr(args, "freeze_quiet", False))
    if getattr(args, "resume", None):
        import torch
        prior = torch.load(args.resume, map_location="cpu", weights_only=False)["infos"]["settings"]
        # A reward/exposure pilot must retain the resumed recovery function,
        # including non-tensor authority flags, rather than reset defaults.
        for key in ("canonical_recovery", "motion_residual_fraction", "cyclic_leg_targets", "leg_action_clip",
                    "recovery_leg_step", "smooth_knee_targets", "recovery_knee_center", "recovery_knee_step", "stance_aware_recovery",
                    "recovery_shape_aware", "quiet_high_common_alpha", "quiet_high_leg_prediction_s", "quiet_leg_prediction_all_heights", "fast_forward_reference",
                    "fast_forward_reference_range", "fast_forward_reference_yaw_fade", "quiet_height_experts",
                    "quiet_native_height_anchors", "quiet_exploration_fraction", "quiet_wheel_exploration_fraction", "quiet_velocity_feature_gain",
                    "quiet_velocity_feedback", "quiet_velocity_feedback_mode", "standing_position_observation",
                    "standing_position_filter_seconds", "standing_position_reference", "quiet_output_wheel_alpha",
                    "gap_entry_handoff_version", "gap_robustness_version", "upright_entry_recovery",
                    "gap_actuator_timing_profile", "gap_severity", "gap_upright_target_limit_version",
                    "gap_recovery_leg_exploration_fraction", "gap_entry_recovery_split_version",
                    "recovery_previous_common_version", "gap_final_target_winding_version",
                    "gap_entry_optimizer_layout", "recovery_support_weight", "gap_entry_encoder_version",
                    "motion_leg_damping_version", "motion_leg_damping_alpha",
                    "quiet_stance_authority_version", "quiet_stance_retained_authority", "quiet_stance_recovery_threshold"):
            if key in prior:
                task[key] = deepcopy(prior[key])
        if prior.get("native_origin_mode") == "isolated_overlap_v1":
            task["native_origin_mode"] = "isolated_overlap_v1"
        if (getattr(args, "recovery_only", False) or getattr(args, "gap_robustness", False)
                or prior.get("gap_robustness_version")):
            # Keep the parent's standing costs and privileged critic channel
            # meanings. This profile changes exposure and trainable weights,
            # while command/height sampling returns to the general motion mix.
            for key in ("quiet_critic_anchor", "quiet_spectral_cost", "standing_support_training",
                        "quiet_position_weight", "standing_early_position_weight", "standing_drift_weight",
                        "standing_drift_speed_scale_m_s", "quiet_sway_m", "quiet_action_rate_cost",
                        "quiet_motor_speed_cost", "quiet_effort_delta_cost"):
                if key in prior:
                    task[key] = deepcopy(prior[key])
    if task.get("gap_entry_handoff_version") not in (None,"orbit-v2"):
        raise ValueError("Unknown gap entry handoff version")
    if task.get("gap_upright_target_limit_version") not in (None,"error-v1"):
        raise ValueError("Unknown upright target limit version")
    if task.get("gap_entry_recovery_split_version") not in (None,"frozen-v1"):
        raise ValueError("Unknown frozen entry head version")
    if task.get("recovery_previous_common_version") not in (None, "circular-v1"):
        raise ValueError("Unknown previous recovery common-angle version")
    if task.get("gap_final_target_winding_version") not in (None, "paired-v1"):
        raise ValueError("Unknown final target winding version")
    if task.get("gap_final_target_winding_version") and task.get("gap_upright_target_limit_version") != "error-v1":
        raise ValueError("Final target winding requires the final upright limiter")
    if task.get("gap_entry_encoder_version") not in (None, "periodic-chart-v1"):
        raise ValueError("Unknown entry encoder version")
    if task.get("motion_leg_damping_version") not in (None, "motion-leg-damping-v1"):
        raise ValueError("Unknown motion leg damping version")
    if task.get("motion_leg_damping_version") and not task.get("gap_entry_encoder_version"):
        raise ValueError("Motion damping requires the authenticated periodic entry actor")
    if task.get("motion_leg_damping_version"):
        from wheeled_algo.motion_target_damping import validate_alpha
        validate_alpha(task.get("motion_leg_damping_alpha"))
    elif "motion_leg_damping_alpha" in task:
        raise ValueError("Motion damping alpha requires an explicit version")
    if getattr(args, "isolated_origins", False):
        task["native_origin_mode"] = "isolated_overlap_v1"
    if getattr(args, "gap_robustness", False) or task.get("gap_robustness_version"):
        from wheeled_tasks.chassis.gap_robustness import apply_profile
        if getattr(args,"gap_actuator_timing",None) is not None:
            task["gap_actuator_timing_profile"] = args.gap_actuator_timing
        severity = (getattr(args,"gap_severity",1.) if getattr(args,"gap_robustness",False)
                    else task["gap_severity"])
        # Domains provide the base task. Explicit tuning modes below must own
        # their reset/command quotas instead of being silently overwritten.
        apply_profile(task, severity)
    if getattr(args, "quiet_only", False):
        task.update(version=task["version"] + "-quiet", quiet_training=True, motion_fraction=.8,
                    quiet_spectral_cost=True, quiet_critic_anchor=True)
    if getattr(args, "standing_support_training", False):
        task.update(version="robust-v2.11-standing-request-support", standing_support_training=True,
                    quiet_position_weight=2., quiet_height_probabilities=[.25, .25, .5])
    if getattr(args, "early_standing_position", False):
        task.update(version="robust-v2.14-early-position", quiet_position_weight=0.,
                    standing_early_position_weight=6.)
    if getattr(args, "high_standing_only", False):
        task.update(version="robust-v2.16-high-standing-only", quiet_height_probabilities=[0., 0., 1.])
    if getattr(args, "standing_drift_objective", False):
        task.update(version="robust-v2.15-standing-drift", standing_drift_weight=2.)
        if getattr(args, "standing_drift_speed_scale", None) is not None:
            task.update(version="robust-v2.22-finer-standing-drift",
                        standing_drift_speed_scale_m_s=args.standing_drift_speed_scale)
    if getattr(args, "quiet_exploration_fraction", None) is not None:
        task.update(version="robust-v2.24-quiet-exploration", quiet_exploration_fraction=args.quiet_exploration_fraction)
    if getattr(args, "quiet_wheel_exploration_fraction", None) is not None:
        task.update(version="robust-v2.32-quiet-wheel-exploration",
                    quiet_wheel_exploration_fraction=args.quiet_wheel_exploration_fraction)
    if getattr(args, "focus_recovery_quiet", False):
        task.update(version=task["version"] + "-recovery-quiet", quiet_training=True, motion_fraction=.6,
                    quiet_spectral_cost=True, quiet_critic_anchor=True)
    if getattr(args, "arc_only", False):
        task.update(version=task["version"] + "-arc", motion_fraction=.8, arc_tuning_fraction=.8)
    if getattr(args, "recovery_only", False):
        task.update(version="robust-v2.10-recovery-capture", stance_aware_recovery=True,
                    recovery_shape_aware=True, motion_push_fraction=1.)
    if getattr(args, "side_recovery_focus", False):
        task.update(version="robust-v2.21-side-recovery-exposure", motion_fraction=.2,
                    recovery_family_weights=[0., .05, .05, .35, .35, .15, .05],
                    side_recovery_birth_focus=True)
    if getattr(args, "motion_recovery_focus", False):
        task.update(version="robust-v2.35-motion-recovery-exposure", motion_fraction=.8,
                    motion_push_fraction=1.)
    if getattr(args, "entry_recovery_only", False):
        if task.get("gap_entry_optimizer_layout") != "append-v1":
            raise ValueError("Entry-only training requires an explicit optimizer-layout migration")
        task["version"] += "-entry-only"
    if getattr(args, "recovery_support_weight", None) is not None:
        task["recovery_support_weight"] = args.recovery_support_weight
    if "recovery_support_weight" in task:
        import math
        weight = task["recovery_support_weight"]
        if not math.isfinite(weight) or not 0. <= weight <= 2.:
            raise ValueError("Recovery support weight must be finite within0..2")
    if getattr(args, "episode_seconds", None) is not None:
        task["episode_seconds"] = args.episode_seconds
    # The fixed physics/control source remains pinned. The task overlay is a
    # separately frozen public contract, not a claim of old-task equivalence.
    changes = dict(target_num_envs=args.num_envs, num_steps_per_env=args.rollout_steps,
        ppo_minibatch_samples=args.num_envs * args.rollout_steps // 4,
        episode_seconds=task["episode_seconds"], record_diagnostics=args.mode != "train",
        full_rl_square_ground=True, flat_floor_width_m=96., flat_half_length_m=48.)
    if task["native_origin_mode"] == "isolated_overlap_v1":
        from wheeled_tasks.chassis.isolated_origins import validate_isolated_origins
        changes["full_rl_isolated_origins"] = True
        validate_isolated_origins(dict(cfg, **changes), args.num_envs)
    if args.friction == "low":
        material = deepcopy(cfg["contact_material"])
        material["ground_static_friction_range"] = [.2, .5]
        material["ground_dynamic_friction_range"] = [.2, .5]
    else:
        material = deepcopy(cfg["contact_material"])
        material["ground_static_friction_range"] = [.5, .5]
        material["ground_dynamic_friction_range"] = [.5, .5]
    changes["contact_material"] = material
    if getattr(args, "friction_min", None) is not None:
        friction_max = getattr(args, "friction_max", .5)
        if not 0. <= args.friction_min <= friction_max <= 1.:
            raise ValueError("Ground friction bounds must be ordered within [0,1]")
        for key in ("ground_static_friction_range", "ground_dynamic_friction_range"):
            material[key] = [args.friction_min, friction_max]
    task["active_friction_range"] = material["ground_static_friction_range"]
    changes["dynamics_randomization"] = rigid_body_domain(task["domains_enabled"], task["domain_scale"])
    if task.get("gap_robustness_version"):
        from wheeled_tasks.chassis.gap_robustness import rigid_profile
        changes["dynamics_randomization"] = rigid_profile(changes["dynamics_randomization"], task["gap_severity"])
    # This base contract describes mechanism construction. Task behavior lives
    # in UnifiedGetupEnv and is independently declared in task_contract.json.
    for key, value in changes.items():
        cfg[key] = deepcopy(value)
        if key in row["contract"]:
            row["contract"][key] = deepcopy(value)
    row["runtime"]["seed"] = args.seed
    row["runtime"]["contact_material"] = deepcopy(material)
    row["runtime"]["modules"]["dynamics_randomization"] = task["domains_enabled"]
    row["invocation"] = dict(num_envs=args.num_envs, seed=dict(rule="fixed", value=args.seed))
    design["scope"] = "unified_getup_mechanism_construction"
    design["review_basis"] = dict(base_contract=str(BASE_CONFIG), base_contract_sha256=digest(BASE_CONFIG),
        task_contract=str(args.output / "task_contract.json"))
    write_json(args.output / "config/design.json", design)
    cfg["design_preflight"] = dict(file=str((args.output / "config/design.json").resolve()),
        sha256=digest(args.output / "config/design.json"), stage="speed")
    write_json(args.output / "config/contract.json", cfg)
    write_json(args.output / "task_contract.json", dict(schema="robust-getup-task-v2", settings=task, base_checkpoint=str(PARENT),
        base_checkpoint_sha256=digest(PARENT), bank=str(args.bank), bank_sha256=digest(args.bank),
        motor_action_clip_policy_order=[task["leg_action_clip"]] * 4 + [task["wheel_action_clip"]] * 2,
        motor_position_scale=.25, wheel_velocity_scale=10.,
        policy_order=["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"],
        ordinary_fall_termination=False, temporary_body_support_allowed=True,
        scripted_recovery=False, internal_state_writes="reset_only",
        curriculum="easy poses and higher friction first; permanent hard-pose and arc exposure",
        locomotion_rehearsal_fraction=task["motion_fraction"], mixed_fraction_within_motion=.4,
        privileged_critic_last_two=("stationary XY displacement / 5 cm" if task["quiet_critic_anchor"] else "flat ground heights"),
        support_half_extent_xy_m=[48., 48.], boundary_xy_m=task["boundary_xy_m"],
        native_origin_mode=task["native_origin_mode"],
        mechanism_config_sha256=digest(args.output / "config/contract.json"),
        post_settle_coverage=True, action_range_changed=True,
        hardware_ready=False, source_training_updates_before_parent_export="unknown"))
    return cfg, task


def load_mean(torch):
    from wheeled_algo.command_residual import mean_from_checkpoint
    state = torch.load(PARENT, map_location="cpu", weights_only=True)
    plan = json.loads((ROOT / state["infos"]["plan_file"]).read_text())
    return mean_from_checkpoint(state, ROOT / plan["actor_bundle"] / "source.onnx")


def fixed_cases(bank, count, seed, ablations=False):
    """Frozen repetitions with body/leg holdout; both arc directions and speeds."""
    rng = random.Random(seed)
    poses = bank["poses"]
    pool = {}
    for i, pose in enumerate(poses):
        pool.setdefault(pose["family"], []).append(i)
    upright = next(i for i, p in enumerate(poses) if p["family"] == "upright" and p["leg_shape"] == "knee65_65")
    templates = []
    for family in ("front", "back", "left", "right", "inverted", "compound"):
        for suffix in ("symmetric", "asymmetric"):
            indices = [i for i in pool[family] if ("50_85" in poses[i]["leg_shape"] or "85_50" in poses[i]["leg_shape"]) == (suffix == "asymmetric")
                       and max(abs(poses[i]["roll_deg"]), abs(poses[i]["pitch_deg"])) == (180 if family == "inverted" else 65 if family == "compound" else 90)]
            templates.append(dict(name=f"{family}_{suffix}", recovery=True,
                pose_index=rng.choice(indices), command=[.5, 1. if len(templates) % 2 else -1., .305]))
    for name, command in (("stand", [0., 0., .305]), ("stand_low", [0., 0., .23]), ("stand_high", [0., 0., .42]), ("forward", [1., 0., .305]),
        ("backward", [-.75, 0., .305]), ("spin_left", [0., 1., .305]), ("spin_right", [0., -1., .305]),
        ("arc_forward_left", [1., 1., .305]), ("arc_forward_right", [1., -1., .305]),
        ("arc_reverse_left", [-.75, 1., .305]), ("arc_reverse_right", [-.75, -1., .305]),
        ("arc_fast_left", [3., 4., .305]), ("arc_fast_right", [3., -4., .305]),
        ("forward_fast", [3., 0., .305]), ("backward_fast", [-2.5, 0., .305]),
        ("spin_fast_left", [0., 4., .305]), ("spin_fast_right", [0., -4., .305])):
        templates.append(dict(name=name, recovery=False, pose_index=upright, command=command))
    for sign, side in ((1., "left"), (-1., "right")):
        templates.append(dict(name=f"moving_push_{side}", recovery=False, pose_index=upright,
                              command=[.7, 0., .305], push=True, push_sign=sign))
        templates.append(dict(name=f"straight_to_arc_{side}", recovery=False, pose_index=upright,
            command=[.8, 0., .305], sequence=[[2., [.8, sign, .305]], [7., [0., 0., .305]]]))
        templates.append(dict(name=f"spin_to_arc_{side}", recovery=False, pose_index=upright,
            command=[0., sign, .305], sequence=[[2., [.8, sign, .305]], [7., [-.6, sign, .305]]]))
    cases = [deepcopy(templates[i % len(templates)]) for i in range(count)]
    if ablations:
        from wheeled_algo.unified_getup_ablation import VARIANTS
        for i, case in enumerate(cases):
            case["policy_variant"] = VARIANTS[i // len(templates) % len(VARIANTS)]
    return cases


def evaluate(env, mean, output, torch, *, seed, seconds=12., cases=None):
    import numpy as np
    from wheeled_tasks.chassis.unified_getup import FAMILIES
    previous_fixed_cases = env.fixed_cases
    env.fixed_cases = (fixed_cases(env.bank, env.num_envs, seed, getattr(env, "ablation_variants", False))
                       if cases is None else cases)
    if len(env.fixed_cases) != env.num_envs:
        raise ValueError("One fixed evaluation case is required for every native environment")
    if getattr(env, "ablation_variants", False):
        from wheeled_algo.unified_getup_ablation import RecoveryAblations
        mean = RecoveryAblations(mean, [case["policy_variant"] for case in env.fixed_cases]).to(env.device)
    env.generator.manual_seed(seed)
    env.domain.rng.manual_seed(seed + 83977)
    env.reset(torch.arange(env.num_envs, device=env.device))
    overrides = getattr(env, "evaluation_overrides", {})
    if "torque_transport_delay_ms" in overrides:
        env.domain.torque_delay.fill_(overrides["torque_transport_delay_ms"])
    if "sensor_delay_policy_ticks" in overrides:
        env.domain.sensor_delay.fill_(overrides["sensor_delay_policy_ticks"])
    if "instant_sensor_group" in overrides:
        groups = dict(imu=list(range(6)), encoders=list(range(6, 18)),
                      leg_encoders=list(range(6, 10)) + list(range(12, 16)), wheel_velocity=[16, 17])
        env.domain.evaluation_immediate_features = groups[overrides["instant_sensor_group"]]
    from wheeled_tasks.chassis.evaluation_interventions import neutralize_evaluation_bias
    bias_intervention = neutralize_evaluation_bias(env, overrides.get("neutralize_bias"))
    initial_path = Path(output).with_name(Path(output).name+"_precontrol_state.npz")
    np.savez_compressed(initial_path,
        root_pose=env.robot.data.root_link_pose_w.torch.detach().cpu().numpy(),
        motor_position=env.robot.data.joint_pos.torch[:, env.ids].detach().cpu().numpy(),
        motor_velocity=env.robot.data.joint_vel.torch[:, env.ids].detach().cpu().numpy())
    env.cfg["record_diagnostics"] = True
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    snapshots = []
    max_gap = torch.zeros(env.num_envs, device=env.device)
    max_impact = max_gap.clone()
    max_impulse = max_gap.clone()
    passive_torque = max_gap.clone()
    terminated = ~alive.clone()
    completed = ~alive.clone()
    recovered = ~alive.clone()
    followed = ~alive.clone()
    family = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    in_motion_fall = ~alive.clone()
    recovered_after_push = ~alive.clone()
    failure_time = torch.full((env.num_envs,), float("nan"), device=env.device)
    causes = {key: ~alive.clone() for key in ("mechanical", "closure_gap", "impact_bad", "boundary_bad")}
    vx_sum, yaw_sum, vx_abs, yaw_abs, frames = [torch.zeros(env.num_envs, device=env.device) for _ in range(5)]
    start = time.monotonic()
    mean.eval()
    domain_parameters = dict(motor_strength=env.domain.strength.detach().cpu().tolist(),
        leg_kp_scale=env.domain.kp.detach().cpu().tolist(), leg_kd_scale=env.domain.kd.detach().cpu().tolist(),
        spring_strength=env.domain.spring.detach().cpu().tolist(), torque_delay_ms=env.domain.torque_delay.detach().cpu().tolist(),
        sensor_delay_policy_ticks=env.domain.sensor_delay.detach().cpu().tolist(), motor_alpha=env.domain.motor_alpha.detach().cpu().tolist(),
        sensor_bias=env.domain.bias.detach().cpu().tolist(), pd_feedback_period_ms=env.getup_feedback.period.detach().cpu().tolist(),
        pd_feedback_delay_ms=env.getup_feedback.delay.detach().cpu().tolist(), pd_encoder_bias_rad=env.getup_feedback.bias.detach().cpu().tolist())
    if env.domain.imu_mount is not None:
        domain_parameters.update({key: value.detach().cpu().tolist() for key,value in env.domain.imu_mount.parameters().items()})
    lane_material = None
    if env.contact_material is not None:
        material = env.contact_material
        wheel_static, _ = material.effective(material.ground_static)
        _, wheel_dynamic = material.effective(material.ground_dynamic)
        lane_material = dict(ground_static=material.ground_static.cpu().tolist(),
            ground_dynamic=material.ground_dynamic.cpu().tolist(),
            effective_wheel_static=wheel_static.cpu().tolist(),
            effective_wheel_dynamic=wheel_dynamic.cpu().tolist())
    with torch.inference_mode():
        for tick in range(round(seconds / env.policy_dt)):
            obs = env.get_observations()
            reference_state = {}
            observer = getattr(env, "standing_observer", None)
            if observer is not None:
                # Input-aligned state: step() also prepares the next observation.
                # Capture now so a severe tilt or recapture is never shifted by
                # one policy tick relative to the action's recorded observation.
                reference_state = dict(
                    standing_reference_active=observer.active.clone(),
                    standing_reference_anchor_m=observer.anchor.clone(),
                    standing_reference_filtered_m=observer.filtered.clone(),
                    standing_reference_severe_reset_count=observer.severe_reset_count.clone(),
                    standing_reference_output_valid=((observer.last_tick == env.tick) & observer.active).clone(),
                    standing_reference_age_s=((env.tick - env.standing_entry_tick) * env.policy_dt).clone())
            _, _, done, info = env.step(mean(obs["policy"]))
            data = info["diagnostics"]
            mask = alive.clone()
            max_gap = torch.maximum(max_gap, data["gap"] * mask)
            max_impact = torch.maximum(max_impact, data["impact"] * mask)
            max_impulse = torch.maximum(max_impulse, data["impulse"] * mask)
            passive_torque = torch.maximum(passive_torque, data["passive_motor_peak"] * mask)
            family = torch.where(data["enable_event"] & mask, data["settled_family"], family)
            recovered |= data["recovered"] & mask
            followed |= data["followup_success"] & mask
            in_motion_fall |= data["in_motion_fall"] & mask
            recovered_after_push |= data["recovered_after_push"] & mask
            failure_time[data["terminated"] & mask] = data["time"][data["terminated"] & mask]
            for key in causes:
                causes[key] |= data[key] & mask
            valid = mask & (data["time"] > 2.) & ~data["passive"] & ~env.recovery_active
            vx_sum += data["velocity"][:, 0] * valid
            yaw_sum += data["omega"][:, 2] * valid
            vx_abs += (data["velocity"][:, 0] - data["commands"][:, 0]).abs() * valid
            yaw_abs += (data["omega"][:, 2] - data["commands"][:, 1]).abs() * valid
            frames += valid
            terminated |= data["terminated"] & mask
            completed |= (data["time"] >= seconds - env.policy_dt / 2) & mask
            if True:  # 200 Hz traces are required to resolve 20..100 Hz standing jitter.
                keep = ("time", "gravity", "height", "commands", "requested_command", "velocity", "omega",
                        "actions", "motor_effort", "motor_velocity", "gap", "impact", "impulse", "passive", "stable",
                        "done", "terminated", "enable_event", "settled_family", "recovered", "followup_success",
                        "mechanical", "closure_gap", "impact_bad", "boundary_bad", "local_position", "motor_position", "wheel_contact",
                        "velocity_world")
                keep += ("pushing", "in_motion_fall", "recovered_after_push")
                keep += ("wheel_clearance", "wheel_clearance_peak", "wheel_contact_fraction")
                keep += ("wheel_air_run_peak", "dual_wheel_air_run_peak", "wheel_air_seconds")
                keep += tuple(key for key in ("leg_speed_peak_rad_s", "leg_effort_peak_nm", "leg_saturation_fraction",
                    "leg_effort_slew_rms_nm", "leg_saturation_run_s") if key in data)
                keep += tuple(key for key in ("recovery_support_cost_density", "native_reward", "reward_density",
                    "reward_potential_progress", "reward_recovery_active") if key in data)
                snapshots.append({k: data[k].detach().cpu().numpy() for k in keep}
                                 | {k: value.detach().cpu().numpy() for k, value in reference_state.items()}
                                 | {"alive": mask.cpu().numpy(), "observations": obs["policy"].cpu().numpy()})
            alive &= ~done
            if tick % 400 == 399:
                print(json.dumps(dict(stage="evaluation", output=str(output), simulated_seconds=(tick + 1) * env.policy_dt,
                    alive=int(alive.sum()), recovered=int(recovered.sum()), elapsed_s=time.monotonic()-start)), flush=True)
                trace_directory = getattr(env, "evaluation_trace_directory", None)
                if trace_directory is not None:
                    trace_directory = Path(trace_directory)
                    trace_directory.mkdir(parents=True, exist_ok=True)
                    block = snapshots[-400:]
                    np.savez_compressed(trace_directory / f"chunk_{tick//400:04d}.npz",
                        **{key: np.stack([s[key] for s in block]) for key in block[0]})
                    write_json(trace_directory / "progress.json", dict(completed_policy_steps=tick+1,
                        simulated_seconds=(tick+1)*env.policy_dt, finished=False))
    arrays = {key: np.stack([s[key] for s in snapshots]) for key in snapshots[0]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectory.npz", **arrays)
    lanes = []
    for i, case in enumerate(env.fixed_cases):
        n = int(frames[i])
        lanes.append(dict(case, lane=i, world_origin_m=env.origins[i].detach().cpu().tolist(),
            survived=bool(completed[i] & ~terminated[i]), terminated=bool(terminated[i]),
            domain_enabled=bool(env.domain.enabled[i]),
            first_reset_domain={key: value[i] for key, value in domain_parameters.items()},
            material_coefficients=({key: value[i] for key, value in lane_material.items()} if lane_material else None),
            recovered=bool(recovered[i]), followup_success=bool(followed[i]),
            post_settle_family=(FAMILIES[int(family[i])] if family[i] >= 0 else None),
            max_gap_m=float(max_gap[i]), max_impact_n=float(max_impact[i]), passive_motor_peak_nm=float(passive_torque[i]),
            max_contact_impulse_ns=float(max_impulse[i]),
            in_motion_fall=bool(in_motion_fall[i]), recovered_after_push=bool(recovered_after_push[i]),
            termination_time_s=float(failure_time[i]) if terminated[i] else None,
            termination_causes=[key for key in causes if causes[key][i]],
            tracking_frames=n, vx_mean_m_s=float(vx_sum[i] / n) if n else None,
            yaw_mean_rad_s=float(yaw_sum[i] / n) if n else None,
            vx_mae_m_s=float(vx_abs[i] / n) if n else None, yaw_mae_rad_s=float(yaw_abs[i] / n) if n else None))
    motion = [v for v in lanes if not v["recovery"]]
    recovery = [v for v in lanes if v["recovery"]]
    report = dict(schema="robust-getup-native-evaluation-v2", seed=seed, seconds=seconds,
        lanes=lanes, elapsed_s=time.monotonic() - start, policy_hz=200, physics_hz=1000,
        recovery_episodes=len(recovery), recovered=sum(v["recovered"] for v in recovery),
        recovery_and_motion_success=sum(v["followup_success"] and v["survived"] for v in recovery),
        motion_episodes=len(motion), motion_survived=sum(v["survived"] for v in motion),
        passive_motor_peak_nm=float(passive_torque.max()), max_gap_m=float(max_gap.max()),
        max_contact_impulse_ns=float(max_impulse.max()),
        material=env.cfg["contact_material"], trajectory_sha256=digest(output / "trajectory.npz"),
        native_origin_mode=("isolated_overlap_v1" if env.cfg.get("full_rl_isolated_origins") else "spaced_corridors_v1"),
        native_environment_isolation=env.startup_report.get("native_environment_isolation"),
        robust_domains=env.domain.summary(), actor_variant=env.actor_variant,
        task_contract_sha256=digest(output.parent / "task_contract.json"),
        checkpoint_sha256=digest(env.evaluation_checkpoint) if getattr(env, "evaluation_checkpoint", None) else None,
        ablation_variants=getattr(env, "ablation_variants", False),
        source_scope="native simulation; no hardware motion", hardware_ready=False)
    report["evaluation_overrides"] = getattr(env, "evaluation_overrides", {})
    report["precontrol_state_sha256"] = digest(initial_path)
    if bias_intervention is not None:
        report["bias_intervention"] = bias_intervention
    write_json(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("lanes", "material")}), flush=True)
    # Evaluation-only callers may construct from a frozen bank without the
    # training floor-start labels. Restore their fixed reset contract too.
    # Training callers retain the original None/reset behavior.
    env.fixed_cases = previous_fixed_cases
    env.cfg["record_diagnostics"] = False
    env.reset(torch.arange(env.num_envs, device=env.device))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap-actuator-timing", choices=("stress","native"),
                        help="Isolate sensor/mass robustness while retaining native actuator response")
    parser.add_argument("--bank", type=Path, default=ROOT / "reports/unified_robust_20261021/pose_bank002.json")
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate"), default="train")
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--isolated-origins", action="store_true",
                        help="Use independent native collision groups at the same near-zero origin; at most 256 environments")
    parser.add_argument("--allow-large-world-coordinates", action="store_true",
                        help="Historical diagnostic replay only; allows support beyond +/-1024 m")
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=211031)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--friction", choices=("high", "low"), default="high")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pre-eval", action="store_true")
    parser.add_argument("--eval-seconds", type=float, default=12.)
    parser.add_argument("--ablation-compare", action="store_true")
    parser.add_argument("--curriculum-updates", type=int, default=128)
    parser.add_argument("--curriculum-origin", type=int, default=0)
    parser.add_argument("--initialize", type=Path, default=ROOT / "runs/unified_getup_20261021/pilot002/model_final.pt")
    parser.add_argument("--no-domains", action="store_true")
    parser.add_argument("--quiet-leg-alpha", type=float, default=.25)
    parser.add_argument("--quiet-wheel-alpha", type=float, default=.65)
    parser.add_argument("--quiet-velocity-feature-scale", type=float,
                        help="Opt-in feature-coordinate experiment; omitted preserves the original .3-scaled dq features")
    parser.add_argument("--quiet-wheel-bound", type=float, default=.5)
    parser.add_argument("--actor-variant", choices=("robust", "parent", "undamped"), default="robust")
    parser.add_argument("--domain-scale", type=float, default=1.)
    parser.add_argument("--gap-robustness", action="store_true", help="Opt-in IMU mount, payload and leg-control robustness pilot")
    parser.add_argument("--gap-severity", type=float, default=1.)
    parser.add_argument("--friction-min", type=float)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--restore-quiet-checkpoint", type=Path)
    parser.add_argument("--freeze-quiet", action="store_true")
    parser.add_argument("--quiet-only", action="store_true", help="Freeze motion/recovery and tune the quiet head, allowing 5 cm slow sway")
    parser.add_argument("--quiet-feedback-only", "--quiet-velocity-only", dest="quiet_velocity_only", action="store_true",
                        help="Quiet-only profile: train only the explicitly migrated feedback branch, keeping existing quiet MLPs fixed")
    parser.add_argument("--standing-support-training", action="store_true",
                        help="Quiet-only profile: persistent request anchor, geometric support cost, 50%% high-height exposure")
    parser.add_argument("--early-standing-position", action="store_true",
                        help="Charge the 5 cm position envelope after the first .2 s instead of deferring it with vibration startup")
    parser.add_argument("--standing-drift-objective", action="store_true",
                        help="Opt-in bounded 2 cm/s standing objective in addition to the 5 cm position deadband")
    parser.add_argument("--standing-drift-speed-scale", type=float,
                        help="Explicit drift cost scale in m/s, .005...02; requires the standing drift objective")
    parser.add_argument("--quiet-exploration-fraction", type=float,
                        help="Quiet-only exploration multiplier, .05...35; preserves motion/recovery sampling and deterministic mean")
    parser.add_argument("--quiet-wheel-exploration-fraction", type=float,
                        help="Quiet-only wheel exploration multiplier, .05...35; preserves leg and nonquiet sampling")
    parser.add_argument("--high-standing-only", action="store_true",
                        help="Opt-in standing pilot that samples only the 420 mm high request")
    parser.add_argument("--focus-recovery-quiet", action="store_true", help="Tune recovery and quiet standing while freezing parent and arc motion")
    parser.add_argument("--arc-only", action="store_true", help="Tune only the shared motion head with extra moderate simultaneous requests")
    parser.add_argument("--recovery-only", action="store_true",
                        help="Tune recovery with bounded world coordinates and more motion disturbances; freeze quiet/motion/std")
    parser.add_argument("--entry-recovery-only", action="store_true",
                        help="Train only the explicitly migrated learned entry head; freeze fallen/motion/quiet/std")
    parser.add_argument("--recovery-support-weight", type=float,
                        help="Opt-in training density cost for near-upright unrecovered wheel clearance,0..2")
    parser.add_argument("--side-recovery-focus", action="store_true",
                        help="Recovery-only reset exposure: 70%% side, 15%% inverted, 15%% other falls; retain20%% upright motion births")
    parser.add_argument("--motion-recovery-focus", action="store_true",
                        help="Recovery-only exposure: 80%% upright births with existing moving pushes, 20%% static recovery; preserve forces/reward")
    args = parser.parse_args()
    if not (1 <= args.updates <= 256 and 32 <= args.num_envs <= 1024 and 64 <= args.rollout_steps <= 512
            and args.num_envs % 4 == 0 and 2 <= args.eval_seconds <= 12):
        parser.error("Use bounded pilots: 1..256 updates, 32..1024 environments, 64..512 rollout steps")
    args.output, args.bank = args.output.resolve(), args.bank.resolve()
    if not (0 < args.quiet_leg_alpha <= 1 and 0 < args.quiet_wheel_alpha <= 1):
        parser.error("Quiet action interpolation must be in (0,1]")
    if not (0 < args.quiet_wheel_bound <= 9 and (args.quiet_velocity_feature_scale is None or args.quiet_velocity_feature_scale > 0)):
        parser.error("Quiet wheel correction must be in (0,9] and feature scale positive")
    actor_lr_max = 3e-3 if args.quiet_velocity_only else 3e-4
    if not (0 <= args.domain_scale <= 1 and 1e-5 <= args.actor_lr <= actor_lr_max
            and (args.friction_min is None or .2 <= args.friction_min <= .5)):
        parser.error("Use bounded domain scale, friction and actor learning rate")
    if args.actor_variant != "robust" and args.mode != "evaluate":
        parser.error("Parent/undamped comparisons are evaluation-only")
    if args.ablation_compare and args.mode != "evaluate":
        parser.error("Read-only ablations require --mode evaluate")
    if args.quiet_only and args.freeze_quiet:
        parser.error("Quiet tuning cannot freeze its only trainable mean head")
    if args.quiet_velocity_only:
        if not (args.mode == "train" and args.quiet_only and args.standing_support_training and args.early_standing_position
                and args.standing_drift_objective and args.resume):
            parser.error("Velocity-only training requires a resumed quiet support/early-position/drift profile")
        if args.high_standing_only or args.restore_quiet_checkpoint or args.quiet_velocity_feature_scale is not None:
            parser.error("Velocity-only tuning preserves all three old quiet MLPs and their feature coordinates")
        import torch
        prior_velocity = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not prior_velocity["infos"]["settings"].get("quiet_velocity_feedback", False):
            parser.error("Velocity-only tuning requires an explicitly migrated velocity-feedback checkpoint")
    if args.standing_support_training and not args.quiet_only:
        parser.error("Standing support tuning requires --quiet-only so existing recovery/motion stay frozen")
    if args.early_standing_position and not args.standing_support_training:
        parser.error("Early position tuning requires --standing-support-training")
    if args.standing_drift_objective and not (args.standing_support_training and args.early_standing_position):
        parser.error("Standing drift tuning requires support and early-position profiles")
    if args.standing_drift_speed_scale is not None and (not args.standing_drift_objective
            or not .005 <= args.standing_drift_speed_scale <= .02):
        parser.error("Standing drift speed scale requires its objective and must be within .005...02 m/s")
    if args.quiet_exploration_fraction is not None and (not args.quiet_only
            or not .05 <= args.quiet_exploration_fraction <= .35):
        parser.error("Quiet exploration revision requires --quiet-only and a fraction within .05...35")
    if args.quiet_wheel_exploration_fraction is not None and (not args.quiet_only
            or not .05 <= args.quiet_wheel_exploration_fraction <= .35):
        parser.error("Quiet wheel exploration revision requires --quiet-only and a fraction within .05...35")
    if args.high_standing_only and not (args.quiet_only and args.standing_support_training
                                        and args.early_standing_position):
        parser.error("High-standing-only tuning requires quiet support and early-position profiles")
    if args.focus_recovery_quiet and (args.quiet_only or args.freeze_quiet):
        parser.error("Choose a single tuning mode")
    if args.arc_only and (args.quiet_only or args.freeze_quiet or args.focus_recovery_quiet):
        parser.error("Arc-only tuning already freezes quiet/recovery/std; choose a single mode")
    if args.recovery_only and (args.arc_only or args.quiet_only or args.freeze_quiet or args.focus_recovery_quiet):
        parser.error("Recovery-only tuning already freezes quiet/motion/std; choose a single mode")
    if args.entry_recovery_only and not (args.mode == "train" and args.recovery_only and args.resume):
        parser.error("Entry-only learning requires train, recovery-only and an explicitly migrated resume")
    if args.recovery_support_weight is not None and not args.entry_recovery_only:
        parser.error("This reward pilot requires entry-only learning to preserve early fallen actions")
    if args.side_recovery_focus and not args.recovery_only:
        parser.error("Side recovery exposure requires --recovery-only")
    if args.motion_recovery_focus and (not args.recovery_only or args.side_recovery_focus):
        parser.error("Moving recovery exposure requires --recovery-only without side-recovery-focus")
    if args.curriculum_origin < 0:
        parser.error("Curriculum origin must be nonnegative; it does not reset optimizer consumption")
    if args.resume:
        import torch
        resume_info = torch.load(args.resume, map_location="cpu", weights_only=False)["infos"]
        function_root = resume_info.get("schema") == "robust-getup-checkpoint-v3"
        fallen_function_root = resume_info.get("schema") == "robust-getup-checkpoint-v5"
        if fallen_function_root and (args.mode != "train" or not args.recovery_only or args.entry_recovery_only):
            parser.error("V5 resumes require the existing fallen output-only training mode")
        revised_function = (resume_info["settings"].get("gap_entry_encoder_version")
                            or resume_info["settings"].get("motion_leg_damping_version"))
        if function_root and (args.mode != "train" or not args.entry_recovery_only):
            parser.error("V3 resumes support entry-only training; use the dedicated native scripts for evaluation")
        if args.mode == "train" and (function_root or revised_function) and not fallen_function_root:
            if not function_root or not args.entry_recovery_only:
                parser.error("Periodic/motion training requires the explicit V3 entry-only optimizer initialization")
        if function_root or revised_function or fallen_function_root:
            from export_robust_getup import load_mean as load_authenticated_mean
            load_authenticated_mean(args.resume)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, task = prepare(args)
    if args.resume and (function_root or fallen_function_root):
        if fallen_function_root:
            from wheeled_algo.fallen_output_checkpoint import validate_training_settings
            root_record = resume_info["policy_function_root"]
        else:
            from wheeled_algo.function_root_checkpoint import validate_training_settings
            root_record = resume_info["training_function_root"]
        from export_robust_getup import load_mean as load_authenticated_mean
        root_info = torch.load(root_record["checkpoint"],
                               map_location="cpu", weights_only=False)["infos"]
        validate_training_settings(task, root_info["settings"])
    if args.curriculum_updates != task["curriculum_updates"]:
        from wheeled_tasks.chassis.unified_getup import validate_settings
        task["curriculum_updates"] = args.curriculum_updates
        validate_settings(task)
        task_path = args.output / "task_contract.json"
        task_contract = json.loads(task_path.read_text())
        task_contract["settings"] = task
        write_json(task_path, task_contract)
    sources = [ROOT / "scripts/train_robust_getup.py", ROOT / "scripts/build_robust_getup_bank.py",
        ROOT / "src/wheeled_tasks/chassis/robust_getup.py", ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py",
        ROOT / "src/wheeled_algo/robust_getup.py",
        ROOT / "src/wheeled_tasks/chassis/unified_getup.py", ROOT / "src/wheeled_tasks/chassis/unified_getup_env.py",
        ROOT / "src/wheeled_algo/unified_getup.py", ROOT / "src/wheeled_tasks/chassis/env.py",
        ROOT / "src/wheeled_tasks/chassis/v5_control.py", ROOT / "src/wheeled_tasks/chassis/paired_leg_targets.py",
        ROOT / "src/wheeled_algo/recovery_rollout.py", ROOT / "src/wheeled_algo/unified_getup_ablation.py"]
    sources = sorted(set(sources) | set((ROOT / "src/wheeled_tasks/chassis").glob("*.py"))
                     | set((ROOT / "src/wheeled_algo").glob("*.py")))
    if task.get("quiet_velocity_feedback_mode") in ("rolling_velocity", "standing_displacement"):
        from wheeled_algo.rolling_velocity import frozen_table_path
        sources.append(frozen_table_path())
        from wheeled_tasks.chassis.standing_geometry107 import SCHEMA, table_path
        from wheeled_tasks.chassis.standing_sensor_winding import SCHEMA as WINDING_SCHEMA
        from wheeled_tasks.chassis.standing_sensor_uncertainty import SCHEMA as SENSOR_SCHEMA
        from wheeled_tasks.chassis.standing_recovery_reference import SCHEMA as REFERENCE_SCHEMA
        if (task.get("standing_position_observation") or {}).get("schema") in (SCHEMA, WINDING_SCHEMA, SENSOR_SCHEMA, REFERENCE_SCHEMA):
            sources.append(table_path())
    hashes = {}
    for p in sources:
        relative = p.relative_to(ROOT)
        hashes[str(relative)] = digest(p)
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    report = dict(schema="robust-getup-pilot-v2", status="starting", pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(), new_optimizer_updates=0,
        new_optimizer_attempts=0, optimizer_update_in_flight=False,
        completed_updates=0, source_sha256=hashes, task_contract_sha256=digest(args.output / "task_contract.json"),
        hardware_ready=False, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(args.output / "progress.json", report)
    launcher = env = runner = None
    start = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from wheeled_tasks.chassis.robust_getup_env import RobustGetupEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_algo.robust_getup import RobustGetupActor, SensorConditionedMean
        from wheeled_algo.exploration_init import install_axis_std_logging
        from wheeled_algo.recovery_rollout import RecoveryRolloutMask
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        manifest = json.loads((ROOT / cfg["asset_directory"] / "manifest.json").read_text())
        control = load_chassis_control(ROOT / cfg["control_math_source"])
        env = RobustGetupEnv(cfg, manifest, control, ROOT, stage_name="speed", num_envs=args.num_envs,
            device=args.device, seed=args.seed, level=1., run_role="train", bank=args.bank, settings=task)
        actor_type, extra = RobustGetupActor, {}
        if "quiet_output_wheel_alpha" in task:
            from wheeled_algo.quiet_output_actor import QuietOutputGetupActor
            actor_type = QuietOutputGetupActor
            extra["quiet_output_wheel_alpha"] = task["quiet_output_wheel_alpha"]
        if task.get("gap_robustness_version"):
            from wheeled_algo.gap_robust_actor import GapRobustGetupActor
            actor_type = GapRobustGetupActor
            extra["upright_entry_recovery"] = task["upright_entry_recovery"]
            if task.get("gap_entry_handoff_version") == "orbit-v2":
                from wheeled_algo.gap_entry_handoff import HandoffGapGetupActor
                actor_type = HandoffGapGetupActor
            if task.get("gap_upright_target_limit_version"):
                if task.get("gap_entry_handoff_version") != "orbit-v2":
                    raise ValueError("Final target limiter requires the authenticated orbit-v2 actor")
                from wheeled_algo.upright_target_limit import UprightTargetLimitedGapActor
                actor_type = UprightTargetLimitedGapActor
                extra["final_target_winding_version"] = task.get("gap_final_target_winding_version")
            if task.get("gap_entry_recovery_split_version"):
                if task.get("gap_entry_handoff_version") != "orbit-v2" or task.get("gap_upright_target_limit_version") != "error-v1":
                    raise ValueError("Frozen entry requires orbit-v2 and the unchanged final target limit")
                from wheeled_algo.gap_entry_preservation import EntryPreservingGapActor
                actor_type = EntryPreservingGapActor
        if task.get("gap_entry_encoder_version"):
            if not task.get("gap_entry_recovery_split_version"):
                raise ValueError("Periodic entry requires the separate entry head")
            from wheeled_algo.periodic_entry_actor import PeriodicEntryGapActor
            actor_type = PeriodicEntryGapActor
        if task.get("motion_leg_damping_version"):
            from wheeled_algo.motion_target_damping import MotionTargetDampedGapActor
            actor_type = MotionTargetDampedGapActor
            extra["motion_leg_alpha"] = task["motion_leg_damping_alpha"]
        mean = actor_type(load_mean(torch), quiet_alpha=task["quiet_actor_alpha"],
            canonical_recovery=task["canonical_recovery"], motion_residual_fraction=task["motion_residual_fraction"],
            cyclic_leg_targets=task["cyclic_leg_targets"], leg_action_clip=task["leg_action_clip"],
            recovery_leg_step=task["recovery_leg_step"], smooth_knee_targets=task["smooth_knee_targets"],
            recovery_knee_center=task["recovery_knee_center"],
            recovery_knee_step=task.get("recovery_knee_step", .15),
            recovery_previous_common_version=task.get("recovery_previous_common_version"),
            quiet_velocity_feature_scale=task["quiet_velocity_feature_scale"], quiet_wheel_bound=task["quiet_wheel_bound"],
            quiet_velocity_feature_gain=task.get("quiet_velocity_feature_gain", 1.),
            quiet_velocity_feedback=task.get("quiet_velocity_feedback", False),
            quiet_velocity_feedback_mode=task.get("quiet_velocity_feedback_mode", "all"),
            stance_aware_recovery=task["stance_aware_recovery"],
            recovery_shape_aware=task["recovery_shape_aware"],
            quiet_high_common_alpha=task.get("quiet_high_common_alpha"),
            quiet_high_leg_prediction_s=task.get("quiet_high_leg_prediction_s", 0.),
            quiet_leg_prediction_all_heights=task.get("quiet_leg_prediction_all_heights", False),
            quiet_height_experts=task.get("quiet_height_experts", False),
            quiet_native_height_anchors=task.get("quiet_native_height_anchors", False),
            fast_forward_reference=task.get("fast_forward_reference", False),
            fast_forward_reference_range=task.get("fast_forward_reference_range", [1.5, 2.]),
            fast_forward_reference_yaw_fade=task.get("fast_forward_reference_yaw_fade", .25), **extra).to(args.device)
        initialization = torch.load(args.initialize, map_location=args.device, weights_only=False)
        if initialization["infos"]["schema"] != "unified-getup-checkpoint-v1":
            raise ValueError("Initialization must be an authenticated v1 recovery checkpoint")
        mean.initialize_from_unified({k.removeprefix("mlp."): v for k, v in initialization["actor_state_dict"].items() if k.startswith("mlp.")})
        mean.action_bound[:4].fill_(task["leg_action_clip"])
        if args.resume and fallen_function_root:
            # Rebuild the authenticated complete runtime, including its explicit
            # zero-strength stance extension. Optimizer layout stays legacy28.
            mean, _ = load_authenticated_mean(args.resume)
            mean = mean.to(args.device)
        env.actor_variant = args.actor_variant
        env.ablation_variants = args.ablation_compare
        report["startup"] = env.startup_report
        report["trainable_parameters"] = sum(p.numel() for p in mean.parameters() if p.requires_grad)
        if args.mode == "smoke":
            smoke_gap = smoke_passive_tau = 0.
            smoke_terminations = 0
            with torch.inference_mode():
                for _ in range(300):
                    _, _, _, info = env.step(mean(env.get_observations()["policy"]))
                    smoke_gap = max(smoke_gap, float(info["diagnostics"]["gap"].max()))
                    smoke_passive_tau = max(smoke_passive_tau, float(info["diagnostics"]["passive_motor_peak"].max()))
                    smoke_terminations += int(info["diagnostics"]["terminated"].sum())
            report.update(status="native_smoke_completed", max_gap_m=smoke_gap,
                          passive_motor_peak_nm=smoke_passive_tau, terminations=smoke_terminations)
        else:
            runner_cfg = json.loads((ROOT / "runs/flat_base_20261021/transient_leg_feedback_pilot003/agent_config.json").read_text())
            runner_cfg.update(num_steps_per_env=args.rollout_steps, device=args.device, seed=args.seed,
                              save_interval=8, logger="tensorboard")
            runner_cfg["algorithm"].update(class_name="PPO", learning_rate=args.actor_lr, schedule="fixed",
                entropy_coef=.002, value_loss_coef=1., num_learning_epochs=3, num_mini_batches=4,
                gamma=task["reward_gamma"], lam=.99, use_clipped_value_loss=False)
            runner_cfg["actor"]["distribution_cfg"] = dict(class_name="HeteroscedasticGaussianDistribution",
                init_std=.5, std_type="log", std_range=[.003, .8])
            write_json(args.output / "agent_config.json", runner_cfg)
            runner = OnPolicyRunner(env, runner_cfg, log_dir=str(args.output), device=args.device)
            distribution_type = SensorConditionedMean
            if task.get("gap_robustness_version"):
                from wheeled_algo.gap_robust_actor import GapSensorConditionedMean
                distribution_type = GapSensorConditionedMean
                if task.get("gap_entry_handoff_version") == "orbit-v2":
                    from wheeled_algo.gap_entry_handoff import HandoffGapConditionedMean
                    distribution_type = HandoffGapConditionedMean
            distribution_kwargs = {}
            if "gap_recovery_leg_exploration_fraction" in task:
                if task.get("gap_entry_handoff_version") != "orbit-v2":
                    raise ValueError("Bounded fallen exploration requires the orbit-v2 distribution")
                from wheeled_algo.gap_recovery_exploration import GapRecoveryNoiseMean
                distribution_type = GapRecoveryNoiseMean
                distribution_kwargs["recovery_leg_fraction"] = task["gap_recovery_leg_exploration_fraction"]
            runner.alg.actor.mlp = distribution_type(mean, task.get("quiet_exploration_fraction", .35),
                task.get("quiet_wheel_exploration_fraction"), **distribution_kwargs).to(args.device)
            if any(p.device != torch.device(args.device) for p in runner.alg.actor.parameters()):
                raise RuntimeError("Actor parameters must all reside on the policy device")
            runner.alg.critic.load_state_dict(initialization["critic_state_dict"], strict=True)
            if task.get("gap_entry_optimizer_layout") is not None:
                from wheeled_algo.entry_optimizer import prepare_entry_optimizer_parameters
                prepare_entry_optimizer_parameters(mean, task)
            runner.alg.optimizer = torch.optim.Adam([
                dict(params=[p for p in runner.alg.actor.parameters() if p.requires_grad], lr=args.actor_lr, name="actor"),
                dict(params=runner.alg.critic.parameters(), lr=1e-3, name="critic")])
            parent_updates = 0
            quiet_source_info = None
            if args.resume:
                state = torch.load(args.resume, map_location=args.device, weights_only=False)
                if state["infos"]["schema"] not in ("robust-getup-checkpoint-v2", "robust-getup-checkpoint-v3", "robust-getup-checkpoint-v5") or state["infos"]["bank_sha256"] != digest(args.bank):
                    raise ValueError("Resume checkpoint or mechanism bank differs")
                runner.alg.load(state, load_cfg=None, strict=True)
                parent_updates = state["infos"]["completed_updates"]
                quiet_source_info = state["infos"]
                env.generator.set_state(state["infos"]["environment_rng"].cpu())
                env.domain.rng.set_state(state["infos"]["domain_rng"].cpu())
                torch.set_rng_state(state["infos"]["torch_rng"].cpu())
                if args.device.startswith("cuda"):
                    torch.cuda.set_rng_state_all([v.cpu() for v in state["infos"]["cuda_rng"]])
                report["restored_from"] = str(args.resume.resolve())
                report["restored_sha256"] = digest(args.resume)
                report["task_revision"] = dict(previous=state["infos"]["settings"], current=task,
                    restore_scope="actor, critic, Adam, RNG; native episodes restart from reset distribution")
                mean.quiet_alpha.copy_(mean.quiet_alpha.new_tensor(task["quiet_actor_alpha"]))
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = args.actor_lr if group["name"] == "actor" else 1e-3
            runner.alg.learning_rate = args.actor_lr
            # A revision may enlarge the representation without changing torque
            # limits. Checkpoint buffers must not silently restore the old clip.
            mean.action_bound[:4].fill_(task["leg_action_clip"])
            mean.quiet_bound[4:].fill_(task["quiet_wheel_bound"])
            report["optimizer_learning_rates"] = {group["name"]: group["lr"] for group in runner.alg.optimizer.param_groups}
            quiet_heads = [mean.quiet] + ([mean.quiet_low, mean.quiet_high] if mean.quiet_height_experts else [])
            if mean.quiet_velocity_feedback_mode == "rolling_velocity":
                mean.rolling_velocity.verify_frozen_geometry()
            velocity_heads = list(mean.quiet_velocity.values()) if mean.quiet_velocity_feedback else []
            if args.restore_quiet_checkpoint:
                if mean.quiet_height_experts:
                    raise ValueError("Use an explicit height-expert migration to restore a quiet reference")
                quiet_state = torch.load(args.restore_quiet_checkpoint, map_location=args.device, weights_only=False)
                quiet_source_info = quiet_state["infos"]
                if quiet_state["infos"]["base_checkpoint_sha256"] != digest(PARENT):
                    raise ValueError("Quiet reference belongs to another locomotion parent")
                mean.quiet.load_state_dict({key.removeprefix("mlp.mean.quiet."): value
                    for key, value in quiet_state["actor_state_dict"].items() if key.startswith("mlp.mean.quiet.")}, strict=True)
                for parameter in mean.quiet.parameters():
                    runner.alg.optimizer.state.pop(parameter, None)
                report["quiet_initialization"] = dict(checkpoint=str(args.restore_quiet_checkpoint.resolve()),
                    sha256=digest(args.restore_quiet_checkpoint), optimizer="only quiet-head Adam moments cleared after weight restore")
            if quiet_source_info is not None:
                previous_quiet = quiet_source_info["settings"]
                old_scale = previous_quiet.get("quiet_velocity_feature_scale") or .3
                new_scale = task["quiet_velocity_feature_scale"] or .3
                old_bound = previous_quiet.get("quiet_wheel_bound", .5)
                if old_scale != new_scale or old_bound != task["quiet_wheel_bound"]:
                    with torch.no_grad():
                        for head in quiet_heads:
                            head[0].weight[:, 16:22].mul_(old_scale / new_scale)
                            head[-1].weight[4:].mul_(old_bound / task["quiet_wheel_bound"])
                            head[-1].bias[4:].mul_(old_bound / task["quiet_wheel_bound"])
                    for head in quiet_heads:
                        for parameter in head.parameters():
                            runner.alg.optimizer.state.pop(parameter, None)
                    report["quiet_coordinate_migration"] = dict(old_velocity_scale=old_scale, new_velocity_scale=new_scale,
                        old_wheel_bound=old_bound, new_wheel_bound=task["quiet_wheel_bound"],
                        purpose="preserve the small-signal initial action while improving quiet-head learning conditioning",
                        optimizer="only rescaled quiet-head moments cleared")
            if task.get("gap_entry_optimizer_layout") is not None:
                mean.entry_recovery.requires_grad_(False)
            if args.freeze_quiet:
                for head in quiet_heads + velocity_heads:
                    head.requires_grad_(False)
            if args.quiet_only:
                mean.recovery.requires_grad_(False)
                mean.arc.requires_grad_(False)
                runner.alg.actor.mlp.log_std.requires_grad_(False)
                if args.quiet_velocity_only:
                    mean.freeze_except_quiet_velocity()
                if args.high_standing_only and mean.quiet_height_experts:
                    mean.quiet.requires_grad_(False)
                    mean.quiet_low.requires_grad_(False)
                    if mean.quiet_velocity_feedback:
                        mean.quiet_velocity["mid"].requires_grad_(False)
                        mean.quiet_velocity["low"].requires_grad_(False)
            if args.focus_recovery_quiet:
                mean.arc.requires_grad_(False)
            if args.arc_only:
                for head in quiet_heads + velocity_heads:
                    head.requires_grad_(False)
                mean.recovery.requires_grad_(False)
                runner.alg.actor.mlp.log_std.requires_grad_(False)
            if args.recovery_only:
                for head in quiet_heads + velocity_heads:
                    head.requires_grad_(False)
                mean.arc.requires_grad_(False)
                runner.alg.actor.mlp.log_std.requires_grad_(False)
            if args.resume and fallen_function_root:
                runner.alg.actor.requires_grad_(False)
                mean.recovery[-1].requires_grad_(True)
            if args.entry_recovery_only:
                from wheeled_algo.entry_optimizer import freeze_except_entry
                freeze_except_entry(runner.alg.actor.mlp)
            recovery_trainable_prefix = "entry_recovery." if args.entry_recovery_only else "recovery."
            quiet_trainable_prefixes = (("quiet_velocity.",) if args.quiet_velocity_only
                else ("quiet_high.", "quiet_velocity.high.") if args.high_standing_only and mean.quiet_height_experts
                else ("quiet.", "quiet_low.", "quiet_high.", "quiet_velocity."))
            frozen_motion = ({k: v.detach().clone() for k, v in mean.state_dict().items()
                              if not k.startswith(quiet_trainable_prefixes)}
                              if args.quiet_only else None)
            frozen_quiet_std = runner.alg.actor.mlp.log_std.detach().clone() if args.quiet_only else None
            frozen_parent_arc = ({k: v.detach().clone() for k, v in mean.state_dict().items()
                                  if k.startswith(("parent.", "arc."))} if args.focus_recovery_quiet else None)
            frozen_non_arc = ({k: v.detach().clone() for k, v in mean.state_dict().items()
                               if not k.startswith("arc.")} if args.arc_only else None)
            frozen_arc_std = runner.alg.actor.mlp.log_std.detach().clone() if args.arc_only else None
            frozen_non_recovery = ({k: v.detach().clone() for k, v in mean.state_dict().items()
                                    if not k.startswith(recovery_trainable_prefix)} if args.recovery_only else None)
            frozen_recovery_std = runner.alg.actor.mlp.log_std.detach().clone() if args.recovery_only else None
            report["trainable_actor_parameters"] = sum(p.numel() for p in runner.alg.actor.parameters() if p.requires_grad)
            report["parent_updates"] = parent_updates
            report["completed_updates"] = parent_updates
            env.actor_updates = parent_updates
            install_axis_std_logging(runner.logger)
            if "gap_recovery_leg_exploration_fraction" in task:
                original_log = runner.logger.log

                def log_distribution(*args, **kwargs):
                    result = original_log(*args, **kwargs)
                    if runner.logger.writer is not None:
                        runner.logger.writer.add_scalar("Exploration/recovery_leg_fraction",
                            task["gap_recovery_leg_exploration_fraction"], kwargs["it"])
                    return result

                runner.logger.log = log_distribution
            mask = RecoveryRolloutMask(runner.alg, num_envs=args.num_envs)
            original_update, original_save = runner.alg.update, runner.save

            def save(path, infos=None):
                info = dict(schema="robust-getup-checkpoint-v2", completed_updates=report["completed_updates"],
                    base_checkpoint=str(PARENT), base_checkpoint_sha256=digest(PARENT),
                    bank_sha256=digest(args.bank), task_contract_sha256=report["task_contract_sha256"],
                    initialize_checkpoint=str(args.initialize.resolve()), initialize_sha256=digest(args.initialize),
                    prior_recovery_updates=initialization["infos"]["completed_updates"],
                    domain_rng=env.domain.rng.get_state(), settings=task,
                    source_sha256=hashes, environment_rng=env.generator.get_state(), torch_rng=torch.get_rng_state(),
                    quiet_initialization=report.get("quiet_initialization"),
                    cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
                    actor_variant=args.actor_variant, migrated_from_sha256=report.get("restored_sha256"),
                    quiet_only_tuning=args.quiet_only,
                    quiet_velocity_only_tuning=args.quiet_velocity_only,
                    focus_recovery_quiet=args.focus_recovery_quiet,
                    arc_only_tuning=args.arc_only,
                    recovery_only_tuning=args.recovery_only,
                    entry_recovery_only_tuning=args.entry_recovery_only,
                    head_composition=(deepcopy(state["infos"].get("head_composition")) if args.resume else None),
                    hardware_ready=False, prior_parent_updates=1346, initial_policy_training_updates="unknown")
                if args.resume:
                    # Observation revisions and zero-PPO probes are ancestry,
                    # not optimizer updates; retain their exact provenance.
                    for key in ("observation_geometry_migration", "observation_winding_migration",
                                "observation_sensor_projection_migration", "observation_entry_reference_migration",
                                "observation_filter_migration", "feedback_scale_probe", "recovery_step_migration",
                                "pose_bank_migration", "recovery_knee_request_migration",
                                "recovery_exploration_migration", "entry_recovery_preservation",
                                "entry_optimizer_migration", "entry_recovery_learning"):
                        if key in state["infos"]:
                            info[key] = deepcopy(state["infos"][key])
                if args.entry_recovery_only:
                    if state["infos"]["schema"] == "robust-getup-checkpoint-v3":
                        from wheeled_algo.function_root_checkpoint import SCHEMA, make_progress, environment_revision
                        info.update(schema=SCHEMA,
                            training_function_root=deepcopy(state["infos"]["training_function_root"]),
                            function_root_stage="native-training",
                            function_root_learning=make_progress(args.resume, state, runner.alg.actor.state_dict(),
                                                                  report["new_optimizer_updates"]))
                        revision = environment_revision(root_info["source_sha256"], hashes, task)
                        if revision is not None:
                            info["function_root_environment_revision"] = revision
                    else:
                        from wheeled_algo.entry_checkpoint import make_entry_learning_record
                        info["entry_recovery_learning"] = make_entry_learning_record(
                            args.resume, state, runner.alg.actor.state_dict(),
                            report["completed_updates"], report["new_optimizer_updates"])
                if args.resume and fallen_function_root:
                    from wheeled_algo.fallen_output_checkpoint import SCHEMA, make_progress, environment_revision
                    info.update(schema=SCHEMA,policy_function_root=deepcopy(state["infos"]["policy_function_root"]),
                        policy_function_stage="native-training",
                        policy_function_learning=make_progress(args.resume,state,runner.alg.actor.state_dict(),report["new_optimizer_updates"]))
                    revision=environment_revision(root_info["source_sha256"],hashes,task)
                    if revision is not None:
                        info["policy_function_environment_revision"]=revision
                target=Path(path)
                pending=target.with_suffix(target.suffix+".pending")
                original_save(str(pending), infos=info)
                if args.resume and fallen_function_root:
                    from wheeled_algo.fallen_output_checkpoint import validate_state
                    from export_robust_getup import load_mean as load_checked_mean
                    root_path=state["infos"]["policy_function_root"]["checkpoint"]
                    root_state=torch.load(root_path,map_location="cpu",weights_only=False)
                    validate_state(torch.load(pending,map_location="cpu",weights_only=False),root_state,
                        torch.load(args.resume,map_location="cpu",weights_only=False),state["infos"]["policy_function_root"]["optimizer_actor_parameter_names"])
                    load_checked_mean(pending)
                pending.replace(target)

            stop_requested = False

            def request_stop(signum, frame):
                nonlocal stop_requested
                stop_requested = True
                report["stop_requested_signal"] = signum

            def update():
                # Charge an attempted update even if native PPO fails part way
                # through its minibatches. Never silently refund failed work.
                report["new_optimizer_attempts"] += 1
                report["optimizer_update_in_flight"] = True
                write_json(args.output / "progress.json", report)
                losses = original_update()
                report["new_optimizer_updates"] += 1
                report["optimizer_update_in_flight"] = False
                report["completed_updates"] = parent_updates + report["new_optimizer_updates"]
                if not all(torch.isfinite(p).all() for model in (runner.alg.actor, runner.alg.critic) for p in model.parameters()):
                    raise RuntimeError("Nonfinite PPO parameters")
                report["last_losses"] = {k: float(v) for k, v in losses.items()}
                report["valid_rollout"] = mask.metrics
                report["elapsed_s"] = time.monotonic() - start
                report["status"] = "training"
                env.actor_updates = report["completed_updates"]
                write_json(args.output / "progress.json", report)
                if stop_requested:
                    save(str(args.output / "model_stopped.pt"))
                    raise TrainingStopped()
                return losses

            runner.save, runner.alg.update = save, update
            if args.actor_variant == "parent":
                mean = mean.parent
                if task.get("standing_position_observation") is not None:
                    from wheeled_algo.robust_getup import ZeroStandingPosition
                    mean = ZeroStandingPosition(mean)
            elif args.actor_variant == "undamped":
                mean.quiet_alpha.fill_(1.)
            if args.pre_eval:
                env.evaluation_checkpoint = args.output / "model_initial.pt"
                save(str(env.evaluation_checkpoint))
                report["baseline_evaluation"] = evaluate(env, mean, args.output / "baseline_eval", torch,
                                                         seed=args.seed + 100, seconds=args.eval_seconds)
            if args.mode == "train":
                signal.signal(signal.SIGTERM, request_stop)
                signal.signal(signal.SIGINT, request_stop)
                runner.current_learning_iteration = parent_updates
                if not (args.output / "model_initial.pt").exists():
                    save(str(args.output / "model_initial.pt"))
                runner.learn(args.updates, init_at_random_ep_len=False)
                if frozen_motion is not None:
                    if (any(not torch.equal(v, mean.state_dict()[k]) for k, v in frozen_motion.items())
                            or not torch.equal(frozen_quiet_std, runner.alg.actor.mlp.log_std)):
                        raise RuntimeError("Quiet tuning modified a frozen motion/recovery parameter")
                    report["frozen_motion_recovery_unchanged"] = True
                    report["frozen_quiet_exploration_unchanged"] = True
                if frozen_parent_arc is not None:
                    if any(not torch.equal(v, mean.state_dict()[k]) for k, v in frozen_parent_arc.items()):
                        raise RuntimeError("Focused tuning modified frozen parent/arc motion weights")
                    report["frozen_parent_arc_unchanged"] = True
                if frozen_non_arc is not None:
                    if (any(not torch.equal(v, mean.state_dict()[k]) for k, v in frozen_non_arc.items())
                            or not torch.equal(frozen_arc_std, runner.alg.actor.mlp.log_std)):
                        raise RuntimeError("Arc tuning changed frozen recovery, quiet, parent or exploration")
                    report["frozen_non_arc_and_std_unchanged"] = True
                if frozen_non_recovery is not None:
                    if (any(not torch.equal(v, mean.state_dict()[k]) for k, v in frozen_non_recovery.items())
                            or not torch.equal(frozen_recovery_std, runner.alg.actor.mlp.log_std)):
                        raise RuntimeError("Recovery tuning changed frozen quiet, motion, parent or exploration")
                    report["frozen_non_entry_and_std_unchanged" if args.entry_recovery_only else "frozen_non_recovery_and_std_unchanged"] = True
                save(str(args.output / "model_final.pt"))
                env.evaluation_checkpoint = args.output / "model_final.pt"
            else:
                env.evaluation_checkpoint = args.output / "model_evaluated.pt"
                save(str(env.evaluation_checkpoint))
            report["evaluation"] = evaluate(env, mean, args.output / "evaluation", torch,
                                             seed=args.seed + 100, seconds=args.eval_seconds)
            report["status"] = "completed"
        report["elapsed_s"] = time.monotonic() - start
        write_json(args.output / "progress.json", report)
        print(json.dumps(dict(status=report["status"], updates=report["new_optimizer_updates"],
                              elapsed_s=report["elapsed_s"])), flush=True)
    except TrainingStopped:
        report.update(status="stopped", stopped_checkpoint=str(args.output / "model_stopped.pt"),
                      elapsed_s=time.monotonic() - start)
        write_json(args.output / "progress.json", report)
        print(json.dumps(dict(status="stopped", updates=report["new_optimizer_updates"])), flush=True)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc(), elapsed_s=time.monotonic() - start)
        write_json(args.output / "progress.json", report)
        raise
    finally:
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()

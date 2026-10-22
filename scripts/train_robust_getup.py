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
import sys
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PARENT = ROOT / "runs/flat_base_20261020/pilot034_directional200_1346/model_001346.pt"
BASE_CONFIG = PARENT.parent / "config/contract.json"


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
                    "recovery_leg_step", "smooth_knee_targets", "recovery_knee_center", "stance_aware_recovery",
                    "recovery_shape_aware", "quiet_high_common_alpha", "quiet_high_leg_prediction_s", "fast_forward_reference",
                    "fast_forward_reference_range", "fast_forward_reference_yaw_fade", "quiet_height_experts"):
            if key in prior:
                task[key] = deepcopy(prior[key])
        if prior.get("native_origin_mode") == "isolated_overlap_v1":
            task["native_origin_mode"] = "isolated_overlap_v1"
    if getattr(args, "isolated_origins", False):
        task["native_origin_mode"] = "isolated_overlap_v1"
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
    if getattr(args, "focus_recovery_quiet", False):
        task.update(version=task["version"] + "-recovery-quiet", quiet_training=True, motion_fraction=.6,
                    quiet_spectral_cost=True, quiet_critic_anchor=True)
    if getattr(args, "arc_only", False):
        task.update(version=task["version"] + "-arc", motion_fraction=.8, arc_tuning_fraction=.8)
    if getattr(args, "recovery_only", False):
        task.update(version="robust-v2.10-recovery-capture", stance_aware_recovery=True,
                    recovery_shape_aware=True, motion_push_fraction=1.)
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
    if "sensor_delay_policy_ticks" in overrides:
        env.domain.sensor_delay.fill_(overrides["sensor_delay_policy_ticks"])
    if "instant_sensor_group" in overrides:
        groups = dict(imu=list(range(6)), encoders=list(range(6, 18)),
                      leg_encoders=list(range(6, 10)) + list(range(12, 16)), wheel_velocity=[16, 17])
        env.domain.evaluation_immediate_features = groups[overrides["instant_sensor_group"]]
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
                snapshots.append({k: data[k].detach().cpu().numpy() for k in keep}
                                 | {"alive": mask.cpu().numpy(), "observations": obs["policy"].cpu().numpy()})
            alive &= ~done
            if tick % 400 == 399:
                print(json.dumps(dict(stage="evaluation", output=str(output), simulated_seconds=(tick + 1) * env.policy_dt,
                    alive=int(alive.sum()), recovered=int(recovered.sum()), elapsed_s=time.monotonic()-start)), flush=True)
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
    write_json(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("lanes", "material")}), flush=True)
    env.fixed_cases = None
    env.cfg["record_diagnostics"] = False
    env.reset(torch.arange(env.num_envs, device=env.device))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument("--friction-min", type=float)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--restore-quiet-checkpoint", type=Path)
    parser.add_argument("--freeze-quiet", action="store_true")
    parser.add_argument("--quiet-only", action="store_true", help="Freeze motion/recovery and tune the quiet head, allowing 5 cm slow sway")
    parser.add_argument("--standing-support-training", action="store_true",
                        help="Quiet-only profile: persistent request anchor, geometric support cost, 50%% high-height exposure")
    parser.add_argument("--early-standing-position", action="store_true",
                        help="Charge the 5 cm position envelope after the first .2 s instead of deferring it with vibration startup")
    parser.add_argument("--standing-drift-objective", action="store_true",
                        help="Opt-in bounded 2 cm/s standing objective in addition to the 5 cm position deadband")
    parser.add_argument("--high-standing-only", action="store_true",
                        help="Opt-in standing pilot that samples only the 420 mm high request")
    parser.add_argument("--focus-recovery-quiet", action="store_true", help="Tune recovery and quiet standing while freezing parent and arc motion")
    parser.add_argument("--arc-only", action="store_true", help="Tune only the shared motion head with extra moderate simultaneous requests")
    parser.add_argument("--recovery-only", action="store_true",
                        help="Tune recovery with bounded world coordinates and more motion disturbances; freeze quiet/motion/std")
    args = parser.parse_args()
    if not (1 <= args.updates <= 256 and 32 <= args.num_envs <= 1024 and 64 <= args.rollout_steps <= 512
            and args.num_envs % 4 == 0 and 2 <= args.eval_seconds <= 12):
        parser.error("Use bounded pilots: 1..256 updates, 32..1024 environments, 64..512 rollout steps")
    args.output, args.bank = args.output.resolve(), args.bank.resolve()
    if not (0 < args.quiet_leg_alpha <= 1 and 0 < args.quiet_wheel_alpha <= 1):
        parser.error("Quiet action interpolation must be in (0,1]")
    if not (0 < args.quiet_wheel_bound <= 9 and (args.quiet_velocity_feature_scale is None or args.quiet_velocity_feature_scale > 0)):
        parser.error("Quiet wheel correction must be in (0,9] and feature scale positive")
    if not (0 <= args.domain_scale <= 1 and 1e-5 <= args.actor_lr <= 3e-4
            and (args.friction_min is None or .2 <= args.friction_min <= .5)):
        parser.error("Use bounded domain scale, friction and actor learning rate")
    if args.actor_variant != "robust" and args.mode != "evaluate":
        parser.error("Parent/undamped comparisons are evaluation-only")
    if args.ablation_compare and args.mode != "evaluate":
        parser.error("Read-only ablations require --mode evaluate")
    if args.quiet_only and args.freeze_quiet:
        parser.error("Quiet tuning cannot freeze its only trainable mean head")
    if args.standing_support_training and not args.quiet_only:
        parser.error("Standing support tuning requires --quiet-only so existing recovery/motion stay frozen")
    if args.early_standing_position and not args.standing_support_training:
        parser.error("Early position tuning requires --standing-support-training")
    if args.standing_drift_objective and not (args.standing_support_training and args.early_standing_position):
        parser.error("Standing drift tuning requires support and early-position profiles")
    if args.high_standing_only and not (args.quiet_only and args.standing_support_training
                                        and args.early_standing_position):
        parser.error("High-standing-only tuning requires quiet support and early-position profiles")
    if args.focus_recovery_quiet and (args.quiet_only or args.freeze_quiet):
        parser.error("Choose a single tuning mode")
    if args.arc_only and (args.quiet_only or args.freeze_quiet or args.focus_recovery_quiet):
        parser.error("Arc-only tuning already freezes quiet/recovery/std; choose a single mode")
    if args.recovery_only and (args.arc_only or args.quiet_only or args.freeze_quiet or args.focus_recovery_quiet):
        parser.error("Recovery-only tuning already freezes quiet/motion/std; choose a single mode")
    if args.curriculum_origin < 0:
        parser.error("Curriculum origin must be nonnegative; it does not reset optimizer consumption")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, task = prepare(args)
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
    hashes = {}
    for p in sources:
        relative = p.relative_to(ROOT)
        hashes[str(relative)] = digest(p)
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    report = dict(schema="robust-getup-pilot-v2", status="starting", pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(), new_optimizer_updates=0,
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
        mean = RobustGetupActor(load_mean(torch), quiet_alpha=task["quiet_actor_alpha"],
            canonical_recovery=task["canonical_recovery"], motion_residual_fraction=task["motion_residual_fraction"],
            cyclic_leg_targets=task["cyclic_leg_targets"], leg_action_clip=task["leg_action_clip"],
            recovery_leg_step=task["recovery_leg_step"], smooth_knee_targets=task["smooth_knee_targets"],
            recovery_knee_center=task["recovery_knee_center"],
            quiet_velocity_feature_scale=task["quiet_velocity_feature_scale"], quiet_wheel_bound=task["quiet_wheel_bound"],
            stance_aware_recovery=task["stance_aware_recovery"],
            recovery_shape_aware=task["recovery_shape_aware"],
            quiet_high_common_alpha=task.get("quiet_high_common_alpha"),
            quiet_high_leg_prediction_s=task.get("quiet_high_leg_prediction_s", 0.),
            quiet_height_experts=task.get("quiet_height_experts", False),
            fast_forward_reference=task.get("fast_forward_reference", False),
            fast_forward_reference_range=task.get("fast_forward_reference_range", [1.5, 2.]),
            fast_forward_reference_yaw_fade=task.get("fast_forward_reference_yaw_fade", .25)).to(args.device)
        initialization = torch.load(args.initialize, map_location=args.device, weights_only=False)
        if initialization["infos"]["schema"] != "unified-getup-checkpoint-v1":
            raise ValueError("Initialization must be an authenticated v1 recovery checkpoint")
        mean.initialize_from_unified({k.removeprefix("mlp."): v for k, v in initialization["actor_state_dict"].items() if k.startswith("mlp.")})
        mean.action_bound[:4].fill_(task["leg_action_clip"])
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
            runner.alg.actor.mlp = SensorConditionedMean(mean).to(args.device)
            if any(p.device != torch.device(args.device) for p in runner.alg.actor.parameters()):
                raise RuntimeError("Actor parameters must all reside on the policy device")
            runner.alg.critic.load_state_dict(initialization["critic_state_dict"], strict=True)
            runner.alg.optimizer = torch.optim.Adam([
                dict(params=[p for p in runner.alg.actor.parameters() if p.requires_grad], lr=args.actor_lr, name="actor"),
                dict(params=runner.alg.critic.parameters(), lr=1e-3, name="critic")])
            parent_updates = 0
            quiet_source_info = None
            if args.resume:
                state = torch.load(args.resume, map_location=args.device, weights_only=False)
                if state["infos"]["schema"] != "robust-getup-checkpoint-v2" or state["infos"]["bank_sha256"] != digest(args.bank):
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
            # A revision may enlarge the representation without changing torque
            # limits. Checkpoint buffers must not silently restore the old clip.
            mean.action_bound[:4].fill_(task["leg_action_clip"])
            mean.quiet_bound[4:].fill_(task["quiet_wheel_bound"])
            report["optimizer_learning_rates"] = {group["name"]: group["lr"] for group in runner.alg.optimizer.param_groups}
            quiet_heads = [mean.quiet] + ([mean.quiet_low, mean.quiet_high] if mean.quiet_height_experts else [])
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
            if args.freeze_quiet:
                for head in quiet_heads:
                    head.requires_grad_(False)
            if args.quiet_only:
                mean.recovery.requires_grad_(False)
                mean.arc.requires_grad_(False)
                runner.alg.actor.mlp.log_std.requires_grad_(False)
                if args.high_standing_only and mean.quiet_height_experts:
                    mean.quiet.requires_grad_(False)
                    mean.quiet_low.requires_grad_(False)
            if args.focus_recovery_quiet:
                mean.arc.requires_grad_(False)
            if args.arc_only:
                for head in quiet_heads:
                    head.requires_grad_(False)
                mean.recovery.requires_grad_(False)
                runner.alg.actor.mlp.log_std.requires_grad_(False)
            if args.recovery_only:
                for head in quiet_heads:
                    head.requires_grad_(False)
                mean.arc.requires_grad_(False)
                runner.alg.actor.mlp.log_std.requires_grad_(False)
            frozen_motion = ({k: v.detach().clone() for k, v in mean.state_dict().items()
                              if not k.startswith(("quiet_high.",) if args.high_standing_only and mean.quiet_height_experts
                                                  else ("quiet.", "quiet_low.", "quiet_high."))}
                              if args.quiet_only else None)
            frozen_quiet_std = runner.alg.actor.mlp.log_std.detach().clone() if args.quiet_only else None
            frozen_parent_arc = ({k: v.detach().clone() for k, v in mean.state_dict().items()
                                  if k.startswith(("parent.", "arc."))} if args.focus_recovery_quiet else None)
            frozen_non_arc = ({k: v.detach().clone() for k, v in mean.state_dict().items()
                               if not k.startswith("arc.")} if args.arc_only else None)
            frozen_arc_std = runner.alg.actor.mlp.log_std.detach().clone() if args.arc_only else None
            frozen_non_recovery = ({k: v.detach().clone() for k, v in mean.state_dict().items()
                                    if not k.startswith("recovery.")} if args.recovery_only else None)
            frozen_recovery_std = runner.alg.actor.mlp.log_std.detach().clone() if args.recovery_only else None
            report["trainable_actor_parameters"] = sum(p.numel() for p in runner.alg.actor.parameters() if p.requires_grad)
            report["parent_updates"] = parent_updates
            report["completed_updates"] = parent_updates
            env.actor_updates = parent_updates
            install_axis_std_logging(runner.logger)
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
                    focus_recovery_quiet=args.focus_recovery_quiet,
                    arc_only_tuning=args.arc_only,
                    recovery_only_tuning=args.recovery_only,
                    hardware_ready=False, prior_parent_updates=1346, initial_policy_training_updates="unknown")
                original_save(path, infos=info)

            def update():
                losses = original_update()
                if not all(torch.isfinite(p).all() for model in (runner.alg.actor, runner.alg.critic) for p in model.parameters()):
                    raise RuntimeError("Nonfinite PPO parameters")
                report["new_optimizer_updates"] += 1
                report["completed_updates"] = parent_updates + report["new_optimizer_updates"]
                report["last_losses"] = {k: float(v) for k, v in losses.items()}
                report["valid_rollout"] = mask.metrics
                report["elapsed_s"] = time.monotonic() - start
                report["status"] = "training"
                env.actor_updates = report["completed_updates"]
                write_json(args.output / "progress.json", report)
                return losses

            runner.save, runner.alg.update = save, update
            if args.actor_variant == "parent":
                mean = mean.parent
            elif args.actor_variant == "undamped":
                mean.quiet_alpha.fill_(1.)
            if args.pre_eval:
                env.evaluation_checkpoint = args.output / "model_initial.pt"
                save(str(env.evaluation_checkpoint))
                report["baseline_evaluation"] = evaluate(env, mean, args.output / "baseline_eval", torch,
                                                         seed=args.seed + 100, seconds=args.eval_seconds)
            if args.mode == "train":
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
                    report["frozen_non_recovery_and_std_unchanged"] = True
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

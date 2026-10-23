#!/usr/bin/env python3
"""Same native scene and fixed cases for parent / quiet-control ablations.

This is an evaluation-only job. It neither initializes a PPO optimizer nor
changes the checkpoint. Separate traces and denominators are saved per actor.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def fixed_forward_cases(template, speeds, replicas, yaws=(0.,)):
    """Independent diagnostic lanes at each requested forward speed."""
    cases = []
    for _ in range(replicas):
        for speed in speeds:
            for yaw in yaws:
                case = deepcopy(template)
                name = f"forward_speed_{speed:g}" + (f"_yaw_{yaw:g}" if yaw else "")
                case.update(name=name, command=[speed, yaw, template["command"][2]])
                cases.append(case)
    return cases


def validate_stationary_push_scales(scales):
    if (not 1 <= len(scales) <= 4 or len(set(scales)) != len(scales)
            or any(not math.isfinite(value) or not 0. < value <= 1. for value in scales)):
        raise ValueError("Use1..4 distinct finite stationary push scales within(0,1]")


def fixed_stationary_push_cases(full, scales=None):
    """Keep the frozen push/pose; explicitly replace only its motion request."""
    if scales is not None:
        validate_stationary_push_scales(scales)
    cases = []
    for original in full:
        if original["name"] in ("moving_push_left", "moving_push_right"):
            case = deepcopy(original)
            case["name"] = original["name"].replace("moving_", "stationary_", 1)
            case["command"] = [0., 0., original["command"][2]]
            if scales is None:
                cases.append(case)
            else:
                for scale in scales:
                    scaled = deepcopy(case)
                    # The existing floating point sign multiplies BOTH native
                    # force and torque. No environment code or RNG changes.
                    scaled["push_sign"] *= scale
                    scaled["diagnostic_push_scale"] = scale
                    cases.append(scaled)
    return cases


def scaled_stationary_push(settings, scale):
    """Separate diagnostic disturbance; the default force/time stays frozen."""
    if not math.isfinite(scale) or not 0. < scale <= 1.:
        raise ValueError("Stationary push scale must be finite within (0,1]")
    settings = deepcopy(settings)
    for key in ("push_force_n", "push_torque_nm"):
        settings[key] *= scale
    return settings


def quiet_filter_variants(alphas):
    if (not 1 <= len(alphas) <= 4 or len(set(alphas)) != len(alphas)
            or any(not math.isfinite(value) or not .05 <= value <= 1. for value in alphas)):
        raise ValueError("Use1..4 distinct finite quiet wheel alphas within[.05,1]")
    return {f"quiet_wheel_alpha_{value:.17g}": value for value in alphas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path,
                        help="Optional exact initial checkpoint for a no-PPO coordinate-change comparison")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=211131)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--isolated-origins", action="store_true", default=None,
                        help="Explicit native layout: independent collision groups at a common near-zero origin")
    parser.add_argument("--allow-large-world-coordinates", action="store_true",
                        help="Historical diagnostic replay; large-world results cannot qualify an export")
    parser.add_argument("--domain-scale", type=float, default=0.,
                        help="Independent descriptive stress test; zero is the nominal confirmation")
    friction = parser.add_mutually_exclusive_group()
    friction.add_argument("--friction-min", type=float, default=.5)
    friction.add_argument("--friction-value", type=float,
                          help="Diagnostic fixed ground static/dynamic coefficient; effective wheel coefficient is also recorded")
    parser.add_argument("--deployed-onnx", type=Path, default=ROOT / "models/wheel_leg_flat_base_20261018/source.onnx",
                        help="Exact user-confirmed recovery_candidate source for the deployed comparison")
    parser.add_argument("--seconds", type=float, default=12., choices=[4., 12., 30., 60.])
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--case-names", nargs="+", help="Explicit diagnostic subset; cannot replace the full acceptance suite")
    subset.add_argument("--forward-speeds", type=float, nargs="+",
                        help="Diagnostic steady forward commands, e.g. 1.5 1.75 2; cannot qualify an export")
    subset.add_argument("--stationary-push-cases", action="store_true",
                        help="Diagnostic zero-command versions of the frozen moving-push cases; cannot qualify an export")
    parser.add_argument("--stationary-push-scale", type=float, default=1.,
                        help="Explicit force/torque scale for stationary-push diagnostics only; default preserves the frozen pusher")
    parser.add_argument("--stationary-push-scales", type=float, nargs="+",
                        help="Diagnostic1..4 strength grid using existing per-case force/torque multiplier")
    parser.add_argument("--quiet-wheel-output-alphas", type=float, nargs="+",
                        help="Nominal static diagnostic: filter complete quiet wheel outputs using the applied previous action")
    parser.add_argument("--quiet-wheel-position-release-bands", type=float, nargs="+",
                        help="Static diagnostic: alpha.25 baseline plus one or two position-release bands in metres")
    parser.add_argument("--quiet-wheel-gyro-release-bands", type=float, nargs="+",
                        help="Nominal static diagnostic: alpha.25 baseline plus one or two roll/pitch gyro-release bands in rad/s")
    parser.add_argument("--quiet-wheel-gyro-release-band", type=float, nargs=2,
                        help="Single-policy static gyro counterfactual; randomized confirmation uses equal-index serial native runs")
    parser.add_argument("--early-roll-full-degrees", type=float, nargs="+",
                        help="Nominal dynamic-push diagnostic: original control plus20deg-start recovery authority variants")
    parser.add_argument("--recovery-leg-steps", type=float, nargs="+",
                        help="Nominal recovery diagnostic: smaller common-angle increments; original.3 control is included")
    parser.add_argument("--recovery-leg-noise-fractions", type=float, nargs="+",
                        help="Zero-PPO sampled-policy exposure diagnostic; mean qualification is excluded")
    parser.add_argument("--recovery-noise-seed", type=int, default=211492,
                        help="Private common Gaussian draws across recovery-noise variants")
    parser.add_argument("--forward-yaws", type=float, nargs="+", default=[0.],
                        help="Diagnostic yaw rates crossed with --forward-speeds")
    parser.add_argument("--sensor-delay-policy-ticks", type=int, choices=[0, 1, 2],
                        help="Diagnostic counterfactual: override only policy observation delay after the matched reset")
    parser.add_argument("--instant-sensor-group", choices=["imu", "encoders", "leg_encoders", "wheel_velocity"],
                        help="Diagnostic: remove delay from only this sensor group; retain its noise and other delays")
    parser.add_argument("--neutralize-bias", choices=["policy_encoder", "policy_gyro", "pd_encoder"],
                        help="Diagnostic: zero one bias after the seeded reset, retaining all other draws and noise")
    parser.add_argument("--standing-command-feedback-gain", type=float,
                        help="Diagnostic standing-only forward request correction from wheel speed; cannot qualify an export")
    parser.add_argument("--standing-command-feedback-estimator", choices=["wheel", "rolling"], default="wheel",
                        help="Diagnostic velocity input: wheel proxy or sealed107 encoder/IMU rolling kinematics")
    parser.add_argument("--variants", nargs="+", default=["parent", "robust", "legs_only", "undamped"],
                        choices=["parent", "robust", "legs_only", "undamped", "reference", "raw_pd", "paired_pd", "deployed"])
    args = parser.parse_args()
    if sum(value is not None for value in (args.quiet_wheel_output_alphas,
            args.quiet_wheel_position_release_bands, args.quiet_wheel_gyro_release_bands,
            args.quiet_wheel_gyro_release_band, args.early_roll_full_degrees, args.recovery_leg_steps,
            args.recovery_leg_noise_fractions)) > 1:
        parser.error("Use only one action-filter or recovery-authority diagnostic grid")
    seeds = args.seeds or [args.seed]
    args.seed = seeds[0]
    max_replicas = 64 if args.case_names or args.forward_speeds or args.stationary_push_cases else 4
    if not 1 <= args.replicas <= max_replicas or len(set(args.variants)) != len(args.variants):
        parser.error("Use 1..4 full-suite replicas or 1..64 diagnostic-subset replicas, and unique variants")
    if not 0. <= args.domain_scale <= 1. or not 0. < args.friction_min <= .5:
        parser.error("Domain scale must be 0..1 and minimum friction in (0,.5]")
    if args.friction_value is not None and not 0. <= args.friction_value <= 1.:
        parser.error("Fixed ground friction must be in [0,1]")
    if (not math.isfinite(args.stationary_push_scale) or not 0. < args.stationary_push_scale <= 1.
            or args.stationary_push_scale != 1. and not args.stationary_push_cases):
        parser.error("Use finite stationary push scale in (0,1] only with --stationary-push-cases")
    if args.stationary_push_scales is not None:
        if not args.stationary_push_cases or args.stationary_push_scale != 1.:
            parser.error("Strength grid requires --stationary-push-cases and default global scale1")
        try:
            validate_stationary_push_scales(args.stationary_push_scales)
        except ValueError as error:
            parser.error(str(error))
    alpha_variants = {}
    if args.quiet_wheel_output_alphas is not None:
        if (args.variants != ["robust"] or args.domain_scale != 0.
                or set(args.case_names or []) != {"stand_low", "stand", "stand_high"}
                or args.sensor_delay_policy_ticks is not None or args.instant_sensor_group
                or args.neutralize_bias or args.standing_command_feedback_gain is not None
                or args.friction_value is not None or args.friction_min != .5):
            parser.error("Quiet output grid requires robust-only, nominal domain0, all three static cases and no other intervention")
        try:
            alpha_variants = quiet_filter_variants(args.quiet_wheel_output_alphas)
        except ValueError as error:
            parser.error(str(error))
        args.variants = list(alpha_variants)
    roll_variants = {}
    if args.early_roll_full_degrees is not None:
        values = args.early_roll_full_degrees
        if (args.variants != ["robust"] or args.domain_scale != 0.
                or set(args.case_names or []) != {"moving_push_left", "moving_push_right"}
                or args.sensor_delay_policy_ticks is not None or args.instant_sensor_group
                or args.neutralize_bias or args.standing_command_feedback_gain is not None
                or args.friction_value is not None or args.friction_min != .5
                or not 1 <= len(values) <= 2 or len(set(values)) != len(values)
                or any(not math.isfinite(value) or not 25. <= value <= 60. for value in values)):
            parser.error("Early roll grid requires nominal robust-only dynamic push pair,1..2 full angles within25..60deg and no other intervention")
        roll_variants = {f"early_roll20_to_{value:.17g}": value for value in values}
        args.variants = ["robust", *roll_variants]
    position_variants = {}
    if args.quiet_wheel_position_release_bands is not None:
        if (args.variants != ["robust"] or args.domain_scale != 0.
                or args.quiet_wheel_output_alphas is not None
                or args.early_roll_full_degrees is not None
                or set(args.case_names or []) != {"stand_low", "stand", "stand_high"}
                or args.sensor_delay_policy_ticks is not None or args.instant_sensor_group
                or args.neutralize_bias or args.standing_command_feedback_gain is not None
                or args.friction_value is not None):
            parser.error("Position-release grid requires nominal domain0, robust-only three static cases and no other intervention; randomized comparisons must use equal-index serial jobs")
        from wheeled_algo.evaluation_position_filter import position_filter_variants
        try:
            position_variants = position_filter_variants(args.quiet_wheel_position_release_bands)
        except ValueError as error:
            parser.error(str(error))
        args.variants = list(position_variants)
    gyro_variants = {}
    gyro_values = (args.quiet_wheel_gyro_release_bands if args.quiet_wheel_gyro_release_bands is not None
                   else args.quiet_wheel_gyro_release_band)
    if gyro_values is not None:
        grid = args.quiet_wheel_gyro_release_bands is not None
        if grid and (args.domain_scale != 0. or args.friction_min != .5):
            parser.error("Gyro-release grid requires nominal domain0; randomized comparisons must use equal-index serial jobs")
        if (args.variants != ["robust"] or len(seeds) != 1
                or set(args.case_names or []) != {"stand_low", "stand", "stand_high"}
                or args.sensor_delay_policy_ticks is not None or args.instant_sensor_group
                or args.neutralize_bias or args.standing_command_feedback_gain is not None
                or args.friction_value is not None):
            parser.error("Gyro-release diagnostic requires robust-only three static cases, one seed and no other intervention")
        from wheeled_algo.evaluation_gyro_filter import gyro_filter_variants
        try:
            checked = gyro_filter_variants(gyro_values)
        except ValueError as error:
            parser.error(str(error))
        gyro_variants = checked if grid else {"robust": gyro_values}
        args.variants = list(gyro_variants)
    step_variants = {}
    if args.recovery_leg_steps is not None:
        if (args.variants != ["robust"] or len(seeds) != 1 or args.domain_scale != 0.
                or set(args.case_names or []) != {"moving_push_left", "moving_push_right", "inverted_asymmetric"}
                or args.sensor_delay_policy_ticks is not None or args.instant_sensor_group
                or args.neutralize_bias or args.standing_command_feedback_gain is not None
                or args.friction_value is not None or args.friction_min != .5):
            parser.error("Recovery step grid requires nominal domain0, robust-only push pair plus inverted-asymmetric, one seed and no other intervention")
        from wheeled_algo.evaluation_recovery_step import recovery_step_variants
        try:
            step_variants = recovery_step_variants(args.recovery_leg_steps)
        except ValueError as error:
            parser.error(str(error))
        args.variants = list(step_variants)
    noise_variants = {}
    if args.recovery_leg_noise_fractions is not None:
        if (args.variants != ["robust"] or len(seeds) != 1 or args.domain_scale != 0.
                or set(args.case_names or []) != {"left_asymmetric", "right_asymmetric", "inverted_asymmetric",
                                                "moving_push_left", "moving_push_right"}
                or args.sensor_delay_policy_ticks is not None or args.instant_sensor_group
                or args.neutralize_bias or args.standing_command_feedback_gain is not None
                or args.friction_value is not None or args.friction_min != .5
                or not 0 <= args.recovery_noise_seed < 2**32):
            parser.error("Recovery noise grid requires nominal domain0, robust-only five recovery exposure cases, one seed and no other intervention")
        from wheeled_algo.evaluation_recovery_noise import recovery_noise_variants
        try:
            noise_variants = recovery_noise_variants(args.recovery_leg_noise_fractions)
        except ValueError as error:
            parser.error(str(error))
        args.variants = list(noise_variants)
    elif args.recovery_noise_seed != 211492:
        parser.error("A recovery noise seed requires its explicit sampled-policy grid")
    if args.forward_speeds and (len(set(args.forward_speeds)) != len(args.forward_speeds)
                               or not all(0. <= speed <= 3. for speed in args.forward_speeds)):
        parser.error("Use distinct diagnostic forward speeds in [0,3] m/s")
    if (len(set(args.forward_yaws)) != len(args.forward_yaws)
            or not all(-4. <= yaw <= 4. for yaw in args.forward_yaws)
            or (args.forward_yaws != [0.] and not args.forward_speeds)):
        parser.error("Use distinct yaw rates in [-4,4] together with --forward-speeds")
    if ("reference" in args.variants) != (args.reference_checkpoint is not None):
        parser.error("Use --reference-checkpoint together with the reference variant")
    if args.standing_command_feedback_gain is not None:
        if (not math.isfinite(args.standing_command_feedback_gain)
                or abs(args.standing_command_feedback_gain) > 8.
                or args.variants != ["robust"]
                or set(args.case_names or []) != {"stand_low", "stand", "stand_high"}
                or len(seeds) != 1):
            parser.error("Standing command feedback requires finite gain within +/-8, robust only, one seed, and the three standing cases")
    elif args.standing_command_feedback_estimator != "wheel":
        parser.error("The rolling estimate requires an explicit diagnostic command-feedback gain")
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    deployed_sha = None
    if "deployed" in args.variants:
        deployed_sha = hashlib.sha256(args.deployed_onnx.read_bytes()).hexdigest()
        if deployed_sha != "a00d13df1d4113fe949c2a6a15b508e5c491fd1a5e6881570bf025a8d3c29c29":
            parser.error("Deployed comparison requires the exact user-confirmed recovery_candidate SHA")
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("robust_training", ROOT / "scripts/train_robust_getup.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    selected_names = None
    if args.case_names:
        selected_names = set(args.case_names)
        bank = json.loads((ROOT / "reports/unified_robust_20261021/pose_bank002.json").read_text())
        known = {c["name"] for c in training.fixed_cases(bank, 35, seeds[0])}
        if len(selected_names) != len(args.case_names) or not selected_names <= known:
            parser.error("Use distinct known fixed-case names")
    case_count = (2 * len(args.stationary_push_scales or [1.]) if args.stationary_push_cases else len(args.forward_speeds) * len(args.forward_yaws)
                  if args.forward_speeds else len(selected_names) if selected_names else 35)
    per_variant = case_count * args.replicas
    count = per_variant * len(args.variants) * len(seeds)
    if len(seeds) > 2 or len(set(seeds)) != len(seeds) or count > 1024:
        parser.error("Use at most two distinct seeds and 1024 total native environments")
    checkpoint_sha = training.digest(args.checkpoint)
    if args.isolated_origins is None:
        import torch
        saved_settings = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["infos"]["settings"]
        args.isolated_origins = saved_settings.get("native_origin_mode") == "isolated_overlap_v1"
    config_args = argparse.Namespace(output=args.output, bank=ROOT / "reports/unified_robust_20261021/pose_bank002.json",
        no_domains=args.domain_scale == 0., domain_scale=args.domain_scale,
        friction_min=args.friction_min if args.friction_value is None else args.friction_value,
        friction_max=.5 if args.friction_value is None else args.friction_value,
        curriculum_updates=128, quiet_leg_alpha=.25, quiet_wheel_alpha=.65,
        num_envs=count, rollout_steps=128, mode="evaluate", friction="high", seed=args.seed,
        isolated_origins=args.isolated_origins,
        allow_large_world_coordinates=args.allow_large_world_coordinates, episode_seconds=max(12., args.seconds))
    cfg, task = training.prepare(config_args)
    if count < len(cfg["scene_groups"]):
        parser.error("Native population must cover every base scene group; increase diagnostic replicas or use the full suite")
    training.write_json(args.output / "progress.json", dict(status="starting", completed_updates=0,
        seeds=seeds, variants=args.variants, checkpoint_sha256=checkpoint_sha, optimizer_updates=0))
    sources = {}
    for path in sorted(set((ROOT / "src/wheeled_algo").glob("*.py"))
                       | set((ROOT / "src/wheeled_tasks/chassis").glob("*.py"))
                       | {Path(__file__), ROOT / "scripts/train_robust_getup.py"}):
        relative = path.resolve().relative_to(ROOT)
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        sources[str(relative)] = training.digest(path)
    launcher = env = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device="cuda:0", enable_cameras=False))
        import numpy as np
        import torch
        from torch import nn
        from export_robust_getup import load_mean as load_verified_mean
        from wheeled_tasks.chassis.robust_getup_env import RobustGetupEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        verified_mean, verified_info = load_verified_mean(args.checkpoint)
        if verified_info["settings"].get("quiet_velocity_feedback_mode") in ("rolling_velocity", "standing_displacement"):
            from wheeled_algo.rolling_velocity import frozen_table_path
            path = frozen_table_path()
            relative = path.relative_to(ROOT)
            target = args.output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            sources[str(relative)] = training.digest(path)
        for key in ("canonical_recovery", "motion_residual_fraction", "cyclic_leg_targets", "leg_action_clip",
                    "recovery_leg_step", "smooth_knee_targets", "recovery_knee_center", "quiet_actor_alpha",
                    "quiet_velocity_feature_scale", "quiet_wheel_bound", "stance_aware_recovery", "recovery_shape_aware",
                    "quiet_high_common_alpha", "quiet_high_leg_prediction_s", "quiet_leg_prediction_all_heights", "fast_forward_reference",
                    "fast_forward_reference_range", "fast_forward_reference_yaw_fade", "quiet_height_experts",
                    "quiet_native_height_anchors", "quiet_exploration_fraction", "quiet_wheel_exploration_fraction", "quiet_velocity_feature_gain",
                    "quiet_velocity_feedback", "quiet_velocity_feedback_mode", "standing_position_observation",
                    "standing_position_filter_seconds", "standing_position_reference", "quiet_output_wheel_alpha"):
            if key in verified_info["settings"]:
                task[key] = deepcopy(verified_info["settings"][key])
        if args.stationary_push_cases:
            task = scaled_stationary_push(task, args.stationary_push_scale)
        task_path = args.output / "task_contract.json"
        public_task = json.loads(task_path.read_text())
        public_task["settings"] = task
        if alpha_variants:
            public_task["evaluation_counterfactual"] = dict(
                kind="whole_quiet_wheel_output_filter_v1", variants=alpha_variants,
                previous_action_slots=[26, 27], learned_parameters_changed=False)
        if roll_variants:
            public_task["evaluation_counterfactual"] = dict(kind="early_roll_authority_v1",
                start_degrees=20., variant_full_degrees=roll_variants, learned_parameters_changed=False)
        if position_variants:
            public_task["evaluation_counterfactual"] = dict(kind="position_released_quiet_wheel_filter_v1",
                alpha_min=.25, variants=position_variants, position_slots=[14, 15], scale_per_m=20.,
                learned_parameters_changed=False, hidden_state_added=False)
        if gyro_variants:
            public_task["evaluation_counterfactual"] = dict(kind="gyro_released_quiet_wheel_filter_v1",
                alpha_min=.25, variants=gyro_variants, gyro_slots=[4, 5], gyro_scale_to_rad_s=2.,
                learned_parameters_changed=False, hidden_state_added=False)
        if step_variants:
            public_task["evaluation_counterfactual"] = dict(kind="smaller_recovery_common_step_v1",
                variants=step_variants, original_step=.3, raw_to_rad_scale=.25,
                learned_parameters_changed=False, knee_and_wheel_rules_changed=False)
        noise_log_std = None
        if noise_variants:
            agent = json.loads((args.checkpoint.parent / "agent_config.json").read_text())
            expected_distribution = dict(class_name="HeteroscedasticGaussianDistribution", init_std=.5,
                                         std_type="log", std_range=[.003,.8])
            if agent["actor"]["distribution_cfg"] != expected_distribution:
                raise ValueError("Sampling diagnostic requires the inspected native distribution contract")
            noise_log_std = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]["mlp.log_std"]
            public_task["evaluation_counterfactual"] = dict(kind="sampled_recovery_leg_noise_v1",
                variants=noise_variants, sample_seed=args.recovery_noise_seed,
                distribution=expected_distribution, learned_parameters_changed=False,
                deterministic_mean_changed=False, eligible_for_mean_qualification=False)
        training.write_json(task_path, public_task)
        reference_mean = None
        if args.reference_checkpoint is not None:
            args.reference_checkpoint = args.reference_checkpoint.resolve()
            reference_mean, _ = load_verified_mean(args.reference_checkpoint)
        state = {"infos": verified_info}
        deployed_mean = None
        if deployed_sha is not None:
            from wheeled_algo.onnx_actor_import import recover_actor
            original, _ = recover_actor(args.deployed_onnx)

            class DeployedMean(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.original = original
                    self.register_buffer("bounds", torch.tensor([3., 3., 3., 3., 9., 9.]))

                def forward(self, obs):
                    action = self.original(obs)
                    return torch.maximum(torch.minimum(action, self.bounds), -self.bounds)

            deployed_mean = DeployedMean()
        models = []
        for name in args.variants:
            selected = deployed_mean if name == "deployed" else reference_mean if name == "reference" else verified_mean
            mean = deepcopy(selected).cuda()
            if args.standing_command_feedback_gain is not None:
                from wheeled_algo.evaluation_command_feedback import StandingCommandFeedbackParent
                estimator = None
                if args.standing_command_feedback_estimator == "rolling":
                    from wheeled_algo.rolling_velocity import RollingBaseVelocity
                    table_path = ROOT / "reports/flat_base_continuation_20261019/low_friction_candidate001/height_lookup.json"
                    if training.digest(table_path) != "57d6516792a40f65c84f66c36d72f14b039b4b61a9779a7b210fe42465befb69":
                        raise ValueError("Diagnostic rolling geometry table changed")
                    estimator = RollingBaseVelocity(json.loads(table_path.read_text()))
                mean.parent = StandingCommandFeedbackParent(mean.parent, args.standing_command_feedback_gain, estimator).cuda()
            if name == "parent":
                mean = mean.parent
            elif name == "legs_only":
                mean.quiet_alpha[4:].fill_(1.)
            elif name == "undamped":
                mean.quiet_alpha.fill_(1.)
            if task.get("standing_position_observation") is not None and (name in ("parent", "deployed")
                    or name == "reference" and not getattr(mean, "quiet_velocity_feedback_mode", None) == "standing_displacement"):
                from wheeled_algo.robust_getup import ZeroStandingPosition
                mean = ZeroStandingPosition(mean)
            if name in alpha_variants:
                from wheeled_algo.evaluation_quiet_output import QuietWheelOutputFilter
                mean = QuietWheelOutputFilter(mean, alpha_variants[name])
            if name in roll_variants:
                from wheeled_algo.evaluation_early_roll import EarlyRollEvaluation
                mean = EarlyRollEvaluation(mean, 20., roll_variants[name])
            if name in position_variants:
                from wheeled_algo.evaluation_position_filter import PositionReleasedWheelFilter
                mean = PositionReleasedWheelFilter(mean, position_variants[name])
            if name in gyro_variants:
                from wheeled_algo.evaluation_gyro_filter import GyroReleasedWheelFilter
                mean = GyroReleasedWheelFilter(mean, gyro_variants[name])
            if name in step_variants:
                from wheeled_algo.evaluation_recovery_step import RecoveryStepEvaluation
                mean = RecoveryStepEvaluation(mean, step_variants[name])
            if name in noise_variants:
                from wheeled_algo.evaluation_recovery_noise import SampledRecoveryEvaluation
                mean = SampledRecoveryEvaluation(mean, verified_info["settings"], noise_log_std,
                                                noise_variants[name], args.recovery_noise_seed).cuda()
            models.append(mean.eval())

        class BatchMean(nn.Module):
            def __init__(self):
                super().__init__()
                self.models = nn.ModuleList(models)

            def forward(self, obs):
                return torch.cat([self.models[index % len(self.models)](obs[index * per_variant:(index + 1) * per_variant])
                                  for index in range(len(self.models) * len(seeds))], 0)

        manifest = json.loads((ROOT / cfg["asset_directory"] / "manifest.json").read_text())
        env = RobustGetupEnv(cfg, manifest, load_chassis_control(ROOT / cfg["control_math_source"]), ROOT,
            stage_name="speed", num_envs=count, device="cuda:0", seed=args.seed, level=1., run_role="train",
            bank=config_args.bank, settings=task)
        env.actor_variant = "batched_" + "_".join(args.variants)
        env.evaluation_checkpoint = args.checkpoint
        env.evaluation_overrides = ({} if args.sensor_delay_policy_ticks is None else
                                   dict(sensor_delay_policy_ticks=args.sensor_delay_policy_ticks))
        if args.friction_value is not None:
            env.evaluation_overrides["fixed_ground_friction"] = args.friction_value
        if alpha_variants:
            env.evaluation_overrides["quiet_wheel_output_alphas"] = args.quiet_wheel_output_alphas
        if roll_variants:
            env.evaluation_overrides["early_roll_full_degrees"] = args.early_roll_full_degrees
        if position_variants:
            env.evaluation_overrides["quiet_wheel_position_release_bands"] = args.quiet_wheel_position_release_bands
        if gyro_variants:
            env.evaluation_overrides["quiet_wheel_gyro_release_bands" if args.quiet_wheel_gyro_release_bands is not None
                                     else "quiet_wheel_gyro_release_band"] = gyro_values
        if step_variants:
            env.evaluation_overrides["recovery_leg_steps"] = step_variants
        if noise_variants:
            env.evaluation_overrides["sampled_recovery_leg_noise"] = dict(
                fractions=noise_variants, sample_seed=args.recovery_noise_seed, deterministic_mean=False)
        if args.forward_speeds:
            env.evaluation_overrides["forward_speeds"] = args.forward_speeds
            env.evaluation_overrides["forward_yaws"] = args.forward_yaws
        if args.stationary_push_cases:
            env.evaluation_overrides["stationary_push_cases"] = True
            if args.stationary_push_scale != 1.:
                env.evaluation_overrides["stationary_push_scale"] = args.stationary_push_scale
            if args.stationary_push_scales is not None:
                env.evaluation_overrides["stationary_push_scales"] = args.stationary_push_scales
        if args.instant_sensor_group:
            env.evaluation_overrides["instant_sensor_group"] = args.instant_sensor_group
        if args.neutralize_bias:
            env.evaluation_overrides["neutralize_bias"] = args.neutralize_bias
        if args.standing_command_feedback_gain is not None:
            env.evaluation_overrides["standing_command_feedback"] = dict(
                gain=args.standing_command_feedback_gain, max_request_correction_m_s=.15,
                velocity_estimator=args.standing_command_feedback_estimator,
                velocity_proxy=".3 * (obs20 - obs21)" if args.standing_command_feedback_estimator == "wheel" else "sealed107_encoder_IMU_rolling_kinematics",
                location="frozen_locomotion_parent_only_original_outer_requests_and_gates_retained")
            if args.standing_command_feedback_estimator == "rolling":
                env.evaluation_overrides["standing_command_feedback"].update(
                    geometry_table_sha256="57d6516792a40f65c84f66c36d72f14b039b4b61a9779a7b210fe42465befb69",
                    model_spec_sha256="0ac82045ab584f20091538aa782d93ea545382dc701faca1f8ae92aeb7cd1997")
        groups = [(seed, name) for seed in seeds for name in args.variants]
        for index, (_, name) in enumerate(groups):
            if name == "raw_pd":
                env.paired_feedback_enabled[index * per_variant:(index + 1) * per_variant] = False
        cases = []
        for seed, name in groups:
            full = training.fixed_cases(env.bank, 35 * args.replicas, seed)
            if args.stationary_push_cases:
                cases.extend(fixed_stationary_push_cases(full, args.stationary_push_scales))
            elif args.forward_speeds:
                template = next(case for case in full if case["name"] == "forward")
                cases.extend(fixed_forward_cases(template, args.forward_speeds, args.replicas, args.forward_yaws))
            else:
                cases.extend(case for case in full if selected_names is None or case["name"] in selected_names)
        report = training.evaluate(env, BatchMean().cuda(), args.output / "all", torch,
                                   seed=args.seed, seconds=args.seconds, cases=cases)
        arrays = dict(np.load(args.output / "all/trajectory.npz"))
        for index, (seed, name) in enumerate(groups):
            section = slice(index * per_variant, (index + 1) * per_variant)
            seed_root = args.output / str(seed) if len(seeds) > 1 else args.output
            target = seed_root / name
            target.mkdir(parents=True)
            if len(seeds) > 1:
                training.write_json(seed_root / "progress.json", dict(status="writing", completed_updates=state["infos"]["completed_updates"], seed=seed))
            np.savez_compressed(target / "trajectory.npz", **{key: value[:, section] for key, value in arrays.items()})
            split = deepcopy(report)
            split["lanes"] = split["lanes"][section]
            for lane, row in enumerate(split["lanes"]):
                row["lane"] = lane
                row["variant"] = name
            split["actor_variant"] = name
            split["case_subset"] = args.case_names
            split["fixed_ground_friction"] = args.friction_value
            split["evaluation_overrides"] = env.evaluation_overrides
            if name in alpha_variants:
                split["quiet_wheel_output_alpha"] = alpha_variants[name]
            if roll_variants:
                split["early_roll_authority_degrees"] = [20., roll_variants[name]] if name in roll_variants else None
            if position_variants:
                split["quiet_wheel_position_release_band_m"] = position_variants[name]
                split["quiet_wheel_output_alpha_min"] = .25
            if gyro_variants:
                split["quiet_wheel_gyro_release_band_rad_s"] = gyro_variants[name]
                split["quiet_wheel_output_alpha_min"] = .25
            if step_variants:
                split["recovery_leg_step_override"] = step_variants[name]
            if noise_variants:
                split["sampled_policy_exposure_diagnostic"] = models[index].sampling_receipt()
            split["seed"] = seed
            split["paired_feedback_winding"] = name != "raw_pd"
            if name == "deployed":
                split["source_onnx_sha256"] = deployed_sha
                split["source_onnx"] = str(args.deployed_onnx.resolve())
                split["policy_output_postprocess"] = dict(raw_leg_clip=3., raw_wheel_clip=9.,
                    basis="inspected RMCS action/policy contract; no old recovery script in this baseline")
            if name == "reference":
                split["checkpoint_sha256"] = training.digest(args.reference_checkpoint)
                split["checkpoint"] = str(args.reference_checkpoint)
            split["quiet_alpha_override"] = ([.25]*4 + [1.]*2 if name == "legs_only" else [1.]*6 if name == "undamped" else None)
            split["trajectory_sha256"] = training.digest(target / "trajectory.npz")
            recovery = [row for row in split["lanes"] if row["recovery"]]
            motion = [row for row in split["lanes"] if not row["recovery"]]
            split.update(recovery_episodes=len(recovery), recovered=sum(row["recovered"] for row in recovery),
                recovery_and_motion_success=sum(row["followup_success"] and row["survived"] for row in recovery),
                motion_episodes=len(motion), motion_survived=sum(row["survived"] for row in motion),
                max_gap_m=max(row["max_gap_m"] for row in split["lanes"]),
                passive_motor_peak_nm=max(row["passive_motor_peak_nm"] for row in split["lanes"]),
                max_contact_impulse_ns=max(row["max_contact_impulse_ns"] for row in split["lanes"]))
            training.write_json(target / "report.json", split)
        if len(seeds) > 1:
            for seed in seeds:
                training.write_json(args.output / str(seed) / "progress.json", dict(status="completed",
                    completed_updates=state["infos"]["completed_updates"], seed=seed, optimizer_updates=0))
        training.write_json(args.output / "progress.json", dict(status="completed", completed_updates=state["infos"]["completed_updates"],
            checkpoint_sha256=checkpoint_sha, seeds=seeds, variants=args.variants, optimizer_updates=0,
            sources=sources, pairing="Same nominal settings/cases per replica; separate native environments"))
    except BaseException:
        # SimulationApp.close may exit the interpreter before Python prints an
        # unwinding exception. Preserve it before entering native shutdown.
        error = traceback.format_exc()
        training.write_json(args.output / "failure.json", dict(status="failed", error=error,
            checkpoint_sha256=checkpoint_sha, optimizer_updates=0))
        print(error, file=sys.stderr, flush=True)
        raise
    finally:
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()

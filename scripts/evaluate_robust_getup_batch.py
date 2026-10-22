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
import os
from pathlib import Path
import sys
import traceback

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def fixed_forward_cases(template, speeds, replicas):
    """Independent diagnostic lanes at each requested forward speed."""
    cases = []
    for _ in range(replicas):
        for speed in speeds:
            case = deepcopy(template)
            case.update(name=f"forward_speed_{speed:g}", command=[speed, 0., template["command"][2]])
            cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path,
                        help="Optional exact initial checkpoint for a no-PPO coordinate-change comparison")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=211131)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--replicas", type=int, default=2)
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
    parser.add_argument("--sensor-delay-policy-ticks", type=int, choices=[0, 1, 2],
                        help="Diagnostic counterfactual: override only policy observation delay after the matched reset")
    parser.add_argument("--instant-sensor-group", choices=["imu", "encoders", "leg_encoders", "wheel_velocity"],
                        help="Diagnostic: remove delay from only this sensor group; retain its noise and other delays")
    parser.add_argument("--variants", nargs="+", default=["parent", "robust", "legs_only", "undamped"],
                        choices=["parent", "robust", "legs_only", "undamped", "reference", "raw_pd", "paired_pd", "deployed"])
    args = parser.parse_args()
    seeds = args.seeds or [args.seed]
    args.seed = seeds[0]
    max_replicas = 64 if args.case_names or args.forward_speeds else 4
    if not 1 <= args.replicas <= max_replicas or len(set(args.variants)) != len(args.variants):
        parser.error("Use 1..4 full-suite replicas or 1..64 diagnostic-subset replicas, and unique variants")
    if not 0. <= args.domain_scale <= 1. or not 0. < args.friction_min <= .5:
        parser.error("Domain scale must be 0..1 and minimum friction in (0,.5]")
    if args.friction_value is not None and not 0. <= args.friction_value <= 1.:
        parser.error("Fixed ground friction must be in [0,1]")
    if args.forward_speeds and (len(set(args.forward_speeds)) != len(args.forward_speeds)
                               or not all(0. <= speed <= 3. for speed in args.forward_speeds)):
        parser.error("Use distinct diagnostic forward speeds in [0,3] m/s")
    if ("reference" in args.variants) != (args.reference_checkpoint is not None):
        parser.error("Use --reference-checkpoint together with the reference variant")
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
    case_count = len(args.forward_speeds) if args.forward_speeds else len(selected_names) if selected_names else 35
    per_variant = case_count * args.replicas
    count = per_variant * len(args.variants) * len(seeds)
    if len(seeds) > 2 or len(set(seeds)) != len(seeds) or count > 1024:
        parser.error("Use at most two distinct seeds and 1024 total native environments")
    checkpoint_sha = training.digest(args.checkpoint)
    config_args = argparse.Namespace(output=args.output, bank=ROOT / "reports/unified_robust_20261021/pose_bank002.json",
        no_domains=args.domain_scale == 0., domain_scale=args.domain_scale,
        friction_min=args.friction_min if args.friction_value is None else args.friction_value,
        friction_max=.5 if args.friction_value is None else args.friction_value,
        curriculum_updates=128, quiet_leg_alpha=.25, quiet_wheel_alpha=.65,
        num_envs=count, rollout_steps=128, mode="evaluate", friction="high", seed=args.seed,
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
        for key in ("canonical_recovery", "motion_residual_fraction", "cyclic_leg_targets", "leg_action_clip",
                    "recovery_leg_step", "smooth_knee_targets", "recovery_knee_center", "quiet_actor_alpha",
                    "quiet_velocity_feature_scale", "quiet_wheel_bound", "stance_aware_recovery", "recovery_shape_aware",
                    "quiet_high_common_alpha", "quiet_high_leg_prediction_s", "fast_forward_reference"):
            if key in verified_info["settings"]:
                task[key] = deepcopy(verified_info["settings"][key])
        task_path = args.output / "task_contract.json"
        public_task = json.loads(task_path.read_text())
        public_task["settings"] = task
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
            if name == "parent":
                mean = mean.parent
            elif name == "legs_only":
                mean.quiet_alpha[4:].fill_(1.)
            elif name == "undamped":
                mean.quiet_alpha.fill_(1.)
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
        if args.forward_speeds:
            env.evaluation_overrides["forward_speeds"] = args.forward_speeds
        if args.instant_sensor_group:
            env.evaluation_overrides["instant_sensor_group"] = args.instant_sensor_group
        groups = [(seed, name) for seed in seeds for name in args.variants]
        for index, (_, name) in enumerate(groups):
            if name == "raw_pd":
                env.paired_feedback_enabled[index * per_variant:(index + 1) * per_variant] = False
        cases = []
        for seed, name in groups:
            full = training.fixed_cases(env.bank, 35 * args.replicas, seed)
            if args.forward_speeds:
                template = next(case for case in full if case["name"] == "forward")
                cases.extend(fixed_forward_cases(template, args.forward_speeds, args.replicas))
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

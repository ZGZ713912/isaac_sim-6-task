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
    from wheeled_tasks.chassis.unified_getup import settings
    cfg = json.loads(BASE_CONFIG.read_text())
    design = json.loads(Path(cfg["design_preflight"]["file"]).read_text())
    row = design["stages"]["speed"]["roles"]["train"]
    task = settings()
    # The fixed physics/control source remains pinned. The task overlay is a
    # separately frozen public contract, not a claim of old-task equivalence.
    changes = dict(target_num_envs=args.num_envs, num_steps_per_env=args.rollout_steps,
        ppo_minibatch_samples=args.num_envs * args.rollout_steps // 4,
        episode_seconds=task["episode_seconds"], record_diagnostics=args.mode != "train")
    if args.friction == "low":
        material = deepcopy(cfg["contact_material"])
        material["ground_static_friction_range"] = [.2, .5]
        material["ground_dynamic_friction_range"] = [.2, .5]
    else:
        material = deepcopy(cfg["contact_material"])
        material["ground_static_friction_range"] = [.5, .5]
        material["ground_dynamic_friction_range"] = [.5, .5]
    changes["contact_material"] = material
    # This base contract describes mechanism construction. Task behavior lives
    # in UnifiedGetupEnv and is independently declared in task_contract.json.
    for key, value in changes.items():
        cfg[key] = deepcopy(value)
        if key in row["contract"]:
            row["contract"][key] = deepcopy(value)
    row["runtime"]["seed"] = args.seed
    row["runtime"]["contact_material"] = deepcopy(material)
    row["invocation"] = dict(num_envs=args.num_envs, seed=dict(rule="fixed", value=args.seed))
    design["scope"] = "unified_getup_mechanism_construction"
    design["review_basis"] = dict(base_contract=str(BASE_CONFIG), base_contract_sha256=digest(BASE_CONFIG),
        task_contract=str(args.output / "task_contract.json"))
    write_json(args.output / "config/design.json", design)
    cfg["design_preflight"] = dict(file=str((args.output / "config/design.json").resolve()),
        sha256=digest(args.output / "config/design.json"), stage="speed")
    write_json(args.output / "config/contract.json", cfg)
    write_json(args.output / "task_contract.json", dict(settings=task, base_checkpoint=str(PARENT),
        base_checkpoint_sha256=digest(PARENT), bank=str(args.bank), bank_sha256=digest(args.bank),
        motor_action_clip_policy_order=[13., 13., 13., 13., 9., 9.],
        motor_position_scale=.25, wheel_velocity_scale=10.,
        policy_order=["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"],
        ordinary_fall_termination=False, temporary_body_support_allowed=True,
        scripted_recovery=False, internal_state_writes="reset_only",
        curriculum="easy poses and higher friction first; permanent hard-pose and arc exposure",
        locomotion_rehearsal_fraction=.4, mixed_fraction_within_motion=.5,
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
            indices = [i for i in pool[family] if ("50_85" in poses[i]["leg_shape"] or "85_50" in poses[i]["leg_shape"]) == (suffix == "asymmetric")]
            templates.append(dict(name=f"{family}_{suffix}", recovery=True,
                pose_index=rng.choice(indices), command=[.5, 1. if len(templates) % 2 else -1., .305]))
    for name, command in (("stand", [0., 0., .305]), ("forward", [1., 0., .305]),
        ("backward", [-.75, 0., .305]), ("spin_left", [0., 1., .305]), ("spin_right", [0., -1., .305]),
        ("arc_forward_left", [1., 1., .305]), ("arc_forward_right", [1., -1., .305]),
        ("arc_reverse_left", [-.75, 1., .305]), ("arc_reverse_right", [-.75, -1., .305]),
        ("arc_fast_left", [3., 4., .305]), ("arc_fast_right", [3., -4., .305])):
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


def evaluate(env, mean, output, torch, *, seed, seconds=12.):
    import numpy as np
    from wheeled_tasks.chassis.unified_getup import FAMILIES
    env.fixed_cases = fixed_cases(env.bank, env.num_envs, seed, getattr(env, "ablation_variants", False))
    if getattr(env, "ablation_variants", False):
        from wheeled_algo.unified_getup_ablation import RecoveryAblations
        mean = RecoveryAblations(mean, [case["policy_variant"] for case in env.fixed_cases]).to(env.device)
    env.generator.manual_seed(seed)
    env.reset(torch.arange(env.num_envs, device=env.device))
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
    causes = {key: ~alive.clone() for key in ("mechanical", "closure_gap", "impact_bad")}
    vx_sum, yaw_sum, vx_abs, yaw_abs, frames = [torch.zeros(env.num_envs, device=env.device) for _ in range(5)]
    start = time.monotonic()
    mean.eval()
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
            if tick % 5 == 0:
                keep = ("time", "gravity", "height", "commands", "requested_command", "velocity", "omega",
                        "actions", "motor_effort", "motor_velocity", "gap", "impact", "impulse", "passive", "stable",
                        "done", "terminated", "enable_event", "settled_family", "recovered", "followup_success",
                        "mechanical", "closure_gap", "impact_bad", "local_position", "motor_position", "wheel_contact")
                keep += ("pushing", "in_motion_fall", "recovered_after_push")
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
        lanes.append(dict(case, lane=i, survived=bool(completed[i] & ~terminated[i]), terminated=bool(terminated[i]),
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
    report = dict(schema="unified-getup-native-evaluation-v1", seed=seed, seconds=seconds,
        lanes=lanes, elapsed_s=time.monotonic() - start, policy_hz=200, physics_hz=1000,
        recovery_episodes=len(recovery), recovered=sum(v["recovered"] for v in recovery),
        recovery_and_motion_success=sum(v["followup_success"] and v["survived"] for v in recovery),
        motion_episodes=len(motion), motion_survived=sum(v["survived"] for v in motion),
        passive_motor_peak_nm=float(passive_torque.max()), max_gap_m=float(max_gap.max()),
        max_contact_impulse_ns=float(max_impulse.max()),
        material=env.cfg["contact_material"], trajectory_sha256=digest(output / "trajectory.npz"),
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
    parser.add_argument("--bank", type=Path, default=ROOT / "reports/unified_getup_20261021/pose_bank001.json")
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate"), default="train")
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=211031)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--friction", choices=("high", "low"), default="high")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pre-eval", action="store_true")
    parser.add_argument("--eval-seconds", type=float, default=12.)
    parser.add_argument("--ablation-compare", action="store_true")
    parser.add_argument("--curriculum-updates", type=int, default=128)
    args = parser.parse_args()
    if not (1 <= args.updates <= 256 and 32 <= args.num_envs <= 1024 and 64 <= args.rollout_steps <= 512
            and args.num_envs % 4 == 0 and 2 <= args.eval_seconds <= 12):
        parser.error("Use bounded pilots: 1..256 updates, 32..1024 environments, 64..512 rollout steps")
    args.output, args.bank = args.output.resolve(), args.bank.resolve()
    if args.ablation_compare and args.mode != "evaluate":
        parser.error("Read-only ablations require --mode evaluate")
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
    sources = [ROOT / "scripts/train_unified_getup.py", ROOT / "scripts/build_unified_getup_bank.py",
        ROOT / "src/wheeled_tasks/chassis/unified_getup.py", ROOT / "src/wheeled_tasks/chassis/unified_getup_env.py",
        ROOT / "src/wheeled_algo/unified_getup.py", ROOT / "src/wheeled_tasks/chassis/env.py",
        ROOT / "src/wheeled_tasks/chassis/v5_control.py", ROOT / "src/wheeled_tasks/chassis/paired_leg_targets.py",
        ROOT / "src/wheeled_algo/recovery_rollout.py", ROOT / "src/wheeled_algo/unified_getup_ablation.py"]
    hashes = {}
    for p in sources:
        relative = p.relative_to(ROOT)
        hashes[str(relative)] = digest(p)
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    report = dict(schema="unified-getup-pilot-v1", status="starting", pid=os.getpid(),
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
        from wheeled_tasks.chassis.unified_getup_env import UnifiedGetupEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_algo.unified_getup import UnifiedGetupActor
        from wheeled_algo.exploration_init import initialize_policy_std, install_axis_std_logging
        from wheeled_algo.recovery_rollout import RecoveryRolloutMask
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        manifest = json.loads((ROOT / cfg["asset_directory"] / "manifest.json").read_text())
        control = load_chassis_control(ROOT / cfg["control_math_source"])
        env = UnifiedGetupEnv(cfg, manifest, control, ROOT, stage_name="speed", num_envs=args.num_envs,
            device=args.device, seed=args.seed, level=1., run_role="train", bank=args.bank, settings=task)
        mean = UnifiedGetupActor(load_mean(torch)).to(args.device)
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
            runner_cfg["algorithm"].update(class_name="PPO", learning_rate=3e-4, schedule="fixed",
                entropy_coef=.002, value_loss_coef=1., num_learning_epochs=3, num_mini_batches=4,
                gamma=.998, lam=.99, use_clipped_value_loss=False)
            runner_cfg["actor"]["distribution_cfg"]["init_std"] = .5
            write_json(args.output / "agent_config.json", runner_cfg)
            runner = OnPolicyRunner(env, runner_cfg, log_dir=str(args.output), device=args.device)
            runner.alg.actor.mlp = mean
            runner.alg.optimizer = torch.optim.Adam([
                dict(params=[p for p in runner.alg.actor.parameters() if p.requires_grad], lr=3e-4, name="actor"),
                dict(params=runner.alg.critic.parameters(), lr=1e-3, name="critic")])
            initialize_policy_std(runner.alg, [.5, .5, .5, .5, .5, .5])
            parent_updates = 0
            if args.resume:
                state = torch.load(args.resume, map_location=args.device, weights_only=False)
                if state["infos"]["schema"] != "unified-getup-checkpoint-v1" or state["infos"]["bank_sha256"] != digest(args.bank):
                    raise ValueError("Resume checkpoint or mechanism bank differs")
                runner.alg.load(state, load_cfg=None, strict=True)
                parent_updates = state["infos"]["completed_updates"]
                env.generator.set_state(state["infos"]["environment_rng"].cpu())
                torch.set_rng_state(state["infos"]["torch_rng"].cpu())
                if args.device.startswith("cuda"):
                    torch.cuda.set_rng_state_all([v.cpu() for v in state["infos"]["cuda_rng"]])
                report["restored_from"] = str(args.resume.resolve())
                report["restored_sha256"] = digest(args.resume)
            report["parent_updates"] = parent_updates
            report["completed_updates"] = parent_updates
            env.actor_updates = parent_updates
            install_axis_std_logging(runner.logger)
            mask = RecoveryRolloutMask(runner.alg, num_envs=args.num_envs)
            original_update, original_save = runner.alg.update, runner.save

            def save(path, infos=None):
                info = dict(schema="unified-getup-checkpoint-v1", completed_updates=report["completed_updates"],
                    base_checkpoint=str(PARENT), base_checkpoint_sha256=digest(PARENT),
                    bank_sha256=digest(args.bank), task_contract_sha256=report["task_contract_sha256"],
                    source_sha256=hashes, environment_rng=env.generator.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
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
            if args.pre_eval:
                env.evaluation_checkpoint = args.resume
                report["baseline_evaluation"] = evaluate(env, mean, args.output / "baseline_eval", torch,
                                                         seed=args.seed + 100, seconds=args.eval_seconds)
            if args.mode == "train":
                runner.current_learning_iteration = parent_updates
                runner.learn(args.updates, init_at_random_ep_len=False)
                save(str(args.output / "model_final.pt"))
                env.evaluation_checkpoint = args.output / "model_final.pt"
            else:
                env.evaluation_checkpoint = args.resume
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

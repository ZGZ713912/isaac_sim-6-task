#!/usr/bin/env python3
"""Bounded first-episode self-righting replay of a fixed 50 Hz flat actor.

This produces material/control/behavior evidence and consumes zero PPO updates.
It grants no deployment or policy-only self-righting qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

POSES = ("upright", "crouched", "front_down_90", "back_down_90",
    "front_down_left_forward_right_back_90", "back_down_left_forward_right_back_90",
    "prone_belly_legs_above", "prone_back_legs_above")


def materialize(args):
    from wheeled_tasks.chassis.flat_base_continuation import write_bundle, digest, encode
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.slow_recovery import recovery_settings, runtime_expectation
    from train_chassis import preflight

    config, seed = write_bundle(ROOT, args.output / "config", 32, evaluation=True, plan_file=args.plan)
    design_path = args.output / "config/design.json"
    design = json.loads(design_path.read_text())
    row = design["stages"]["speed"]["roles"]["quick"]
    source = deepcopy(config["evaluation"])
    cases = []
    for pose in POSES:
        case = deepcopy(source["cases"][1])
        case.update(name="recovery_" + pose, command=[0., 0., .305], terrain="flat", task="survive",
            episode_seconds=14., stationary=False, recovery_case=True, reset_seed_key="recovery_" + pose,
            skill=dict(kind="recovery_handoff", command=[0., 0., .305], mode=0,
                sample_amplitude=False, recovery_pose=pose, recovery_followup_command=[0., 0., .305]))
        cases.append(case)
    settings = recovery_settings(ROOT, args.profiles, args.lookup, slow=args.variant == "slow")
    evaluation = deepcopy(source)
    evaluation.update(cases=cases, episode_seconds=14., diagnostic_only=True,
        protocol_id="flat-self-righting-engineering-" + args.variant + "-v1")
    for mapping in (config, row["contract"]):
        mapping.update(evaluation=deepcopy(evaluation), episode_seconds=14.,
            recovery_training=deepcopy(settings), record_diagnostics=True,
            scene_groups=[dict(name=case["name"], fraction=1 / len(cases), terrain=["flat"]) for case in cases],
            skill_specs={case["name"]: deepcopy(case["skill"]) for case in cases})
    runtime = row["runtime"]
    runtime["modules"]["recovery_training"] = True
    runtime.update(recovery_pose_curriculum=None, recovery_complete_started_blend=False)
    if args.variant == "slow":
        runtime["slow_recovery"] = runtime_expectation(settings)
    design["scope"] = "flat_actor_first_episode_self_righting_no_training_or_qualification"
    design_path.write_bytes(encode(design))
    config["design_preflight"]["sha256"] = digest(design_path)
    contract_path = args.output / "config/contract.json"
    contract_path.write_bytes(encode(config))
    config, manifest = preflight(contract_path)
    control = load_chassis_control(ROOT / config["control_math_source"])
    static = design_preflight(config, control, ROOT, stage="speed", role="quick", num_envs=32, seed=seed)
    return config, manifest, control, seed, static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--lookup", required=True)
    parser.add_argument("--variant", choices=("legacy", "slow"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    from wheeled_tasks.chassis.flat_base_continuation import digest, encode, read_plan
    plan = read_plan(ROOT, args.plan)
    config, manifest, control, seed, static = materialize(args)
    files = [*sorted((ROOT / "src/wheeled_tasks/chassis").glob("*.py")), Path(__file__),
        *sorted((ROOT / "src/wheeled_algo").glob("*.py"))]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    report = dict(schema="flat-self-righting-engineering-replay-v1", status="static_preflight_passed",
        recorded_at=datetime.now(timezone.utc).isoformat(), variant=args.variant, seed=seed,
        checkpoint_sha256=digest(args.checkpoint), plan_sha256=digest(ROOT / args.plan),
        source_sha256=hashes, static_preflight=static, new_optimizer_updates=0,
        qualification_eligible=False, hardware_motion_started=False)
    save = lambda: (args.output / "report.json").write_bytes(encode(report))
    save()
    if args.preflight_only:
        return 0
    launcher, env = None, None
    started = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device="cuda:0", enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import numpy as np
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_algo.command_residual import mean_from_checkpoint
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if (state["infos"]["asset_manifest_sha256"] != plan["asset_manifest_sha256"]
                or state["infos"]["flat_base_source_onnx_sha256"] != plan["source_onnx_sha256"]):
            raise ValueError("Fixed actor and recovery mechanics differ")
        actor = mean_from_checkpoint(state, ROOT / plan["actor_bundle"] / "source.onnx").to("cuda:0").eval()
        env = FixedCaseEnv(config, manifest, control, ROOT, stage_name="speed", num_envs=32,
            device="cuda:0", seed=seed, level=1., run_role="quick")
        report["startup"] = env.startup_report
        report["status"] = "running"
        save()
        obs = env.reset_suite()
        alive = torch.ones(32, dtype=torch.bool, device="cuda:0")
        keys = ("gravity", "omega", "height", "velocity", "position", "gap", "motor_position", "motor_velocity",
            "motor_effort", "done", "success", "recovery_phase", "recovery_route", "recovery_failed",
            "recovery_failure_code", "recovery_pure_rl", "recovery_true_success", "recovery_motion_released",
            "recovery_rl_stable_s", "recovery_followup_s", "recovery_leg_target_error_rad",
            "recovery_wheel_assist_active", "recovery_wheel_velocity_target", "recovery_observed_height")
        traces = {key: [] for key in ("active", *keys)}
        reason_traces = {}
        with torch.inference_mode():
            for tick in range(env.max_episode_length + 1):
                context = obs["policy"][:, 28:35]
                if not torch.equal(context, context.new_tensor(plan["flat_context"]).expand_as(context)):
                    raise RuntimeError("Flat actor context changed in recovery replay")
                obs, _, done, extras = env.step(actor(obs["policy"]))
                diagnostic = extras["diagnostics"]
                traces["active"].append(alive.cpu().numpy().copy())
                for key in keys:
                    traces[key].append(diagnostic[key].cpu().numpy().copy())
                for key, values in diagnostic["reasons"].items():
                    reason_traces.setdefault(key, []).append(values.cpu().numpy().copy())
                alive &= ~done.bool()
                if tick % 100 == 0:
                    print("RECOVERY_REPLAY", args.variant, tick, int(alive.sum()),
                        round(time.monotonic() - started, 2), flush=True)
                if not alive.any():
                    break
        arrays = {key: np.asarray(values) for key, values in traces.items()}
        reasons = {key: np.asarray(values) for key, values in reason_traces.items()}
        np.savez_compressed(args.output / "traces.npz", **arrays,
            **{"reason_" + key: value for key, value in reasons.items()}, groups=np.asarray(env.scene_groups))
        cases = {}
        for group in sorted(set(env.scene_groups)):
            ids = np.flatnonzero(np.asarray(env.scene_groups) == group)
            mask = arrays["active"][:, ids]
            row = dict(requested_episodes=len(ids),
                successful_recovery_episodes=int((arrays["recovery_true_success"][:, ids] & mask).any(0).sum()),
                pure_rl_episodes=int((arrays["recovery_pure_rl"][:, ids] & mask).any(0).sum()),
                motion_released_episodes=int((arrays["recovery_motion_released"][:, ids] & mask).any(0).sum()),
                stable_hold_peak_s=float(np.where(mask, arrays["recovery_rl_stable_s"][:, ids], 0.).max()),
                closure_gap_max_m=float(np.where(mask, arrays["gap"][:, ids], 0.).max()),
                wheel_assist_active_frames=int((arrays["recovery_wheel_assist_active"][:, ids] & mask).sum()),
                termination_reasons={key: int((value[:, ids] & mask).any(0).sum()) for key, value in reasons.items()},
                visited_routes=sorted(set(arrays["recovery_route"][:, ids][mask].tolist())),
                visited_phases=sorted(set(arrays["recovery_phase"][:, ids][mask].tolist())))
            cases[group] = row
        namespace = "RecoveryEngineering/" + args.variant + "/low_friction"
        with SummaryWriter(str(args.output)) as writer:
            for name, row in cases.items():
                for key in ("successful_recovery_episodes", "pure_rl_episodes", "stable_hold_peak_s",
                            "closure_gap_max_m", "wheel_assist_active_frames"):
                    writer.add_scalar(f"{namespace}/{name}/{key}", row[key], state["infos"]["successful_updates_total"])
        if any(digest(ROOT / name) != value for name, value in hashes.items()):
            raise RuntimeError("Execution sources changed during fixed replay")
        report.update(status="evaluated", cases=cases, elapsed_s=time.monotonic() - started,
            source_unchanged=True, policy_steps=tick + 1)
        print("RECOVERY_RESULT", json.dumps(cases), flush=True)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    return 0 if report["status"] == "evaluated" else 1


if __name__ == "__main__":
    raise SystemExit(main())

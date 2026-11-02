#!/usr/bin/env python3
"""Bounded, zero-PPO native diagnosis of a saved standard-prone policy.

Keep first-episode action/state/failure traces and the first wheel-reference
fault inputs. This process never controls, pauses or writes the live learner.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--num-envs", type=int, default=80)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-plan", type=Path,
        help="Explicit zero-PPO intervention only; does not qualify the saved policy under a new contract")
    args = parser.parse_args()
    if not 80 <= args.num_envs <= 256 or not 1 <= args.steps <= 2400:
        parser.error("Use80..256 native environments and1..2400 policy steps")
    args.output.mkdir(parents=True, exist_ok=False)
    from train_scratch_standing import digest, write_json
    report = dict(schema="standard-getup-diagnosis-v1", status="starting", new_PPO=0,
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
                  hardware_ready=False, behavior_qualified=False, wheel_fault_events=[])
    write_json(args.output / "report.json", report)
    launcher = env = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False,
            kit_args="--/plugins/carb.tasking.plugin/threadCount=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4"))
        import numpy as np
        import torch
        import train_scratch_standing as base
        from train_prone_motion import configuration
        from play_prone_motion import build_actor
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
        base.torch = torch
        torch.set_num_threads(4)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        run = args.checkpoint.parent
        plan = json.loads((run / "curriculum_plan.resolved.json").read_text())
        if args.candidate_plan is not None:
            candidate = json.loads(args.candidate_plan.read_text())
            for key in ("standing_odometry", "full_vehicle", "pose_bank", "mechanism", "wheel_model", "timing"):
                if candidate[key] != plan[key]:
                    raise ValueError("Counterfactual must retain actor ABI and plant: "+key)
            plan = candidate
            report.update(counterfactual=True, candidate_plan=str(args.candidate_plan.resolve()),
                          candidate_plan_sha256=digest(args.candidate_plan), policy_qualification=False)
        agent = json.loads((run / "agent_config.json").read_text())
        cfg, settings, bank = configuration(args.output / "native", plan, num_envs=args.num_envs,
            evaluation=True, profile="nominal", strength=0., device=args.device, seed=107)
        manifest = json.loads((ROOT / cfg["asset_directory"] / "manifest.json").read_text())
        env = ProneMotionEnv(cfg, manifest, load_chassis_control(ROOT / cfg["control_math_source"]), ROOT,
            stage_name="speed", num_envs=args.num_envs, device=args.device, seed=107, level=1.,
            run_role="train", bank=bank, settings=settings)
        cases = []
        for recovery in (True, False):
            for heading in (0., math.pi/2., math.pi, 3.*math.pi/2.):
                pose = env.prone_index if recovery else env.nominal_pose_index
                h = env.poses[pose]["root_position_m"][2]
                cases.append(dict(role="prone" if recovery else "stand", recovery=recovery,
                    pose_index=pose, heading_rad=heading, birth_height=h, pitch_deg=0.,
                    target=[0., 0., .305 if recovery else .32], command=[0., 0., h]))
        env.evaluation_cases = [deepcopy(cases[i % len(cases)]) for i in range(args.num_envs)]
        env.reset(torch.arange(args.num_envs, device=args.device))
        observer = env.local_standing_odometry
        original = observer.local_wheel_angles

        def wheel_angles(frame, wheel, enabled, tick):
            old_fault = observer.source_fault.clone()
            previous, previous_speed, previous_tick = (observer.source_last.clone(),
                observer.source_speed.clone(), observer.source_tick.clone())
            result = original(frame, wheel, enabled, tick)
            new = (observer.source_fault & ~old_fault).nonzero().flatten().tolist()
            for lane in new:
                if lane >= len(cases) or len(report["wheel_fault_events"]) >= 24:
                    continue
                speed = frame[lane, 20:22] * 10.
                seconds = max(1, tick-int(previous_tick[lane])) * observer.dt
                bound = (observer.cfg["wheel_increment_margin_rad"]
                    + observer.cfg["wheel_increment_speed_allowance"] * seconds
                    * torch.maximum(speed.abs(), previous_speed[lane].abs()))
                report["wheel_fault_events"].append(dict(lane=lane, tick=tick,
                    previous_tick=int(previous_tick[lane]), interval_s=seconds,
                    wheel=wheel[lane].tolist(), previous_wheel=previous[lane].tolist(),
                    delta=(wheel[lane]-previous[lane]).tolist(), speed=speed.tolist(),
                    previous_speed=previous_speed[lane].tolist(), bound=bound.tolist()))
            return result

        observer.local_wheel_angles = wheel_angles
        obs = env.get_observations()
        actor = build_actor(obs, dict(agent=agent, state=state), args.device)
        count = len(cases)
        alive = torch.ones(count, dtype=torch.bool, device=args.device)
        arrays = {}
        with torch.inference_mode():
            for step in range(args.steps):
                action = actor(obs)
                obs, reward, done, extra = env.step(action)
                d = extra["diagnostics"]
                values = {key: d[key][:count] for key in (
                    "time", "height", "gravity", "velocity", "omega", "motor_position", "motor_velocity",
                    "actions", "gap", "impact", "impulse", "passive", "stable", "terminated",
                    "mechanical", "closure_gap", "closure_gap_exceeded", "impact_bad", "boundary_bad", "local_position",
                    "recovery_completed_event", "nonwheel_force_peak_n", "wheel_contact")}
                values.update(valid=alive.clone(), reward=reward[:count],
                    odometry_fault=observer.source_fault[:count].clone(),
                    odometry_valid=observer.valid[:count].clone())
                values.update({"failure_"+key: value[:count] for key,value in d["task_failures"].items()})
                values.update({"reward_"+key: value[:count] for key,value in d["task_reward_terms"].items()})
                for key,value in values.items():
                    arrays.setdefault(key, []).append(value.cpu().numpy().copy())
                alive &= ~done[:count]
                if step % 200 == 0:
                    print(f"step={step} alive={int(alive.sum())} first_fault_events={len(report['wheel_fault_events'])}", flush=True)
                if not bool(alive.any()):
                    break
        arrays = {key: np.stack(value) for key,value in arrays.items()}
        np.savez_compressed(args.output / "trace.npz", **arrays)
        rows = []
        for lane,case in enumerate(cases):
            mask = arrays["valid"][:,lane]
            a = {key:value[mask,lane] for key,value in arrays.items()}
            row = dict(case, samples=int(mask.sum()), completed=bool(a["recovery_completed_event"].any()),
                height_start_m=float(a["height"][0]), height_peak_m=float(a["height"].max()),
                final_height_m=float(a["height"][-1]), closure_peak_m=float(a["gap"].max()),
                closure_record_only=env.getup_cfg.get("closure_gap_record_only",False),
                closure_reference_exceeded_fraction=float(a["closure_gap_exceeded"].mean()),
                terminated=bool(a["terminated"].any()),
                failure_reasons=[key for key in a if key.startswith("failure_") and a[key].any()],
                native_failure_reasons=[key for key in ("mechanical", "closure_gap", "impact_bad", "boundary_bad") if a[key].any()],
                end_time_s=float(a["time"][-1]), odometry_fault_fraction=float(a["odometry_fault"].mean()),
                odometry_valid_fraction=float(a["odometry_valid"].mean()))
            rows.append(row)
        report.update(status="completed", cases=rows, trace_sha256=digest(args.output / "trace.npz"),
            physics_dt=env.dt, policy_dt=env.policy_dt, solver=plan.get("solver_iterations"),
            startup=env.startup_report.get("control_timing"))
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        write_json(args.output / "report.json", report)
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()

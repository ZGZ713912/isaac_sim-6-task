#!/usr/bin/env python3
"""Independent CPU PhysX wiring probe; never constructs a learner or optimizer."""
from datetime import datetime, timezone
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=80)
    args = parser.parse_args()
    if args.num_envs < 20 or args.num_envs % 20:
        parser.error("Use a multiple of20 environments to include all roles")
    args.output.mkdir(parents=True, exist_ok=False)
    from train_prone_motion import configuration
    from train_scratch_standing import write_json, digest
    plan = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4.json").read_text())
    report = dict(schema="leg-friction-native-probe-v1", status="starting", device="cpu",
        started_utc=datetime.now(timezone.utc).isoformat(), num_envs=args.num_envs,
        checkpoint_reads=[], new_PPO=0, optimizer_attempts=0, native_probes=[],
        behavior_qualified=False, hardware_motion=False,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),
            ROOT/"src/wheeled_tasks/chassis/env.py", ROOT/"src/wheeled_tasks/chassis/robust_getup_env.py",
            ROOT/"src/wheeled_tasks/chassis/leg_motor_friction.py",
            ROOT/"scripts/evaluate_prone_motion.py", ROOT/"contracts/wheel_leg_prone_motion_v4.json")})
    path = args.output/"report.json"
    write_json(path, report)
    launcher = env = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device="cpu", enable_cameras=False))
        import torch
        import train_scratch_standing as base
        base.torch = torch
        from isaaclab.sim import SimulationContext
        from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from evaluate_prone_motion import probe
        torch.set_num_threads(4)
        def forbidden_load(*a, **kw):
            report["checkpoint_reads"].append(str(a[0] if a else kw.get("f")))
            raise RuntimeError("Checkpoint reads forbidden in the zero-PPO friction probe")
        torch.load = forbidden_load
        for index, (profile, strength) in enumerate((("nominal", 0.), ("requested", 1.))):
            directory = args.output/profile
            cfg, settings, bank = configuration(directory, plan, profile=profile, strength=strength,
                num_envs=args.num_envs, device="cpu", performance_backend="eager", seed=plan["seed"]+31*index)
            manifest = json.loads((ROOT/cfg["asset_directory"]/"manifest.json").read_text())
            env = ProneMotionEnv(cfg, manifest, load_chassis_control(ROOT/cfg["control_math_source"]), ROOT,
                stage_name="speed", num_envs=args.num_envs, device="cpu", seed=plan["seed"]+31*index,
                level=1., run_role="train", bank=bank, settings=settings)
            env.actor_updates = 0
            env.reset(torch.arange(env.num_envs))
            write_json(directory/"startup.json", env.startup_report)
            result = probe(env, plan)
            result.update(profile=profile, strength=strength, physical_steps_per_lane=1500,
                leg_motor_friction=env.startup_report["leg_motor_friction"])
            write_json(directory/"native.json", result)
            report["native_probes"].append(result)
            write_json(path, report)
            print(json.dumps(dict(profile=profile, checks=result["checks"], passed=result["all_checks_pass"])), flush=True)
            env.close()
            env = None
            SimulationContext.clear_instance()
            gc.collect()
        report.update(status="completed", passed=all(r["all_checks_pass"] for r in report["native_probes"]))
        if not report["passed"]:
            raise RuntimeError("Native leg friction probe failed; retain all negative evidence")
    except Exception:
        report.update(status="failed", passed=False, error=traceback.format_exc())
        raise
    finally:
        write_json(path, report)
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()

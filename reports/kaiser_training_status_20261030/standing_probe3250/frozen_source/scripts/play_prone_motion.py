#!/usr/bin/env python3
"""Read-only preview of a prone-motion checkpoint with explicit scene commands."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from train_prone_motion import configuration, learner_config
from train_scratch_standing import digest, write_json
from wheeled_tasks.chassis.prone_motion_cli import add_common, output_path, validate_device, TASKS
from wheeled_tasks.chassis import prone_motion as task


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    add_common(result, training=False)
    result.add_argument("--checkpoint", type=Path, required=True, help="Read-only .pt; never used to resume training")
    result.add_argument("--run", type=Path, help="Run containing agent_config.json and pose_bank.json; default: checkpoint parent")
    result.add_argument("--scene", choices=task.ROLES, default="prone")
    result.add_argument("--height", type=float, default=.305, help="Target height in metres (.23..42)")
    result.add_argument("--birth_height", "--birth-height", type=float, default=.305, choices=(.23, .305, .42))
    result.add_argument("--vx", type=float, help="Signed forward target m/s; default follows --scene")
    result.add_argument("--yaw_rate", "--yaw-rate", type=float, help="Signed body yaw target rad/s; default follows --scene")
    result.add_argument("--seconds", type=float, help="Simulated preview duration; default: original 70s horizon")
    result.add_argument("--loop", action="store_true", help="Repeat the selected scene; resets are counted in the receipt")
    result.add_argument("--real_time", "--real-time", action=argparse.BooleanOptionalAction, default=True,
                        help="Pace GUI to simulated time; headless runs as fast as possible")
    return result


def load_bundle(args):
    import torch
    validate_device(args.device)
    if args.plan is not None:
        raise ValueError("Play uses the checkpoint's resolved plan; --plan cannot replace its mechanism")
    if type(args.num_envs) is not int or not 1 <= args.num_envs <= 256:
        raise ValueError("Play --num_envs must be in 1..256 for the isolated native worlds")
    if not math.isfinite(args.height) or not .23 <= args.height <= .42:
        raise ValueError("--height must be in [.23,.42] metres")
    for name, limit in (("vx", 2.), ("yaw_rate", math.tau*100/60)):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or abs(value) > limit):
            raise ValueError(f"--{name} must be finite and within +/-{limit:g}")
    if not math.isfinite(args.dr_strength) or not 0 <= args.dr_strength <= 1:
        raise ValueError("--dr_strength must be in [0,1]")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    run = (args.run or checkpoint.parent).expanduser().resolve(strict=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    info = state["infos"]
    if info.get("schema") != "prone-motion-checkpoint-v1" or info.get("training_origin") != "scratch":
        raise ValueError("Need a native prone-motion checkpoint, with its fresh-training provenance")
    plan_file = run/"curriculum_plan.resolved.json"
    if not plan_file.is_file():
        plan_file = run/"curriculum_plan.input.json"
    plan = task.validate(info.get("resolved_plan", json.loads(plan_file.read_text())))
    if task.validate(json.loads(plan_file.read_text())) != plan:
        raise ValueError("Run plan differs from the checkpoint's resolved plan")
    if plan["schema"] != "wheel-leg-prone-motion-v"+args.task.rsplit("v", 1)[1]:
        raise ValueError("--task differs from checkpoint version; select its original task")
    completed = state["iter"]
    if type(completed) is not int or not 0 <= completed <= plan["max_updates"] or info["completed_updates"] != completed:
        raise ValueError("Checkpoint iteration accounting is inconsistent")
    bank = run/"pose_bank.json"
    if digest(bank) != info["bank_sha256"]:
        raise ValueError("Checkpoint pose bank SHA changed")
    agent = json.loads((run/"agent_config.json").read_text())
    expected = learner_config(plan, agent["device"])
    if agent != expected:
        raise ValueError("Saved learner configuration differs from its resolved plan")
    seconds = plan["evaluation"]["seconds"] if args.seconds is None else args.seconds
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("--seconds must be finite and positive")
    seed = plan["evaluation"]["seed"] if args.seed is None else args.seed
    if type(seed) is not int or not 0 <= seed <= 2**32-1:
        raise ValueError("--seed must fit uint32")
    profile, strength = (("nominal", 0.) if args.domain_randomization == "off" or args.dr_strength == 0 else
        ("requested", args.dr_strength) if args.domain_randomization == "on" else task.domain_stage(plan, completed))
    if args.domain_randomization == "curriculum":
        strength *= args.dr_strength
    return dict(checkpoint=checkpoint, run=run, state=state, info=info, plan=plan,
        bank=bank, agent=agent, seconds=seconds, seed=seed, profile=profile, strength=strength)


def build_actor(observations, bundle, device):
    from rsl_rl.utils import resolve_class
    cls, settings = resolve_class(deepcopy(bundle["agent"]["actor"]))
    actor = cls(observations, bundle["agent"]["obs_groups"], "actor", 6, **settings).to(device).eval()
    actor.load_state_dict(bundle["state"]["actor_state_dict"], strict=True)
    return actor


def prepare(args, bundle, output):
    # Build the original mechanism, then declare a separate unscored playback
    # design; a one-robot preview has no training-population coverage obligation.
    cfg, settings, _ = configuration(output, bundle["plan"], profile=bundle["profile"], strength=bundle["strength"],
        evaluation=True, num_envs=max(80, args.num_envs), seed=bundle["seed"],
        device=args.device, performance_backend=args.performance_backend)
    design_file = output/"config/design.json"
    design = json.loads(design_file.read_text())
    design.update(run_scope="interactive_unscored", scope="prone_motion_checkpoint_preview")
    row = design["stages"]["speed"]["roles"].pop("train")
    design["stages"]["speed"]["roles"] = {"playback": row}
    cfg["target_num_envs"] = args.num_envs
    cfg["evaluation"]["seed"] = bundle["seed"]
    cfg["scene_groups"] = [dict(name="preview", fraction=1., terrain=["flat"])]
    row["contract"].update(target_num_envs=args.num_envs, scene_groups=deepcopy(cfg["scene_groups"]))
    if "evaluation" in row["contract"]:
        row["contract"]["evaluation"]["seed"] = bundle["seed"]
    if "evaluation.seed" in row["contract"]:
        row["contract"]["evaluation.seed"] = bundle["seed"]
    row["invocation"] = dict(num_envs=args.num_envs, seed=dict(rule="fixed", value=bundle["seed"]))
    write_json(design_file, design)
    cfg["design_preflight"]["sha256"] = digest(design_file)
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    control = load_chassis_control(ROOT/cfg["control_math_source"])
    proof = design_preflight(cfg, control, ROOT, stage="speed", role="playback", seed=bundle["seed"], num_envs=args.num_envs)
    write_json(output/"config/contract.json", cfg)
    contract_file = output/"task_contract.json"
    contract = json.loads(contract_file.read_text())
    contract.update(scope="unscored_playback", playback_num_envs=args.num_envs,
                    mechanism_config_sha256=digest(output/"config/contract.json"))
    write_json(contract_file, contract)
    write_json(output/"playback_preflight.json", proof)
    write_json(output/"playback_settings.json", settings)
    return cfg, settings, control


def case_for(args, env, plan):
    prone = args.scene == "prone"
    target = task.target(args.scene, plan["max_updates"], plan, final=True, height=args.height)
    if args.vx is not None:
        target[0] = args.vx
    if args.yaw_rate is not None:
        target[1] = args.yaw_rate
    return dict(role=args.scene, recovery=prone, birth_height=.173 if prone else args.birth_height,
        pose_index=env.prone_index if prone else env.height_pose[args.birth_height], pitch_deg=0.,
        target=target, command=[0., 0., .305 if prone else args.birth_height])


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        bundle = load_bundle(args)
        output = output_path(args, ROOT, training=False)
    except (ValueError, OSError, KeyError) as exc:
        cli.error(str(exc))
    if args.print_output:
        print(output)
        return
    output.mkdir(parents=True, exist_ok=False)
    cfg, settings, control = prepare(args, bundle, output)
    import torch
    from tensordict import TensorDict
    dummy = TensorDict(dict(policy=torch.zeros(args.num_envs, 35), critic=torch.zeros(args.num_envs, 83)), batch_size=[args.num_envs])
    actor = build_actor(dummy, bundle, "cpu")
    with torch.inference_mode():
        if actor(dummy).shape != (args.num_envs, 6):
            raise ValueError("Actor output ABI differs from six-axis prone-motion")
    receipt = dict(status="preflight_completed" if args.preflight_only else "starting", argv=sys.argv[1:],
        checkpoint=str(bundle["checkpoint"]), checkpoint_sha256=digest(bundle["checkpoint"]),
        completed_updates=bundle["state"]["iter"], source_revision=bundle["info"]["source_revision"],
        current_play_source="current checkout; this preview is not a replay of frozen training source",
        mechanism="checkpoint's original resolved plan", new_PPO=0, optimizer_loaded=False,
        hardware_motion=False, behavior_qualified=False, scene=args.scene, num_envs=args.num_envs,
        seed=bundle["seed"], device=args.device, headless=args.headless,
        profile=bundle["profile"], strength=bundle["strength"], seconds=bundle["seconds"], resets=0, ticks=0)
    write_json(output/"receipt.json", receipt)
    if args.preflight_only:
        print(f"Play preflight passed: update {receipt['completed_updates']}, {args.scene}, {args.num_envs} env(s). Zero PPO.", flush=True)
        return
    launcher = env = None
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=args.headless, device=args.device, enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
        manifest = json.loads((ROOT/cfg["asset_directory"]/"manifest.json").read_text())
        env = ProneMotionEnv(cfg, manifest, control, ROOT, stage_name="speed", num_envs=args.num_envs,
            device=args.device, seed=bundle["seed"], level=1., run_role="playback", bank=bundle["bank"], settings=settings)
        env.actor_updates = bundle["state"]["iter"]
        env.evaluation_cases = [case_for(args, env, bundle["plan"]) for _ in range(env.num_envs)]
        env.reset(torch.arange(env.num_envs, device=env.device))
        actor = actor.to(args.device)
        observation = env.get_observations()
        receipt.update(status="running", physics_dt=env.dt, policy_dt=env.policy_dt, selected_case=env.evaluation_cases[0])
        write_json(output/"receipt.json", receipt)
        horizon = None if args.loop and args.seconds is None else round(bundle["seconds"]/env.policy_dt)
        started = time.monotonic()
        with torch.inference_mode():
            while launcher.app.is_running() and not stopped:
                if horizon is not None and receipt["ticks"] >= horizon:
                    break
                action = actor(observation)
                if not torch.isfinite(action).all():
                    raise ValueError("Policy produced nonfinite actions")
                observation, _, done, _ = env.step(action)
                receipt["ticks"] += 1
                if not args.headless:
                    pose = env.robot.data.root_link_pose_w.torch[0, :3].cpu().tolist()
                    env.sim.set_camera_view((pose[0]+1.7, pose[1]-1.7, pose[2]+.8), pose)
                    env.sim.render()
                if bool(env.finished.all()):
                    if not args.loop:
                        receipt["ended_on_native_done"] = True
                        break
                    receipt["resets"] += 1
                    env.reset(torch.arange(env.num_envs, device=env.device))
                    observation = env.get_observations()
                if args.real_time and not args.headless:
                    time.sleep(max(0., started+receipt["ticks"]*env.policy_dt-time.monotonic()))
        receipt.update(status="stopped" if stopped else "completed_preview_unqualified", simulated_seconds=receipt["ticks"]*env.policy_dt,
            finished_lanes=env.finished.cpu().tolist())
    except BaseException:
        receipt.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        write_json(output/"receipt.json", receipt)
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()

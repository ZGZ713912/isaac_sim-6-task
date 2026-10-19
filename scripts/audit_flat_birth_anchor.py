#!/usr/bin/env python3
"""Read the first stationary anchor and its real reward in a frozen train scene.

No optimizer is constructed. This audit waits for a named native queue to finish
before creating its scene, and never controls or signals that queue.
"""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.flat_base_continuation import read_plan, write_bundle, digest, encode

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--after-queue", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
plan_file = "contracts/flat_base_finetune_v7.json"
plan = read_plan(ROOT, plan_file)
paths = [Path(__file__), ROOT / plan_file, ROOT / "scripts/flat_rmcs_command_reference.py",
    ROOT / "src/wheeled_algo/command_residual.py", ROOT / "src/wheeled_algo/guarded_feedback_residual.py",
    ROOT / "src/wheeled_algo/onnx_actor_import.py", *(ROOT / "src/wheeled_tasks/chassis").glob("*.py")]
hashes = {str(p.relative_to(ROOT)):digest(p) for p in paths}
report = dict(schema="flat-training-birth-anchor-native-audit-v1", pid=os.getpid(),
    status="waiting_existing_native_queue", new_optimizer_updates=0,
    checkpoint_sha256=digest(args.checkpoint), plan_sha256=digest(ROOT / plan_file),
    source_sha256=hashes, hardware_motion_started=False,
    scope="First two seconds of the exact mixed training scene; diagnostic only, not acceptance",
    started_at=datetime.now(timezone.utc).isoformat())


def save():
    tmp = args.output / "audit.tmp"
    tmp.write_bytes(encode(report)); tmp.replace(args.output / "audit.json")


launcher = env = None
try:
    save()
    deadline = time.monotonic() + 3600
    while True:
        queue = json.loads(args.after_queue.read_text())
        if queue["status"] == "complete":
            break
        if queue["status"] in ("failed", "regression_stopped"):
            raise ValueError("The preceding native queue did not complete")
        if time.monotonic() > deadline:
            raise TimeoutError("The preceding native queue did not finish within one hour")
        time.sleep(5)
    for path, expected in hashes.items():
        if digest(ROOT / path) != expected:
            raise ValueError("Frozen audit source changed: " + path)
    report["preceding_queue_sha256"] = digest(args.after_queue)
    config, seed = write_bundle(ROOT, args.output / "config", 512, plan_file=plan_file)
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    _, manifest = preflight(args.output / "config/contract.json")
    control = load_chassis_control(ROOT / config["control_math_source"])
    report["static_preflight"] = design_preflight(config, control, ROOT, stage="speed", role="train", num_envs=512, seed=seed)
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(dict(headless=True, device="cuda:0", enable_cameras=False,
        kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
    import numpy as np
    import torch
    from wheeled_tasks.chassis.env import ChassisEnv
    from wheeled_tasks.chassis.flat_command_reference import install_training_reference
    from wheeled_algo.command_residual import mean_from_checkpoint
    torch.set_num_threads(4); torch.manual_seed(seed)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state["infos"]["plan_sha256"] != report["plan_sha256"]:
        raise ValueError("Audit checkpoint belongs to a different training plan")
    mean = mean_from_checkpoint(state, ROOT / plan["actor_bundle"] / "source.onnx").to("cuda:0").eval()
    env = ChassisEnv(config, manifest, control, ROOT, stage_name="speed", num_envs=512,
        device="cuda:0", seed=seed, level=1., run_role="train")
    report["training_reference"] = install_training_reference(env, plan["training_command_reference"])
    env.reset(torch.arange(512, device=env.device))
    observations = env.get_observations()
    selected = env.stationary_group.nonzero(as_tuple=False).flatten()
    initial_xy = (env.robot.data.root_link_pose_w.torch[:, :2] - env.origins[:, :2]).clone()
    alive = torch.ones(512, dtype=torch.bool, device=env.device)
    first_tick = torch.full((512,), -1, dtype=torch.long, device=env.device)
    acquired_xy = torch.zeros(512, 2, device=env.device)
    traces = {k:[] for k in ("active", "episode_ticks", "position_xy", "anchor_xy", "anchor_valid", "anchor_reward_density", "commands", "height", "velocity")}
    report.update(status="running", startup=env.startup_report, stationary_lanes=len(selected))
    save()
    with torch.inference_mode():
        for _ in range(100):
            actions = mean(observations["policy"])
            observations, _, done, extras = env.step(actions)
            data = extras["diagnostics"]
            valid = env.stationary_anchor.valid.clone()
            entering = alive & ~done & valid & (first_tick < 0)
            first_tick[entering] = data["episode_ticks"][entering]
            acquired_xy[entering] = env.stationary_anchor.position[entering] - initial_xy[entering]
            values = dict(active=alive, episode_ticks=data["episode_ticks"], position_xy=data["position"][:, :2]-initial_xy,
                anchor_xy=env.stationary_anchor.position-initial_xy, anchor_valid=valid,
                anchor_reward_density=extras["native_reward_components"]["stationary_anchor"],
                commands=data["commands"], height=data["height"], velocity=data["velocity"])
            for key,value in values.items():
                traces[key].append(value[selected].cpu().numpy().copy())
            alive &= ~done.bool()
    arrays={key:np.asarray(value) for key,value in traces.items()}
    names=np.asarray([env.scene_groups[int(i)] for i in selected.cpu()])
    np.savez_compressed(args.output / "traces.npz", **arrays, groups=names)
    ticks, delta = first_tick[selected].cpu().numpy(), acquired_xy[selected].cpu().numpy()
    cases={}
    for name in dict.fromkeys(names):
        mask=names==name; captured=ticks[mask]>=0; distance=np.linalg.norm(arrays["position_xy"][:,mask],axis=-1)
        costs=arrays["anchor_reward_density"][:,mask]*arrays["active"][:,mask]*env.policy_dt
        cases[str(name)]=dict(lanes=int(mask.sum()), captured=int(captured.sum()),
            first_anchor_tick_mean=float(ticks[mask][captured].mean()) if captured.any() else None,
            displacement_before_first_anchor_mean_m=float(np.linalg.norm(delta[mask][captured],axis=-1).mean()) if captured.any() else None,
            displacement_before_first_anchor_max_m=float(np.linalg.norm(delta[mask][captured],axis=-1).max()) if captured.any() else None,
            max_birth_displacement_m=float(np.where(arrays["active"][:,mask],distance,0.).max()),
            stationary_anchor_payment_mean_per_lane=float(costs.sum(0).mean()))
    report.update(status="audited", cases=cases, traces_sha256=digest(args.output / "traces.npz"),
        conclusion_scope="First-anchor timing and actual penalty payment only; no claim of a unique cause")
except Exception:
    report.update(status="failed", error=traceback.format_exc());traceback.print_exc()
finally:
    report["finished_at"]=datetime.now(timezone.utc).isoformat();save()
    if env is not None:
        env.close()
    if launcher is not None:
        launcher.app.close()
raise SystemExit(0 if report["status"]=="audited" else 1)

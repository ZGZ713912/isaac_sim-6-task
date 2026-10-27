#!/usr/bin/env python3
"""Cold-review a frozen prone actor without constructing PPO or an optimizer."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=261101)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    info, run = state["infos"], checkpoint.parent
    if (info.get("schema") != "scratch-standing-checkpoint-v1"
            or not info.get("continuation_forbidden")
            or info.get("initialization", {}).get("checkpoint_loaded") is not False
            or "standard_prone" not in info.get("settings", {})):
        raise ValueError("Need an authenticated from-zero standard-prone checkpoint")
    runtime = output / "frozen_runtime"
    for name, expected in info["source_hashes"].items():
        source = run / "source_snapshot" / name
        if sha(source) != expected:
            raise ValueError("Frozen source mismatch: " + name)
        destination = runtime / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    geometry = info["settings"]["standard_prone"]["encoder_geometry"]
    table = ROOT / geometry["table"]
    if sha(table) != geometry["table_sha256"]:
        raise ValueError("Frozen encoder geometry differs")
    destination = runtime / geometry["table"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(table.read_bytes())
    sys.path[:0] = [str(runtime / "src"), str(runtime / "scripts")]
    spec = importlib.util.spec_from_file_location("frozen_prone_training", runtime / "scripts/train_scratch_standing.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    cfg = json.loads((run / "config/contract.json").read_text())
    agent = json.loads((run / "agent_config.json").read_text())
    if (sha(run / "config/contract.json") != info["mechanism_config_sha256"]
            or sha(run / "agent_config.json") != info["agent_config_sha256"]
            or sha(run / "pose_bank.json") != info["bank_sha256"]):
        raise ValueError("Frozen mechanism, actor or reset contract differs")
    original = Path(cfg["design_preflight"]["file"])
    if sha(original) != cfg["design_preflight"]["sha256"]:
        raise ValueError("Frozen design differs")
    design = json.loads(original.read_text())
    row = design["stages"]["speed"]["roles"]["train"]
    old_seed = row["runtime"]["seed"]
    row["runtime"]["seed"] = args.seed
    row["invocation"] = dict(num_envs=cfg["target_num_envs"], seed=dict(rule="fixed", value=args.seed))
    design["scope"] = "readonly_standard_prone_checkpoint_review"
    design["review_basis"] = dict(checkpoint_sha256=sha(checkpoint), seed_override=[old_seed, args.seed])
    write(output / "config/design.json", design)
    cfg["design_preflight"] = dict(file=str(output / "config/design.json"),
        sha256=sha(output / "config/design.json"), stage="speed")
    write(output / "config/contract.json", cfg)
    progress = dict(schema="readonly-standard-prone-review-v1", status="starting",
        observed_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        source_commit=info["committed_source"]["commit"], completed_updates=info["completed_updates"],
        new_PPO=0, optimizer_constructed=False, seed=args.seed,
        frozen_source_count=len(info["source_hashes"]), geometry_sha256=sha(table),
        evaluator_sha256=sha(__file__), hardware_ready=False)
    write(output / "receipt.json", progress)
    launcher = env = writer = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        from rsl_rl.utils import resolve_class
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.chassis.standard_prone_env import StandardProneEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.physical_standing_reward import install
        from evaluate_standard_prone import evaluate
        manifest = json.loads((ROOT / cfg["asset_directory"] / "manifest.json").read_text())
        env = StandardProneEnv(cfg, manifest, load_chassis_control(ROOT / cfg["control_math_source"]), ROOT,
            stage_name="speed", num_envs=cfg["target_num_envs"], device=args.device, seed=args.seed,
            level=1., run_role="train", bank=run / "pose_bank.json", settings=deepcopy(info["settings"]))
        install(env, speed_weight=0., contact_weight=0., torque_weight=0.)
        actor_class, actor_cfg = resolve_class(agent["actor"])
        actor_file = Path(inspect.getsourcefile(actor_class)).resolve()
        if not actor_file.is_relative_to(runtime):
            raise ValueError("Actor implementation must come from frozen runtime")
        actor = actor_class(env.get_observations(), agent["obs_groups"], "actor", 6, **actor_cfg).to(args.device)
        actor.load_state_dict(state["actor_state_dict"], strict=True)
        actor.eval()
        writer = SummaryWriter(str(output))
        progress["status"] = "evaluating"
        write(output / "receipt.json", progress)
        progress["evaluation"] = evaluate(env, actor, output / "evaluation", writer, info["completed_updates"])
        progress["status"] = "completed_unqualified"
    except BaseException:
        progress.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        write(output / "receipt.json", progress)
        if writer is not None:
            writer.flush()
            writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()

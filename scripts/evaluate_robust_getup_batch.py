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

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=211131)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=["parent", "robust", "legs_only", "undamped"],
                        choices=["parent", "robust", "legs_only", "undamped"])
    args = parser.parse_args()
    seeds = args.seeds or [args.seed]
    args.seed = seeds[0]
    if not 1 <= args.replicas <= 4 or len(set(args.variants)) != len(args.variants):
        parser.error("Use one to four replicas and unique variants")
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("robust_training", ROOT / "scripts/train_robust_getup.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    per_variant = 35 * args.replicas
    count = per_variant * len(args.variants) * len(seeds)
    if len(seeds) > 2 or len(set(seeds)) != len(seeds) or count > 1024:
        parser.error("Use at most two distinct seeds and 1024 total native environments")
    checkpoint_sha = training.digest(args.checkpoint)
    config_args = argparse.Namespace(output=args.output, bank=ROOT / "reports/unified_robust_20261021/pose_bank002.json",
        no_domains=True, curriculum_updates=128, quiet_leg_alpha=.25, quiet_wheel_alpha=.65,
        num_envs=count, rollout_steps=128, mode="evaluate", friction="high", seed=args.seed)
    cfg, task = training.prepare(config_args)
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
        state = {"infos": verified_info}
        models = []
        for name in args.variants:
            mean = deepcopy(verified_mean).cuda()
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
        groups = [(seed, name) for seed in seeds for name in args.variants]
        cases = [case for seed, name in groups for case in training.fixed_cases(env.bank, per_variant, seed)]
        report = training.evaluate(env, BatchMean().cuda(), args.output / "all", torch,
                                   seed=args.seed, seconds=12., cases=cases)
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
            split["seed"] = seed
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
    finally:
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()

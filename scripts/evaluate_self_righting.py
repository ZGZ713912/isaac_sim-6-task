#!/usr/bin/env python3
"""Independent fallen-pose screening of a fresh recovery actor; zero PPO."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def cases_for(bank, standard=False):
    from wheeled_tasks.chassis.multi_pose_recovery import FAMILIES, LEG_CLASSES
    cases, missing = [], []
    groups = [("belly_down", "rear_rear")] if standard else [(f, c) for f in FAMILIES for c in LEG_CLASSES]
    for family, leg_class in groups:
        matches = [(i, p) for i, p in enumerate(bank["poses"]) if
            p.get("recovery", {}).get("family") == family and
            p["recovery"]["leg_class"] == leg_class and
            p["recovery"]["split"] == ("standard" if standard else "holdout") and
            not p["recovery"].get("dynamic")]
        if not matches:
            missing.extend(dict(family=family, leg_class=leg_class, heading_rad=h,
                                passed=False, reason="no_admitted_independent_static_pose") for h in (0., math.pi / 2))
            continue
        i, p = matches[0]
        for heading in (0., math.pi / 2):
            cases.append(dict(role="prone", recovery=True, pose_index=i, family=family,
                leg_class=leg_class, split=p["recovery"]["split"], heading_rad=heading,
                birth_height=p["root_position_m"][2], pitch_deg=0., target=[0., 0., .305], command=[0., 0., .305]))
    return cases, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=24.)
    args = parser.parse_args()
    import play_prone_motion as play
    from train_scratch_standing import digest, write_json
    import torch
    import numpy as np
    from tensordict import TensorDict
    from torch.utils.tensorboard import SummaryWriter
    cli = play.parser().parse_args(["--checkpoint", str(args.checkpoint), "--run", str(args.run),
        "--num-envs", str(args.num_envs), "--seconds", str(args.seconds), "--domain-randomization", "off",
        "--headless", "--no-real-time"])
    bundle = play.load_bundle(cli)
    # Evaluation may add scripts, but may not silently replace the training plant.
    training_hashes = json.loads((args.run / "source_hashes.json").read_text())
    for name, value in training_hashes.items():
        if name.startswith("src/") and digest(ROOT / name) != value:
            raise ValueError("Evaluation plant changed since training: " + name)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, settings, control = play.prepare(cli, bundle, args.output / "configuration")
    report = dict(schema="self-righting-independent-screen-v1", status="starting", new_PPO=0,
        checkpoint=str(bundle["checkpoint"]), checkpoint_sha256=digest(bundle["checkpoint"]),
        completed_updates=bundle["state"]["iter"], source_revision=bundle["info"]["source_revision"],
        evaluation_script_sha256=digest(Path(__file__)), hardware_io=False, full_capability_qualified=False,
        wheelbipe16001_handoff_tested=False, followup_controller="same recovery actor",
        impulse_scope="per-body positive world-Z contact integral before part-group reduction",
        pose_scope="first admitted independent static pose per family/leg class, headings0 and90deg",
        criteria=dict(active_recovery_max_s=20., post_completion_stable_hold_s=3.), cases=[])
    write_json(args.output / "report.json", report)
    env = launcher = writer = None
    started = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device="cuda:0", enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
        env = ProneMotionEnv(cfg, json.loads((ROOT / cfg["asset_directory"] / "manifest.json").read_text()),
            control, ROOT, stage_name="speed", num_envs=args.num_envs, device="cuda:0", seed=261105,
            level=1., run_role="playback", bank=bundle["bank"], settings=settings)
        env.actor_updates = bundle["state"]["iter"]
        dummy = TensorDict(dict(policy=torch.zeros(args.num_envs, 35), critic=torch.zeros(args.num_envs, 87)),
                           batch_size=[args.num_envs])
        actor = play.build_actor(dummy, bundle, "cuda:0")
        cases, missing = cases_for(env.bank, bundle["plan"]["multi_pose_recovery"].get("reset_scope") == "standard_prone_only")
        report["cases"].extend(missing)
        writer = SummaryWriter(str(args.output / "events"))
        report.update(physics_dt_s=env.dt, policy_dt_s=env.policy_dt,
                      impulse_window_s=env.recovery_impact.config["window_seconds"], group_order=["chassis", "gimbal", "legs", "wheels"])
        with torch.inference_mode():
            for offset in range(0, len(cases), env.num_envs):
                batch = cases[offset:offset + env.num_envs]; n = len(batch)
                env.evaluation_cases = [deepcopy(batch[i % n]) for i in range(env.num_envs)]
                env.reset(torch.arange(env.num_envs, device=env.device))
                obs = env.get_observations()
                active = torch.ones(n, dtype=torch.bool, device=env.device)
                hold = torch.zeros(n, device=env.device)
                rows = [dict(c, passed=False, completed=False, active_time_s=None, terminated=False,
                    stable_hold_s=0., force_peak_n=[0.] * 4, impulse_peak_ns=[0.] * 4,
                    closure_peak_m=0., policy_samples=0) for c in batch]
                trace = []
                for _ in range(round(args.seconds / env.policy_dt)):
                    obs, _, done, extra = env.step(actor(obs))
                    d = extra["diagnostics"]
                    stable = d["stable"][:n] & d["recovered"][:n] & ~d["terminated"][:n]
                    hold = torch.where(stable & active, hold + env.policy_dt, 0.)
                    trace.append(torch.stack((d["time"][:n], d["height"][:n], d["gravity"][:n, 2],
                        d["gap"][:n], d["recovered"][:n].float(), d["terminated"][:n].float(), active.float()), -1).cpu().numpy())
                    for lane, row in enumerate(rows):
                        if not bool(active[lane]): continue
                        row["policy_samples"] += 1
                        row["closure_peak_m"] = max(row["closure_peak_m"], float(d["gap"][lane]))
                        row["force_peak_n"] = np.maximum(row["force_peak_n"], d["recovery_episode_force_peak_n"][lane].cpu().numpy()).tolist()
                        row["impulse_peak_ns"] = np.maximum(row["impulse_peak_ns"], d["recovery_episode_impulse_peak_ns"][lane].cpu().numpy()).tolist()
                        row["terminated"] |= bool(d["terminated"][lane])
                        if bool(d["recovery_completed_event"][lane]):
                            row.update(completed=True, active_time_s=float(d["recovery_active_seconds"][lane]))
                        row["stable_hold_s"] = max(row["stable_hold_s"], float(hold[lane]))
                        row["passed"] = bool(row["completed"] and not row["terminated"] and
                            row["active_time_s"] <= 20. and row["stable_hold_s"] >= 3. - 1e-5)
                        if row["passed"] or bool(done[lane]): active[lane] = False
                    if not bool(active.any()): break
                np.savez_compressed(args.output / f"batch_{offset:03d}.npz", trajectory=np.stack(trace))
                report["cases"].extend(rows)
                write_json(args.output / "report.json", report)
        passed = sum(r["passed"] for r in report["cases"])
        report.update(status="completed_unqualified", pass_count=passed, case_count=len(report["cases"]),
            success_rate=passed / len(report["cases"]), elapsed_s=time.monotonic() - started)
        writer.add_scalar("ColdRecovery/success_rate", report["success_rate"], report["completed_updates"])
        for family in sorted({r["family"] for r in report["cases"]}):
            group = [r for r in report["cases"] if r["family"] == family]
            writer.add_scalar(f"ColdRecovery/{family}/success_rate", sum(r["passed"] for r in group) / len(group), report["completed_updates"])
        write_json(args.output / "report.json", report)
        print(json.dumps({k:report[k] for k in ("status", "pass_count", "case_count", "success_rate", "elapsed_s")}))
    except BaseException:
        report.update(status="failed", error=traceback.format_exc(), elapsed_s=time.monotonic() - started)
        write_json(args.output / "report.json", report)
        raise
    finally:
        if writer is not None: writer.close()
        if env is not None: env.close()
        if launcher is not None: launcher.app.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent normally-up prone holdouts to default standing; no donor policy or PPO."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from types import SimpleNamespace

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]


def main():
    from train_scratch_standing import digest, write_json
    from train_prone_motion import source_hashes
    from play_prone_motion import load_bundle, prepare, build_actor
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--device", default="cuda:0")
    args = cli.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    load_args = SimpleNamespace(checkpoint=args.checkpoint, run=None, device=args.device, plan=None,
        task=None, num_envs=16, height=.305, vx=None, yaw_rate=None, scene="prone", seconds=30.2,
        seed=None, domain_randomization="off", dr_strength=0., performance_backend="eager", sampling_backend="packed")
    bundle = load_bundle(load_args)
    plan = bundle["plan"]
    if "prone_support" not in plan:
        raise ValueError("Need a checkpoint from the normally-up prone support task")
    expected = bundle["info"]["source_hashes"]
    mismatches = [p for p,h in expected.items() if digest(ROOT/p) != h]
    if mismatches:
        raise ValueError("Evaluation must use the original frozen sources: "+str(mismatches))
    report = dict(schema="wheel-leg-prone-support-cold-evaluation-v1", status="preparing", new_PPO=0,
        checkpoint=str(bundle["checkpoint"]), checkpoint_sha256=digest(bundle["checkpoint"]),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_hashes=expected, zero_prefix=True, behavior_scope="normally_up_belly_grounded_to_default0.305m_standing",
        full_recovery56_tested=False, standing_tested=False, actual_16001_handoff=False,
        actual_handoff_time_s=None, cases=[])
    write_json(output/"report.json", report)
    env = launcher = observer = writer = None
    stopped = False
    def stop(*unused):
        nonlocal stopped
        stopped = True
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import torch
        import numpy as np
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.chassis.prone_support_env import ProneSupportEnv
        from wheeled_tasks.chassis.evaluation_trace import RecoveryTraceObserver, contact_integrals, field_values
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        cfg, settings, control = prepare(load_args, bundle, output)
        manifest = json.loads((ROOT/cfg["asset_directory"]/"manifest.json").read_text())
        env = ProneSupportEnv(cfg, manifest, control, ROOT, stage_name="speed", num_envs=16,
            device=args.device, seed=bundle["seed"], level=1., run_role="playback",
            bank=bundle["bank"], settings=settings)
        cases = []
        for index in env.support_holdout:
            for heading in (0., np.pi/2):
                pose = env.poses[index]
                cases.append(dict(role="prone", recovery=True, pose_index=index,
                    birth_height=pose["root_position_m"][2], target=[0.,0.,.305], command=[0.,0.,.305],
                    pitch_deg=0., heading_rad=float(heading), evaluation_seconds=30.2,
                    name=pose["name"]+f"_heading{round(heading*180/np.pi)}",
                    leg_class=pose["recovery"]["leg_class"], support_birth_phase=2))
        if len(cases) != 16:
            raise ValueError("This finite evaluation requires the eight held-out normally-up leg configurations in two headings")
        env.evaluation_cases = cases
        env.reset(torch.arange(16, device=args.device))
        observation = env.get_observations()
        actor = build_actor(observation, bundle, args.device)
        observer = RecoveryTraceObserver(env, output/"raw_physics", seconds=30.2)
        observer.begin_batch(0, cases, cases)
        writer = SummaryWriter(str(output/"events"))
        first = [None]*16
        first_streak = [None]*16
        lost = [False]*16
        post = [False]*16
        hard = [False]*16
        scored = [True]*16
        force = np.zeros((16,4))
        impulse = force.copy()
        steps = round(30.2/env.policy_dt)
        for step in range(steps):
            if stopped:
                raise InterruptedError("Cold prone support evaluation stopped")
            with torch.inference_mode():
                action = actor(observation)
                observer.begin_policy_step(observation, action, scored, step)
                observation, reward, done, info = env.step(action)
            d = info["diagnostics"]
            observer.end_policy_step(d, done, info["time_outs"])
            active_impulse = d["recovery_episode_impulse_peak_ns"].detach().cpu().numpy()
            active_force = d["recovery_episode_force_peak_n"].detach().cpu().numpy()
            force, impulse = np.maximum(force, active_force), np.maximum(impulse, active_impulse)
            for lane in range(16):
                if not scored[lane]:
                    continue
                hard[lane] |= bool(d["terminated"][lane])
                age = float(d["support_active_seconds"][lane])
                hold = float(d["support_native_hold_s"][lane])
                if bool(d["standing_completed_event"][lane]) and first[lane] is None:
                    first[lane], first_streak[lane] = age, hold
                if first[lane] is not None:
                    lost[lane] |= hold+1e-5 < first_streak[lane]
                    post[lane] |= (age-first[lane] >= 10.-1e-5 and
                                   hold >= first_streak[lane]+10.-1e-5 and not lost[lane])
                scored[lane] &= not bool(done[lane])
            writer.add_scalar("ProneSupportCold/completed_events", int(d["standing_completed_event"].sum()), step)
            writer.add_scalar("ProneSupportCold/native_hold_peak_s", float(d["support_native_hold_s"].max()), step)
        rows = []
        for lane, case in enumerate(cases):
            passed = first[lane] is not None and first[lane] <= 20. and post[lane] and not hard[lane] and not lost[lane]
            rows.append(dict(name=case["name"], pose_index=case["pose_index"], leg_class=case["leg_class"],
                heading_rad=case["heading_rad"], passed=passed, actual_standing_completed=first[lane] is not None,
                enable_to_default_standing_s=first[lane], post_standing_10s=post[lane],
                post_standing_lost=lost[lane], hard_fault=hard[lane], prefix_s=0.,
                force_peak_n=force[lane].tolist(), impulse_peak_ns=impulse[lane].tolist(),
                actual_16001_handoff_time_s=None))
        audit = observer.end_batch(rows)
        observer.close("completed")
        observer = None
        # Separately integrate the entire scored physical trajectory, including
        # disabled placement; delaying enable cannot hide its contact loads.
        raw = output/"raw_physics/batch_000/physical"
        manifest_raw = json.loads((raw/"manifest.json").read_text())
        values = np.load(raw/"values.npy", mmap_mode="r")
        clock = np.load(raw/"clock.npy", mmap_mode="r")
        def frames():
            for frame in range(manifest_raw["valid_frames"]):
                yield (clock[frame,0], clock[frame,1],
                    field_values(values[frame], manifest_raw["fields"], "contact/force_world_n"),
                    field_values(values[frame], manifest_raw["fields"], "mask/scored") > 0)
        whole = contact_integrals(frames(), list(env.robot.body_names))
        for lane, row in enumerate(rows):
            row["whole_episode_impulse_peak_ns"] = whole["group_impulse_peak_ns"][lane]
            writer.add_scalar("ProneSupportCold/pass", float(row["passed"]), lane)
            writer.add_scalar("ProneSupportCold/completed", float(row["actual_standing_completed"]), lane)
            for group, value in zip(whole["group_order"], row["whole_episode_impulse_peak_ns"]):
                writer.add_scalar("ProneSupportCold/whole_impulse/"+group+"_Ns", value, lane)
        write_json(output/"whole_episode_contact_integrals.json", whole)
        report.update(status="completed_unqualified", standing_tested=True, cases=rows, denominator=len(rows),
            passed=sum(r["passed"] for r in rows), actual_completions=sum(r["actual_standing_completed"] for r in rows),
            raw_audit_same_objects=audit["same_controller_objects"],
            physical_frames=audit["physical_frames"], policy_frames=audit["policy_frames"])
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if observer is not None:
            observer.close("failed")
        if writer is not None:
            writer.close()
        report["source_mismatches_at_exit"] = [p for p,h in expected.items() if digest(ROOT/p) != h]
        write_json(output/"report.json", report)
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()

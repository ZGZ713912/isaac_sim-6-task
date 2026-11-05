#!/usr/bin/env python3
"""One native scripted inverted->belly screen; no learned actor or standing handoff."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

from train_scratch_standing import digest, write_json
from wheeled_tasks.chassis.synchronized_rollover import Sweep


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--direction", type=int, choices=(-1, 1), required=True)
    cli.add_argument("--profile", choices=("direct", "braked"), required=True)
    cli.add_argument("--speed", type=float, default=4.)
    cli.add_argument("--seconds", type=float, default=8.)
    cli.add_argument("--device", default="cuda:0")
    cli.add_argument("--preflight-only", action="store_true")
    args = cli.parse_args()
    if not math.isfinite(args.seconds) or not 1. <= args.seconds <= 20.:
        cli.error("Active screen must last 1..20s unless an original fault terminates it")
    import numpy as np
    import torch
    from build_full_vehicle_training import ResetGeometry, geometry
    from play_prone_motion import prepare
    from wheeled_tasks.chassis import prone_motion as task
    from wheeled_tasks.chassis.prone_motion_cli import validate_device
    validate_device(args.device)
    sweep = Sweep(args.direction, args.speed, args.profile)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT/"contracts/wheel_leg_self_righting_passive_window_v1.json"
    plan = task.validate(json.loads(plan_path.read_text()))
    bundle = dict(plan=plan, profile="nominal", strength=0., seed=261105)
    args.num_envs, args.performance_backend, args.sampling_backend = 1, "eager", "packed"
    cfg, settings, control = prepare(args, bundle, output)
    spec_path = ROOT/cfg["asset_directory"]/"model_spec.json"
    spec = json.loads(spec_path.read_text())
    builder = ResetGeometry(spec, spec_path.parent)
    q110 = builder.knee(110.)
    rotation = np.diag([1., -1., -1.])
    minimum = min(v[2] for _, v in builder.bounds(q110, rotation))
    root = [0., 0., -minimum+.002, 1., 0., 0., 0.]
    gap = float(np.linalg.norm(geometry.closure_error(spec, q110).reshape(-1, 3), axis=1).max())
    if gap > 1e-7:
        raise ValueError("Initial linkage does not close")
    four = [q110[n] for n in ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")]
    names = subprocess.check_output(["git", "ls-files", "src", "scripts", "contracts"], cwd=ROOT, text=True).splitlines()
    hashes = {n: digest(ROOT/n) for n in names if (ROOT/n).is_file()}
    report = dict(schema="wheel-leg-synchronized-rollover-screen-v1", status="preflight_completed",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=hashes, contract_sha256=digest(ROOT/"contracts/wheel_leg_synchronized_rollover_screen_v1.json"),
        direction=args.direction, profile=args.profile, speed_rad_s=args.speed,
        initial_root_pose=root, initial_full_joint_positions=q110, initial_gap_m=gap,
        pose_scope="synthetic exact Rx180 and mechanical110; geometry checked; not an admitted holdout",
        actor_loaded=False, checkpoint_loaded=False, new_PPO=0, hardware_io=False,
        original_protections_changed=False, final_standing_pass=False, actual_16001_handoff=False,
        enable_to_handoff_s=None, failure_denominator=1)
    write_json(output/"report.json", report)
    if args.preflight_only:
        return
    launcher = env = runtime = writer = None
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
        from wheeled_tasks.chassis.handoff_evaluation import (
            HandoffEvaluation, construct_evaluation_env, contact_summary, history_snapshot)
        from torch.utils.tensorboard import SummaryWriter
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

        class CaseEnv(ProneMotionEnv):
            armed = False
            suppressed_resets = 0
            def reset(self, ids, **kwargs):
                if self.armed:
                    self.suppressed_resets += 1
                    return
                if getattr(self, "course_ready", False) and self.evaluation_cases is None:
                    # Append an explicitly synthetic screen birth after authenticating
                    # the original bank. Never mutate its certified rows or evidence.
                    template = next(i for i, p in enumerate(self.poses)
                        if p.get("recovery", {}).get("family") == "roof_down")
                    pose = deepcopy(self.poses[template])
                    pose.update(name="synthetic_roof110_sweep", full_joint_positions=q110,
                        root_position_m=root[:3], root_quaternion_xyzw=root[3:],
                        root_linear_velocity_body_m_s=[0., 0., 0.],
                        root_angular_velocity_body_rad_s=[0., 0., 0.], roll_deg=180., pitch_deg=0.)
                    index = len(self.poses)
                    self.poses.append(pose)
                    self.bank_joints = torch.cat((self.bank_joints,
                        self.bank_joints.new_tensor([[q110[n] for n in self.robot.joint_names]])))
                    self.bank_roots = torch.cat((self.bank_roots, self.bank_roots.new_tensor([root])))
                    self.bank_velocities = torch.cat((self.bank_velocities, torch.zeros_like(self.bank_velocities[:1])))
                    self.bank_family = torch.cat((self.bank_family, self.bank_family[template:template+1]))
                    self.bank_tilt = torch.cat((self.bank_tilt, self.bank_tilt.new_tensor([180.])))
                    self.evaluation_cases = [dict(role="prone", recovery=True, pose_index=index,
                        birth_height=root[2], pitch_deg=0., heading_rad=0.,
                        target=[0., 0., .305], command=[0., 0., .305], evaluation_seconds=args.seconds+2.)]
                return super().reset(ids, **kwargs)

        manifest = json.loads((ROOT/cfg["asset_directory"]/"manifest.json").read_text())
        env = construct_evaluation_env(CaseEnv, cfg, manifest, control, ROOT, stage_name="speed", num_envs=1,
            device=args.device, seed=261105, level=1., run_role="playback",
            bank=ROOT/"model/recovery_multi_v202_fine_pd200_20261105/pose_bank.json", settings=settings)
        env.armed = True
        runtime = HandoffEvaluation(env)
        # The recorder owns a separate contact buffer; do not replace task contact data.
        def read_contacts(dt):
            import warp as wp
            result = torch.empty_like(env.contact_force)
            for ids, view in env.contact_views:
                matrix = wp.to_torch(view.get_contact_force_matrix(dt=dt))
                result[ids] = matrix.reshape(len(ids), env.body_count, -1, 3).sum(2)
            runtime.raw_contacts = result
        original_observe = runtime.observe_physics
        def observe(dt, effort, physical, active):
            original_observe(dt, effort, physical, active)
            runtime.records[-1]["contact_force"] = runtime.raw_contacts[0].cpu().tolist()
        runtime.read_contacts, runtime.observe_physics = read_contacts, observe
        initial_history = history_snapshot(env)
        wheel_object, motor_object = env.identified_wheels, env.actuator_response
        original_decode = env.v5.decode
        last_targets = four.copy()
        def decode(actions, current):
            _, _, raw = original_decode(actions, current)
            legs = current.new_tensor([last_targets])
            env.encoder_increment_actions.last_targets = legs
            return legs, current.new_zeros((1, 2)), raw
        env.v5.decode = decode
        writer = SummaryWriter(str(output/"events"))
        events = []
        quiet = 0.
        first_upright = None
        stable_belly = None
        report.update(status="running", native_admission=runtime.admission, selected_case=env.evaluation_cases[0])
        write_json(output/"report.json", report)
        with torch.inference_mode():
            while not stopped:
                gravity = env.robot.data.projected_gravity_b.torch[0]
                tilt = math.degrees(math.acos(max(-1., min(1., -float(gravity[2])))))
                if runtime.enabled_at is not None:
                    sweep.step(env.policy_dt, tilt)
                    last_targets[:] = sweep.targets(four)
                _, _, done, extra = env.step(torch.zeros((1, 6), device=env.device))
                d = extra["diagnostics"]
                if bool(d["enable_event"][0]) and runtime.enabled_at is None:
                    runtime.enabled_at = runtime.time
                    report["enable_history"] = history_snapshot(env)
                height = float(d["height"][0])
                tilt = math.degrees(math.acos(max(-1., min(1., -float(d["gravity"][0, 2])))))
                omega = float(d["omega"][0].norm())
                active = runtime.enabled_at is not None and runtime.time > runtime.enabled_at+1e-9
                if active and tilt < 20. and first_upright is None:
                    first_upright = runtime.time
                belly = active and tilt < 20. and .1 < height < .24 and omega < 1.
                quiet = quiet+env.policy_dt if belly else 0.
                if quiet >= .2-1e-10 and stable_belly is None:
                    stable_belly = runtime.time
                faults = {k: bool(d[k][0]) for k in
                          ("mechanical", "impact_bad", "boundary_bad", "terminated", "done")}
                faults.update({k: bool(v[0]) for k, v in d.get("upright_failures", {}).items()})
                row = dict(time=runtime.time, active=active, tilt_deg=tilt, height_m=height,
                    omega_rad_s=omega, sweep_rad=sweep.angle, target_velocity_rad_s=sweep.velocity,
                    targets=last_targets.copy(), braking=sweep.braking, captured=sweep.captured,
                    recovered=bool(d["recovered"][0]), completed=bool(d["recovery_completed_event"][0]),
                    stable=bool(d["stable"][0]), faults=faults)
                events.append(row)
                for tag, value in (("tilt_deg", tilt), ("height_m", height),
                                   ("sweep_rad", sweep.angle), ("completed", row["completed"])):
                    writer.add_scalar("SynchronizedRollover/"+tag, value, len(events))
                if bool(done[0]) or stable_belly is not None:
                    break
                if runtime.enabled_at is not None and runtime.time-runtime.enabled_at >= args.seconds-1e-9:
                    break
                if runtime.enabled_at is None and runtime.time > settings["passive_max_seconds"]+.02:
                    raise RuntimeError("Native passive window never enabled")
        records = runtime.records
        whole = contact_summary(records, env.robot.body_names, 0., None)
        disabled = contact_summary(records, env.robot.body_names, 0., runtime.enabled_at)
        active_contact = contact_summary(records, env.robot.body_names, runtime.enabled_at, None)
        finite = all(np.isfinite(np.asarray([r[k] for r in records])).all() for k in
                     ("joint_position", "joint_velocity", "root_pose", "contact_force", "motor_effort"))
        report.update(status="stopped" if stopped else "completed", finite=bool(finite),
            enable_time_s=runtime.enabled_at, duration_s=runtime.time, physical_frames=len(records),
            first_upright_from_enable_s=first_upright-runtime.enabled_at if first_upright is not None else None,
            belly_capture_from_enable_s=stable_belly-runtime.enabled_at if stable_belly is not None else None,
            belly_capture=stable_belly is not None, final_event=events[-1],
            best_active_tilt_deg=min((r["tilt_deg"] for r in events if r["active"]), default=None),
            original_recovery_completed=any(r["completed"] for r in events),
            whole_contact=whole, disabled_contact=disabled, active_contact=active_contact,
            joint_names=list(env.robot.joint_names), body_names=list(env.robot.body_names),
            same_controller_objects=wheel_object is env.identified_wheels and motor_object is env.actuator_response,
            initial_history=initial_history, final_history=history_snapshot(env),
            suppressed_terminal_resets=env.suppressed_resets,
            native_dt_error_s=max(abs(r["dt"]-r["native_dt_s"]) for r in records))
        for group, value in whole["group_positive_world_z_20ms_peak_ns"].items():
            writer.add_scalar("SynchronizedRollover/whole_impulse20ms_"+group+"_Ns", value, 0)
        write_json(output/"policy_events.json", events)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        print(report["error"], file=sys.stderr, flush=True)
        raise
    finally:
        if runtime is not None and runtime.records:
            np.savez_compressed(output/"physical_trajectory.npz",
                **{k: np.asarray([r[k] for r in runtime.records]) for k in runtime.records[0]})
        report["source_mismatches_at_exit"] = [n for n, h in hashes.items() if digest(ROOT/n) != h]
        write_json(output/"report.json", report)
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()

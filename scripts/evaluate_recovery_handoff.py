#!/usr/bin/env python3
"""Independent one-case recovery -> fixed16001 evaluation, without physical resets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from play_prone_motion import parser, load_bundle, prepare, build_actor, case_for
from train_scratch_standing import digest, write_json


def main():
    cli = parser()
    cli.description = __doc__
    cli.add_argument("--mode", choices=("recovery", "standing-reference"), default="recovery")
    cli.add_argument("--standing-policy", type=Path, required=True)
    args = cli.parse_args()
    if args.num_envs != 1 or args.loop or args.domain_randomization != "off" or not args.headless:
        cli.error("Independent handoff currently requires one world, headless, domain-randomization off and no loop")
    if args.mode == "standing-reference":
        args.scene, args.birth_height = "stand", .305
    elif args.scene != "prone":
        cli.error("Recovery must start from its explicit fallen case")
    import numpy as np
    import torch
    import onnxruntime as ort
    from tensordict import TensorDict
    from wheeled_tasks.chassis.prone_motion_cli import output_path
    contract = json.loads((ROOT/"contracts/wheel_leg_recovery_handoff_evaluation_v1.json").read_text())
    policy = args.standing_policy.expanduser().resolve(strict=True)
    if digest(policy) != contract["standing"]["policy_sha256"]:
        cli.error("Standing policy is not the fixed16001 ONNX")
    bundle = load_bundle(args)
    output = output_path(args, ROOT, training=False)
    output.mkdir(parents=True, exist_ok=False)
    cfg, settings, control = prepare(args, bundle, output)
    observation = TensorDict(dict(policy=torch.zeros(1, 35), critic=torch.zeros(1, 87)), batch_size=[1])
    actor = build_actor(observation, bundle, "cpu")
    session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    dummy = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 35), dtype=np.float32)})[0]
    if dummy.shape != (1, 6) or not np.isfinite(dummy).all():
        raise ValueError("Fixed standing ONNX ABI differs")
    tracked = subprocess.check_output(["git", "ls-files", "src", "scripts", "contracts"], cwd=ROOT, text=True).splitlines()
    hashes = {name: digest(ROOT/name) for name in tracked if (ROOT/name).is_file()}
    report = dict(schema="wheel-leg-recovery-handoff-result-v1", status="preflight_completed", mode=args.mode,
        checkpoint=str(bundle["checkpoint"]), checkpoint_sha256=digest(bundle["checkpoint"]),
        checkpoint_source=bundle["info"]["source_revision"], checkpoint_updates=bundle["state"]["iter"],
        evaluation_source=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=hashes, standing_policy=str(policy), standing_policy_sha256=digest(policy),
        contract_sha256=digest(ROOT/"contracts/wheel_leg_recovery_handoff_evaluation_v1.json"),
        new_PPO=0, hardware_io=False, actual_handoff=False, recovery_completed=False,
        behavior_qualified=False, post_handoff_seconds=0., enable_to_handoff_s=None,
        terminal_physical_reset_suppressed=0)
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
        from wheeled_tasks.chassis.handoff_evaluation import HandoffEvaluation, construct_evaluation_env, contact_summary
        from torch.utils.tensorboard import SummaryWriter
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        class ContinuousCaseEnv(ProneMotionEnv):
            case_armed = False
            suppressed_retirement_resets = 0
            def reset(self, ids, **kwargs):
                if self.case_armed:
                    # One case only. Preserve the original terminal evidence,
                    # but never reseed a retired case before returning it.
                    self.suppressed_retirement_resets += 1
                    return
                if getattr(self, "course_ready", False) and self.evaluation_cases is None:
                    # The constructor's last reset already belongs to this
                    # cold case. A single-world evaluator must not invoke the
                    # training population allocator, which needs two lanes.
                    self.evaluation_cases = [case_for(args, self, bundle["plan"])]
                return super().reset(ids, **kwargs)
        manifest = json.loads((ROOT/cfg["asset_directory"]/"manifest.json").read_text())
        env = construct_evaluation_env(ContinuousCaseEnv, cfg, manifest, control, ROOT, stage_name="speed", num_envs=1,
            device=args.device, seed=bundle["seed"], level=1., run_role="playback", bank=bundle["bank"], settings=settings)
        env.actor_updates = bundle["state"]["iter"]
        env.evaluation_cases = [case_for(args, env, bundle["plan"])]
        env.reset(torch.arange(1, device=env.device))
        env.case_armed = True
        runtime = HandoffEvaluation(env, standing_reference=args.mode == "standing-reference")
        actor = actor.to(args.device)
        observations = env.get_observations()
        writer = SummaryWriter(str(output/"events"))
        report.update(status="running", admission=runtime.admission, selected_case=env.evaluation_cases[0],
            initial_physics_dt=cfg["physics_dt"], post_physics_dt=.005, recovery_policy_dt=.01, standing_policy_dt=.02)
        write_json(output/"report.json", report)
        with torch.inference_mode():
            active_limit = min(bundle["seconds"], 20.)
            while not runtime.recovery_budget_exhausted(active_limit) and runtime.mode == "recovery" and not stopped:
                action = actor(observations)
                runtime.controller_frames.append(dict(time=runtime.time, controller="recovery", policy_dt=env.policy_dt,
                    observation=observations["policy"][0].cpu().tolist(), raw_action=action[0].cpu().tolist()))
                observations, _, done, extra = env.step(action)
                runtime.observe_recovery(extra["diagnostics"])
                writer.add_scalar("ContinuousRecovery/completed_event", float(extra["diagnostics"]["recovery_completed_event"][0]), len(runtime.policy_records))
                if bool(done[0]):
                    break
            if runtime.mode != "recovery":
                followup_start = runtime.time
                # A later failure stays in the full fixed window; no suffix selection.
                for tick in range(500):
                    if stopped:
                        break
                    frame = runtime.standing_observations()
                    raw = session.run(None, {session.get_inputs()[0].name: frame.cpu().numpy()})[0]
                    runtime.controller_frames.append(dict(time=runtime.time, controller="fixed16001", policy_dt=.02,
                        observation=frame[0].cpu().tolist(), raw_action=raw[0].tolist()))
                    runtime.step_standing(torch.as_tensor(raw, device=env.device))
                report["post_handoff_seconds"] = runtime.time-followup_start
            records = runtime.records
            report.update(status="stopped_unqualified" if stopped else "completed_unqualified",
                actual_handoff=runtime.handoff_at is not None,
                recovery_completed=any(r["completed"] for r in runtime.policy_records),
                enable_to_handoff_s=(runtime.handoff_at-runtime.enabled_at
                    if runtime.handoff_at is not None and runtime.enabled_at is not None else None),
                transition=runtime.transition, original_termination=runtime.original_termination,
                terminal_physical_reset_suppressed=env.suppressed_retirement_resets,
                simulation_seconds=runtime.time, physical_samples=len(records),
                joint_names=list(env.robot.joint_names), body_names=list(env.robot.body_names),
                enable_time_s=runtime.enabled_at, handoff_time_s=runtime.handoff_at,
                standing_reference_is_self_righting=False)
            report["recovery_active_window_s"] = (max(0., (runtime.handoff_at if runtime.handoff_at is not None
                else runtime.time)-runtime.enabled_at) if runtime.enabled_at is not None else None)
            report["recovery_active_budget_s"] = active_limit
            report["recovery_stop_reason"] = ("stopped" if stopped else "actual_handoff" if runtime.handoff_at is not None
                else "standing_reference" if args.mode == "standing-reference" else "original_done" if runtime.original_termination
                else "active_budget_exhausted" if runtime.enabled_at is not None else "passive_enable_timeout")
            if records:
                post = [r for r in records if r["standing"]]
                if post:
                    xyz = np.asarray([r["root_pose"][:3] for r in post])
                    gravity = np.asarray([r["gravity"] for r in post])
                    tilt = np.degrees(np.arccos(np.clip(-gravity[:, 2], -1., 1.)))
                    report.update(post_tilt_peak_deg=float(tilt.max()), post_height_min_m=float(xyz[:, 2].min()),
                        post_xy_excursion_m=float(np.linalg.norm(xyz[:, :2]-xyz[0, :2], axis=-1).max()))
                finite = all(np.isfinite(np.asarray([r[k] for r in records])).all()
                    for k in ("joint_position", "joint_velocity", "motor_effort", "physical_effort", "contact_force", "root_pose"))
                report["finite"] = bool(finite)
                report["native_dt_max_error_s"] = max(abs(r["native_dt_s"]-r["dt"]) for r in records)
                report["native_clock_scope"] = "dt forwarded to the same native simulate object; committed only after fetch_results; checked against SimulationContext step count"
                report["contact"] = contact_summary(records, env.robot.body_names, runtime.enabled_at, runtime.handoff_at)
                if report["contact"]["measurement_available"]:
                    for group, value in report["contact"]["group_positive_world_z_20ms_peak_ns"].items():
                        writer.add_scalar("ContinuousRecovery/impulse20ms_"+group+"_Ns", value, 0)
            report["screen_pass"] = bool(report["actual_handoff"] and report.get("finite")
                and report["post_handoff_seconds"] >= 10.-1e-8
                and report.get("post_tilt_peak_deg", float("inf")) < 20.
                and report.get("post_height_min_m", 0.) > .23)
            writer.add_scalar("ContinuousRecovery/actual_handoff", float(report["actual_handoff"]), 0)
            writer.add_scalar("ContinuousRecovery/failure_denominator", 1, 0)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        print(report["error"], file=sys.stderr, flush=True)
        raise
    finally:
        if runtime is not None:
            np.savez_compressed(output/"physical_trajectory.npz",
                **{k: np.asarray([r[k] for r in runtime.records]) for k in runtime.records[0]} if runtime.records else {})
            write_json(output/"policy_events.json", runtime.policy_records)
            write_json(output/"controller_frames.json", runtime.controller_frames)
            if runtime.transition is not None:
                write_json(output/"transition.json", runtime.transition)
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

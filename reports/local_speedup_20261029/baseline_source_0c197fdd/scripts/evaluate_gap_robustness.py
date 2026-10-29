#!/usr/bin/env python3
"""Bounded native checks of real-entry legs and explicit IMU mounting errors."""
import argparse
import faulthandler
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from train_robust_getup import prepare, evaluate, write_json, digest, fixed_cases


def cases(bank, replicas=1, group="all"):
    def profile(pitch=0., roll=0., wobble=0.):
        return dict(fixed_degrees=[roll, pitch, 0.], amplitude_degrees=[wobble, wobble],
                    frequency_hz=[5., 3.], phase_rad=[0., .5])
    profiles = dict(clean=profile(), pitch_plus2=profile(pitch=2.), pitch_minus2=profile(pitch=-2.),
        roll_plus2=profile(roll=2.), roll_minus2=profile(roll=-2.),
        wobble=profile(wobble=.5), combined=profile(pitch=2., wobble=.5))
    # Appending training poses must not silently resample the historical
    # evaluation's symmetric fallback pose from a larger RNG choice pool.
    original_count = bank.get("extension", {}).get("original_pose_count", len(bank["poses"]))
    if not isinstance(original_count, int) or not 0 < original_count <= len(bank["poses"]):
        raise ValueError("Invalid legacy evaluation pose population")
    full = fixed_cases(dict(bank, poses=bank["poses"][:original_count]), 35, 0)
    templates = []
    for i, pose in enumerate(bank["poses"]):
        if pose.get("real_start_reconstruction"):
            templates.append((dict(name=pose["name"], pose_index=i, recovery=True,
                                   command=[0., 0., .305]), ["clean", "pitch_plus2"]))
    for name in ("left_symmetric", "right_symmetric", "inverted_symmetric"):
        templates.append((deepcopy(next(c for c in full if c["name"] == name)), ["clean", "combined"]))
    for name in ("stand_low", "stand", "stand_high"):
        templates.append((deepcopy(next(c for c in full if c["name"] == name)), list(profiles)))
    for name in ("forward", "backward", "arc_forward_left", "arc_forward_right"):
        templates.append((deepcopy(next(c for c in full if c["name"] == name)), ["clean", "pitch_plus2", "combined"]))
    if group == "rmcs-inverted":
        templates = [(dict(name=pose["name"], pose_index=i, recovery=True, command=[0., 0., .305]),
                      ["clean", "combined"])
                     for i, pose in enumerate(bank["poses"])
                     if pose.get("rmcs_reference") and (abs(pose["roll_deg"]) == 180. or abs(pose["pitch_deg"]) == 180.)]
        if not templates:
            raise ValueError("RMCS inverted evaluation requires the revalidated reference bank")
    result = []
    for _ in range(replicas):
        for template, names in templates:
            for name in names:
                case = deepcopy(template)
                case.update(name=template["name"]+"__"+name, behavior_case=template["name"], imu_case=name,
                            imu_mount=profiles[name])
                result.append(case)
    return result


def summarize(output, step):
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    report = json.loads((output/"report.json").read_text())
    data = np.load(output/"trajectory.npz")
    rows = []
    writer = SummaryWriter(str(output/"metrics"), filename_suffix=".gap-evaluation")
    for lane in report["lanes"]:
        i = lane["lane"]
        alive = data["alive"][:, i] & ~data["passive"][:, i]
        times = data["time"][:, i]
        indices = np.flatnonzero(alive)
        row = dict(lane=i, name=lane["name"], behavior_case=lane["behavior_case"], imu_case=lane["imu_case"],
            survived=lane["survived"], recovered=lane["recovered"], termination_causes=lane["termination_causes"])
        if len(indices):
            first = indices[0]
            initial = alive & (times <= times[first]+.2)
            tail = alive & (times >= report["seconds"]-4.)
            position = data["local_position"][:, i, :2]
            row.update(max_leg_speed_rad_s=float(data["leg_speed_peak_rad_s"][alive, i].max()),
                max_leg_effort_nm=float(data["leg_effort_peak_nm"][alive, i].max()),
                first_200ms_leg_effort_peak_nm=float(data["leg_effort_peak_nm"][initial, i].max()),
                saturation_fraction=float(data["leg_saturation_fraction"][alive, i].mean()),
                longest_saturation_s=float(data["leg_saturation_run_s"][alive, i].max()),
                effort_slew_rms_nm=float(np.sqrt(np.mean(data["leg_effort_slew_rms_nm"][alive, i]**2))),
                max_position_excursion_m=float(np.linalg.norm(position[alive]-position[first], axis=-1).max()),
                tail_stable_fraction=float(data["stable"][tail, i].mean()) if tail.any() else 0.,
                tail_leg_speed_rms_rad_s=float(np.sqrt(np.mean(data["motor_velocity"][tail, i][:, [0, 1, 3, 4]]**2))) if tail.any() else None,
                tail_leg_effort_rms_nm=float(np.sqrt(np.mean(data["motor_effort"][tail, i][:, [0, 1, 3, 4]]**2))) if tail.any() else None,
                tail_saturation_fraction=float(data["leg_saturation_fraction"][tail, i].mean()) if tail.any() else None,
                vx_mae_m_s=float(np.abs(data["velocity"][alive, i, 0]-data["commands"][alive, i, 0]).mean()),
                yaw_mae_rad_s=float(np.abs(data["omega"][alive, i, 2]-data["commands"][alive, i, 1]).mean()),
                tail_abs_vx_m_s=float(np.abs(data["velocity"][tail, i, 0]).mean()) if tail.any() else None)
            if tail.sum() > 200:
                leg = data["motor_position"][tail, i][:, [0, 1, 3, 4]]
                spectrum = np.fft.rfft(leg-leg.mean(0), axis=0)
                frequencies = np.fft.rfftfreq(len(leg), .005)
                spectrum[(frequencies < 3.) | (frequencies > 95.)] = 0.
                row["tail_leg_motion_3_95hz_rms_rad"] = float(np.sqrt(np.mean(np.fft.irfft(spectrum, n=len(leg), axis=0)**2)))
        for key, value in row.items():
            if isinstance(value, (int, float)) and key != "lane" and value is not None:
                writer.add_scalar("GapEvaluation/"+row["name"]+"/"+key, value, step)
        rows.append(row)
    writer.close()
    summary = dict(schema="real-gap-evaluation-v1", rows=rows, episodes=len(rows),
        survived=sum(r["survived"] for r in rows), recovered=sum(r["recovered"] for r in rows),
        full_training_admitted=False, hardware_ready=False, trajectory_sha256=digest(output/"trajectory.npz"))
    write_json(output/"gap_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=ROOT/"reports/real_robot_robustness_20261024/pose_bank_gap002.json")
    parser.add_argument("--seed", type=int, default=241001)
    parser.add_argument("--severity", type=float, default=1.)
    parser.add_argument("--friction-min",type=float,default=.3)
    parser.add_argument("--friction-max",type=float,default=.5)
    parser.add_argument("--actuator-timing",choices=("stress","native"),default="stress")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--case-group", choices=("all", "standing", "entry", "rmcs-inverted"), default="all")
    parser.add_argument("--nominal-dynamics", action="store_true",
                        help="Separate explicit IMU errors from mass/COM/actuator randomization")
    parser.add_argument("--legacy-pd-scaling", action="store_true",
                        help="Explicit pre-fix actuator control for the initial entry-only ablation")
    parser.add_argument("--zero-torque-delay", action="store_true",
                        help="Diagnostic intervention: remove only extra torque transport delay")
    parser.add_argument("--hold-standing-legs", action="store_true",
                        help="Diagnostic only: keep initial measured leg targets in standing cases")
    parser.add_argument("--sampled-policy", action="store_true",
                        help="Diagnostic only: sample the exact checkpoint orbit-v2 training distribution")
    parser.add_argument("--sample-seed", type=int, default=241507,
                        help="Private fixed-shape Gaussian draws, separate from the environment seed")
    args = parser.parse_args()
    if not 1 <= args.replicas <= 4 or not 4. <= args.seconds <= 30.:
        parser.error("Use1..4 replicas and4..30 seconds")
    if args.zero_torque_delay and args.hold_standing_legs:
        parser.error("Use one diagnostic intervention at a time")
    if args.sampled_policy and (args.zero_torque_delay or args.hold_standing_legs or args.legacy_pd_scaling):
        parser.error("Sampled-policy exposure cannot combine other action/actuator interventions")
    if not .2 <= args.friction_min <= args.friction_max <= .5:
        parser.error("Ground friction bounds must be ordered within .2...5")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    args.output.mkdir(parents=True, exist_ok=False)
    fault_log = (args.output/"fault.log").open("w")
    faulthandler.enable(file=fault_log, all_threads=True)
    bank = json.loads(args.bank.read_text())
    fixed = cases(bank, args.replicas, args.case_group)
    if args.case_group in ("standing", "entry"):
        fixed = [c for c in fixed if (c["behavior_case"].startswith("stand") if args.case_group == "standing"
                                     else c["behavior_case"].startswith("real_start"))]
    config_args = argparse.Namespace(output=args.output, bank=args.bank, resume=args.checkpoint,
        no_domains=args.nominal_dynamics, domain_scale=0. if args.nominal_dynamics else .35,
        gap_robustness=True, gap_severity=args.severity,
        gap_actuator_timing=args.actuator_timing,
        friction_min=args.friction_min, friction_max=args.friction_max,
        curriculum_updates=128, quiet_leg_alpha=.25, quiet_wheel_alpha=.65,
        num_envs=len(fixed), rollout_steps=128, mode="evaluate", friction="high", seed=args.seed,
        isolated_origins=True, episode_seconds=max(12., args.seconds))
    cfg, task = prepare(config_args)
    if len(fixed) < len(cfg["scene_groups"]):
        raise ValueError("Native population must cover base groups; increase --replicas")
    if args.legacy_pd_scaling:
        task["gap_correct_pd_scaling"] = False
    task["imu_mount_profiles"] = [c["imu_mount"] for c in fixed]
    contract = json.loads((args.output/"task_contract.json").read_text())
    contract["settings"] = task
    contract["diagnostic_cases"] = fixed
    contract["evaluation_overrides"] = {"torque_transport_delay_ms": 0} if args.zero_torque_delay else {}
    if args.hold_standing_legs:
        contract["evaluation_overrides"] = {"hold_standing_leg_targets": "first_measured"}
    if args.sampled_policy:
        contract["evaluation_overrides"]["sampled_policy_exposure"] = dict(seed=args.sample_seed,
            distribution="checkpoint_orbit_v2_conditioned_normal", eligible_for_mean_qualification=False)
    contract["qualification"] = None
    write_json(args.output/"task_contract.json", contract)
    progress = dict(schema="real-gap-native-v1", status="starting", pid=os.getpid(), started_unix=time.time(),
        seed=args.seed,
        nominal_dynamics=args.nominal_dynamics,
        ground_friction_range=[args.friction_min,args.friction_max],
        actuator_timing_profile=args.actuator_timing,
        evaluation_overrides=contract["evaluation_overrides"],
        checkpoint_sha256=digest(args.checkpoint), optimizer_updates=0, hardware_started=False,
        full_training_admitted=False, cases=len(fixed), seconds=args.seconds, severity=args.severity)
    sources = [Path(__file__).resolve(), ROOT/"scripts/train_robust_getup.py", ROOT/"scripts/export_robust_getup.py"]
    sources += list((ROOT/"src/wheeled_algo").glob("*.py"))
    sources += list((ROOT/"src/wheeled_tasks/chassis").glob("*.py"))
    progress["source_sha256"] = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = args.output/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        progress["source_sha256"][str(relative)] = digest(source)
    write_json(args.output/"progress.json", progress)
    launcher = env = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device="cuda:0", enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import torch
        from wheeled_tasks.chassis.robust_getup_env import RobustGetupEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from export_robust_getup import load_mean
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        mean, info = load_mean(args.checkpoint)
        progress["checkpoint"] = str(args.checkpoint)
        progress["actor_settings"] = info["settings"]
        manifest = json.loads((ROOT/cfg["asset_directory"]/"manifest.json").read_text())
        env = RobustGetupEnv(cfg, manifest, load_chassis_control(ROOT/cfg["control_math_source"]), ROOT,
            stage_name="speed", num_envs=len(fixed), device="cuda:0", seed=args.seed,
            level=1., run_role="train", bank=args.bank, settings=task, fixed_cases=fixed)
        env.evaluation_checkpoint = args.checkpoint
        env.actor_variant = "robust"
        env.evaluation_trace_directory = args.output/"partial_trace"
        env.evaluation_overrides = contract["evaluation_overrides"]
        progress["actuator_model"] = "scaled-PD-before-clip" if task.get("gap_correct_pd_scaling") else "legacy-post-clip-correction"
        mean = mean.to("cuda:0")
        sampled = None
        if args.sampled_policy:
            from wheeled_algo.gap_recovery_exploration import SampledGapPolicyEvaluation
            checkpoint_state = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
            sampled = SampledGapPolicyEvaluation(mean, info["settings"],
                checkpoint_state["actor_state_dict"]["mlp.log_std"], args.sample_seed).to("cuda:0")
            mean = sampled
        if args.hold_standing_legs:
            class HoldStandingLegs(torch.nn.Module):
                def __init__(self, original):
                    super().__init__()
                    self.original = original
                    self.register_buffer("mask", torch.tensor(
                        [c["behavior_case"].startswith("stand") for c in fixed],device="cuda:0")[:,None])
                    self.initial = None

                def forward(self, obs):
                    action = self.original(obs)
                    if self.initial is None:
                        self.initial = (obs[:,10:14]/.25).clone()
                    return torch.cat((torch.where(self.mask,self.initial,action[:,:4]),action[:,4:]),-1)
            mean = HoldStandingLegs(mean)
        write_json(args.output/"startup.json", env.startup_report)
        progress["status"] = "evaluating"
        write_json(args.output/"progress.json", progress)
        evaluate(env, mean, args.output/"evaluation", torch, seed=args.seed, seconds=args.seconds, cases=fixed)
        if sampled is not None:
            report_path = args.output/"evaluation/report.json"
            native_report = json.loads(report_path.read_text())
            native_report["sampled_policy_exposure_diagnostic"] = sampled.sampling_receipt()
            write_json(report_path, native_report)
        progress.update(status="completed", summary=summarize(args.output/"evaluation", info["completed_updates"]))
    except BaseException:
        progress.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        progress["finished_unix"] = time.time()
        write_json(args.output/"progress.json", progress)
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()

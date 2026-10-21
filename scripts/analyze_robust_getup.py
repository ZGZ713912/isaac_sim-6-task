#!/usr/bin/env python3
"""Physical-unit episode metrics, including policy-rate standing vibration."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hf_rms(values, dt, min_hz=20.):
    """RMS in the 20..95 Hz band; only continuous 200 Hz windows qualify."""
    if len(values) < 200 or not .0049 < dt < .0051:
        return None
    values = values.astype(np.float64)
    values -= values.mean(0)
    # A Hann window avoids mistaking a slow drift/end-point discontinuity for
    # actuator chatter. Window-energy normalization retains physical units.
    window = np.hanning(len(values))[:, None]
    transformed = np.fft.rfft(values * window, axis=0)
    frequency = np.fft.rfftfreq(len(values), dt)
    selected = (frequency >= min_hz) & (frequency <= 95.)
    energy = 2. * (np.abs(transformed[selected]) ** 2).sum(0) / (len(values) * (window ** 2).sum())
    return float(np.sqrt(energy.mean()))


def analyze(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    if digest(directory / "trajectory.npz") != report["trajectory_sha256"]:
        raise ValueError("Evaluation trajectory changed")
    data = dict(np.load(directory / "trajectory.npz"))
    buckets = defaultdict(list)
    practical_by_lane = [False] * len(report["lanes"])
    for original in report["lanes"]:
        row = dict(original)
        lane = row["lane"]
        time = data["time"][:, lane]
        alive = data["alive"][:, lane]
        dt = float(np.median(np.diff(time)))
        # The original uninterrupted 400-frame latch remains in the report.
        # This additional practical endpoint tolerates isolated contact/dq
        # threshold flicker while requiring a completed episode and real motion.
        last = alive & (time > report["seconds"] - 2.)
        practical = False
        if row["recovered"] and row["survived"] and last.sum() >= 399:
            error_vx = np.abs(data["velocity"][last, lane, 0] - data["commands"][last, lane, 0])
            error_yaw = np.abs(data["omega"][last, lane, 2] - data["commands"][last, lane, 1])
            good = data["stable"][last, lane] & (error_vx < .25) & (error_yaw < .4)
            practical = bool(good.mean() >= .95)
        row["practical_followup_success"] = practical
        practical_by_lane[lane] = practical
        quiet = (alive & (time >= 3.) & (np.abs(data["commands"][:, lane, :2]).max(-1) < .01))
        if row["name"].startswith("stand") and quiet.sum() >= 200:
            q = data["motor_position"][quiet, lane]
            dq = data["motor_velocity"][quiet, lane]
            effort = data["motor_effort"][quiet, lane]
            action = data["actions"][quiet, lane]
            row["quiet"] = dict(frames=int(quiet.sum()), interval_s=float(quiet.sum() * dt),
                vx_rms_m_s=float(np.sqrt(np.mean(data["velocity"][quiet, lane, 0] ** 2))),
                yaw_rms_rad_s=float(np.sqrt(np.mean(data["omega"][quiet, lane, 2] ** 2))),
                height_mae_m=float(np.abs(data["height"][quiet, lane] - data["commands"][quiet, lane, 2]).mean()),
                leg_speed_rms_rad_s=float(np.sqrt(np.mean(dq[:, [0, 1, 3, 4]] ** 2))),
                wheel_speed_rms_rad_s=float(np.sqrt(np.mean(dq[:, [2, 5]] ** 2))),
                leg_target_hf_rms_rad=hf_rms(action[:, [0, 1, 3, 4]] * .25, dt),
                wheel_target_hf_rms_rad_s=hf_rms(action[:, [2, 5]] * 10., dt),
                leg_effort_hf_rms_nm=hf_rms(effort[:, [0, 1, 3, 4]], dt),
                wheel_effort_hf_rms_nm=hf_rms(effort[:, [2, 5]], dt),
                leg_effort_delta_rms_nm=float(np.sqrt(np.mean(np.diff(effort[:, [0, 1, 3, 4]], axis=0) ** 2))))
            row["quiet"].update(
                leg_target_fast_rms_rad_3_95=hf_rms(action[:, [0, 1, 3, 4]] * .25, dt, 3.),
                wheel_target_fast_rms_rad_s_3_95=hf_rms(action[:, [2, 5]] * 10., dt, 3.),
                leg_motion_fast_rms_rad_3_95=hf_rms(q[:, [0, 1, 3, 4]], dt, 3.),
                wheel_motion_fast_rms_rad_s_3_95=hf_rms(dq[:, [2, 5]], dt, 3.))
            if "local_position" in data:
                xy = data["local_position"][quiet, lane, :2]
                row["quiet"]["body_xy_max_excursion_m"] = float(np.linalg.norm(xy - xy[:1], axis=-1).max())
        # Report actual integrated tracking errors, not just the terminal pose.
        selected = alive & (time >= 2.) & ~data["passive"][:, lane]
        if selected.any():
            vx = data["velocity"][selected, lane, 0]
            yaw = data["omega"][selected, lane, 2]
            command = data["commands"][selected, lane]
            row["both_axes_tracking_fraction"] = float(((np.abs(vx - command[:, 0]) < .25)
                                                       & (np.abs(yaw - command[:, 1]) < .4)).mean())
        buckets[row["name"]].append(row)
    cases = {}
    settled = defaultdict(lambda: dict(episodes=0, recovered=0, recovery_and_motion_success=0,
                                      practical_recovery_and_motion_success=0))
    for name, rows in buckets.items():
        case = dict(episodes=len(rows), survived=sum(r["survived"] for r in rows),
            recovered=sum(r["recovered"] for r in rows),
            recovery_and_motion_success=sum(r["followup_success"] and r["survived"] for r in rows),
            practical_recovery_and_motion_success=sum(r["practical_followup_success"] for r in rows),
            in_motion_falls=sum(r.get("in_motion_fall", False) for r in rows),
            recovered_after_push=sum(r.get("recovered_after_push", False) for r in rows),
            max_gap_mm=max(r["max_gap_m"] for r in rows) * 1000.)
        for key in ("vx_mean_m_s", "yaw_mean_rad_s", "vx_mae_m_s", "yaw_mae_rad_s", "both_axes_tracking_fraction"):
            values = [r[key] for r in rows if r.get(key) is not None]
            case[key] = float(np.mean(values)) if values else None
        quiet = [r["quiet"] for r in rows if "quiet" in r]
        if quiet:
            case["quiet"] = {key: float(np.mean([r[key] for r in quiet if r[key] is not None]))
                             for key in quiet[0]}
            case["quiet"]["episodes_with_valid_windows"] = len(quiet)
        cases[name] = case
        for row in rows:
            if row["recovery"]:
                f = settled[row["post_settle_family"] or "not_enabled"]
                f["episodes"] += 1
                f["recovered"] += int(row["recovered"])
                f["recovery_and_motion_success"] += int(row["followup_success"] and row["survived"])
                f["practical_recovery_and_motion_success"] += int(row["practical_followup_success"])
    return dict(schema="robust-getup-behavior-analysis-v2", evaluation=str(directory.resolve()),
        report_sha256=digest(directory / "report.json"), checkpoint_sha256=report["checkpoint_sha256"],
        seed=report["seed"], actor_variant=report["actor_variant"],
        motion_episodes=report["motion_episodes"], motion_survived=report["motion_survived"],
        actual_settled_attitudes=dict(settled), cases=cases, passive_motor_peak_nm=report["passive_motor_peak_nm"],
        practical_followup_by_lane=practical_by_lane,
        practical_moving_fall_recoveries=sum(practical_by_lane[row["lane"]] and row.get("push", False)
            and row.get("in_motion_fall", False) and row.get("recovered_after_push", False) for row in report["lanes"]),
        practical_definition="completed episode; last 2 s >=95% stable and abs(vx error)<.25 m/s and abs(yaw error)<.4 rad/s",
        max_gap_m=report["max_gap_m"], signal_sampling_hz=200,
        vibration_scope="20..95 Hz from 200 Hz simulation samples; not a CAN/hardware measurement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args()
    result = analyze(args.evaluation)
    (args.evaluation / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        run = args.evaluation.parent
        progress = json.loads((run / "progress.json").read_text())
        step = 0 if args.evaluation.name == "baseline_eval" else progress["completed_updates"]
        with SummaryWriter(log_dir=str(run), filename_suffix=".behavior") as writer:
            comparison = args.evaluation.name not in ("evaluation", "baseline_eval")
            for name, case in result["cases"].items():
                prefix = (("Comparison/" + result["actor_variant"] + "/") if comparison else "Behavior/") + name + "/"
                for key, value in case.items():
                    if isinstance(value, (float, int)):
                        writer.add_scalar(prefix + key, value, step)
                for key, value in case.get("quiet", {}).items():
                    writer.add_scalar(prefix + key, value, step)
            writer.add_scalar("Behavior/passive_motor_peak_nm", result["passive_motor_peak_nm"], step)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "practical_followup_by_lane")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

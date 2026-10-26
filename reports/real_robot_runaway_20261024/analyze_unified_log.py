"""Read-only diagnosis of the unified115 physical robot CSV; no motor or PPO calls.

Run with the bundled analysis Python. Add --model-dir when using an interpreter
with onnxruntime to reproduce the optional output-signature comparison.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


NAMES = ["left_hip_joint", "left_knee_joint", "right_hip_joint", "right_knee_joint",
         "left_wheel", "right_wheel"]
NOMINAL = np.array([-.42, .13742282595395358, .42, -.1374155762580851])
OFFSETS = np.array([1.6, 2.93, -1.6, -2.93])
HINGE_MIN = -.140638855
HINGE_MARGIN = .03
TARGET_BOUNDS = np.array([[-.14062820701680934, 1.3692123767724917],
                          [-.1406283965709381, 1.3692125038958158]])


def quantile(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return dict(zip(["min", "p50", "p95", "max"], np.percentile(x, [0, 50, 95, 100]).tolist()))


def gates(o):
    original = np.clip((o[:, 9] + .94) / .44, 0, 1)
    severe = np.clip((o[:, 9] + .5) / (.5 - .08715574274765817), 0, 1)
    phase = np.clip((np.max(abs(o[:, 10:14]), axis=1) - 1) / .8, 0, 1)
    pair = o[:, 10:14].reshape(-1, 2, 2)
    d = pair[:, :, 1] - pair[:, :, 0]
    shape = np.clip((np.max(abs(np.arctan2(np.sin(d), np.cos(d))), axis=1) - .2) / .2, 0, 1)
    recovery = np.maximum(severe, np.maximum(phase, shape) * original)
    request = np.maximum(abs(o[:, 0]) / .12, abs(o[:, 2]) / .25)
    tilt = np.clip((-o[:, 9] - .94) / .04, 0, 1)
    gyro = np.clip(np.linalg.norm(2 * o[:, 4:6], axis=1) - .4, 0, 1)
    quiet = np.clip(1 - request, 0, 1) * tilt * (1 - gyro)
    return quiet, recovery


def target_positions(raw, q):
    """Exact paired107 target winding used by host PD, excluding torque limits."""
    desired = NOMINAL + raw[:, :4] * .25
    result = np.empty_like(desired)
    for side, sign in enumerate([1, -1]):
        hip, aux = 2 * side, 2 * side + 1
        mean = desired[:, [hip, aux]].mean(axis=1)
        delta = np.clip(sign * (desired[:, aux] - desired[:, hip]), *TARGET_BOUNDS[side])
        turns = np.floor((q[:, [hip, aux]].mean(axis=1) - mean) / (2 * np.pi) + .5)
        result[:, hip] = mean - sign * delta / 2 + turns * 2 * np.pi
        result[:, aux] = mean + sign * delta / 2 + turns * 2 * np.pi
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/home/noir/wheel_leg_unified_rl_2026-10-23_18-05-59.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--pose-bank", type=Path, default=Path(__file__).resolve().parents[1] / "unified_robust_20261021/pose_bank002.json")
    args = parser.parse_args()
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    rows, excluded = [], []
    with args.source.open(newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                excluded.append(dict(line=line, columns=len(row), expected_columns=len(header),
                                     index=row[0] if row else None, reason="incomplete row"))
                continue
            rows.append([float(x) for x in row])
    a = np.array(rows)
    ix = {name: i for i, name in enumerate(header)}
    col = lambda key: a[:, ix[key]]
    group = lambda prefix, suffix="": a[:, [ix[prefix + name + suffix] for name in NAMES]]
    obs = a[:, [ix[f"/wheel_leg/rl/observation/frame/element_{i}"] for i in range(35)]]
    q = group("/wheel_leg/rl/model/", "/angle")
    dq = group("/wheel_leg/rl/model/", "/velocity")
    raw = group("/wheel_leg/rl/action/")
    tau = group("/wheel_leg/", "/control_torque")
    api_q = group("/wheel_leg/", "/angle")
    # A sub-sampled log does not identify the actual CAN packet period.
    # Wheel feedback provides a monotonic clock close to each sampled row.
    clock = np.maximum(col("/wheel_leg/left_wheel/feedback_steady_ns"),
                       col("/wheel_leg/right_wheel/feedback_steady_ns")) * 1e-9
    quiet, recovery = gates(obs)
    active = col("/wheel_leg/rl/state") == 3
    spans = np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2)
    targets = target_positions(raw, q)
    hinge = np.c_[q[:, 1] - q[:, 0], q[:, 2] - q[:, 3]]
    hardware_q = -api_q[:, :4] + OFFSETS
    hardware_hinge = np.c_[hardware_q[:, 1] - hardware_q[:, 0], hardware_q[:, 2] - hardware_q[:, 3]]
    pitch = np.degrees(np.arcsin(np.clip(obs[:, 7], -1, 1)))
    roll = np.degrees(np.arctan2(-obs[:, 8], -obs[:, 9]))
    wheel_proxy = .03 * (dq[:, 4] + dq[:, 5])
    result = dict(schema="physical-unified115-log-diagnosis-v1",
        source=str(args.source), source_sha256=source_sha, complete_rows=len(a),
        columns=len(header), excluded_rows=excluded, source_modified=False,
        hardware_started=False, optimizer_updates=0,
        clock="max(left/right wheel feedback steady_ns); nearby feedback time, not exact executor timestamp",
        clock_span_seconds=float(clock[-1] - clock[0]),
        sampled_row_dt_seconds=quantile(np.diff(clock)),
        policy_rate_hz=np.unique(col("/wheel_leg/rl/policy_rate")).tolist(),
        nominal_leg_position_rad=NOMINAL.tolist(),
        nominal_hinge_rad=[float(NOMINAL[1] - NOMINAL[0]), float(NOMINAL[2] - NOMINAL[3])],
        measured_hinge_rejection_below_rad=HINGE_MIN - HINGE_MARGIN,
        wheel_speed_note="Wheel rolling speed proxy only. Not independently measured chassis speed; signs, wheel contact and slip need physical verification.",
        protection_note="Four fault edges coincide with right hinge crossing the current calibrated lower rejection bound. CSV lacks a fault reason code; console fault text was not available.",
        joint_order=NAMES, episodes=[])
    result["user_confirmed_conditions"] = dict(
        start="Legs disabled, robot resting upright on the ground, then full RL enabled",
        imu="IMU mounted on the mechanically loose electronics platform",
        platform_mode_frequency_hz=None, independent_rigid_chassis_attitude_available=False)
    result["calibration_consistency"] = dict(
        equation="q_model = -q_api + [1.6,2.93,-1.6,-2.93]",
        maximum_active_leg_position_residual_rad=float(np.max(abs(q[active, :4] - hardware_q[active]))),
        note="Checks agreement with recorded preprocessing, not correctness against physical mechanism zeroes")
    if args.pose_bank.exists():
        bank = json.loads(args.pose_bank.read_text())
        upright = [p for p in bank["poses"] if p["family"] == "upright"]
        bank_q = np.array([[p["full_joint_positions"][key] for key in
                           ["L_joint1", "LL_joint1", "R_joint1", "RR_joint1"]] for p in upright])
        bank_delta = np.c_[bank_q[:, 1] - bank_q[:, 0], bank_q[:, 2] - bank_q[:, 3]]
        result["initial_pose_bank_audit"] = dict(path=str(args.pose_bank),
            sha256=hashlib.sha256(args.pose_bank.read_bytes()).hexdigest(),
            pose_count=len(bank["poses"]), upright_pose_count=len(upright),
            upright_seed_hinge_min_rad=bank_delta.min(axis=0).tolist(),
            upright_seed_hinge_max_rad=bank_delta.max(axis=0).tolist(),
            ordinary_reset="Non-recovery reset fixes upright_pool[1], knee65_65; height anchor modes are separate",
            limitation="Initial seed coverage only; does not establish every physical state encountered during training")
    cases = []
    for number, (s, e) in enumerate(spans, 1):
        t = clock[s:e] - clock[s]
        first = lambda mask: float(t[np.flatnonzero(mask)[0]]) if np.any(mask) else None
        command = obs[s:e, :3]
        limit = np.array([39.9] * 4 + [4.49] * 2)
        row = dict(episode=number, start_index=int(a[s, 0]), end_exclusive_index=int(a[e, 0]),
            approximate_duration_seconds=float(clock[e] - clock[s]),
            fault_latched_on_exit=bool(col("/wheel_leg/rl/fault_latched")[e]),
            command_min=command.min(axis=0).tolist(), command_max=command.max(axis=0).tolist(),
            height_m=float(obs[s, 3] / 5),
            initial_gravity=obs[s, 7:10].tolist(), initial_pitch_deg=float(pitch[s]),
            initial_roll_deg=float(roll[s]), initial_q_rad=q[s].tolist(),
            initial_hinge_rad=hinge[s].tolist(),
            initial_quiet_gate=float(quiet[s]), initial_recovery_gate=float(recovery[s]),
            leg_velocity_absolute_peak_rad_s=np.max(abs(dq[s:e, :4]), axis=0).tolist(),
            torque_absolute_peak_nm=np.max(abs(tau[s:e]), axis=0).tolist(),
            torque_near_limit_sample_fraction=np.mean(abs(tau[s:e]) >= limit, axis=0).tolist(),
            first_leg_40nm_seconds=first(np.max(abs(tau[s:e, :4]), axis=1) >= 39.9),
            first_roll_pitch_gyro_above_1rad_s_seconds=first(np.linalg.norm(2 * obs[s:e, 4:6], axis=1) > 1),
            first_absolute_pitch_above_20deg_seconds=first(abs(pitch[s:e]) > 20),
            first_absolute_wheel_proxy_above_half_m_s_seconds=first(abs(wheel_proxy[s:e]) > .5),
            pitch_deg=quantile(pitch[s:e]),
            wheel_rolling_proxy_m_s=quantile(wheel_proxy[s:e]),
            position_obs_absolute_peak=np.max(abs(obs[s:e, 14:16]), axis=0).tolist(),
            right_hinge_last_active_rad=float(hinge[e-1, 1]),
            mapped_right_hinge_at_exit_rad=float(hardware_hinge[e, 1]),
            right_hinge_below_rejection_at_exit=bool(hardware_hinge[e, 1] < HINGE_MIN - HINGE_MARGIN),
            telemetry={key: quantile(col("/wheel_leg/rl/" + key)[s:e]) for key in
                ["action_age", "feedback/motor_age_ms", "feedback/imu_age_ms", "performance/inference_us"]})
        result["episodes"].append(row)
        cases.append(dict(name=f"real_start_{number}", source_csv_index=int(a[s, 0]),
            model_q_p_order_rad=q[s].tolist(), model_dq_p_order_rad_s=dq[s].tolist(),
            imu_quaternion_wxyz=a[s, [ix["/wheel_leg/imu/quaternion/" + k] for k in "wxyz"]].tolist(),
            observation=obs[s].tolist(), command=command[0].tolist(),
            root_height_m=None, external_body_pose_measured=False,
            note="Diagnostic seed only; root height/contact and rigid-body pose require reconstruction. IMU was mounted on a loose platform."))
    if args.model_dir:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        ids = np.flatnonzero(active)
        result["onnx_output_signature"] = []
        for name in ["self_wake_V1.0.onnx", "recovery_candidate.onnx"]:
            path = args.model_dir / name
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            predicted = np.concatenate([session.run(None, {"obs": obs[j:j+1].astype("float32")})[0] for j in ids])
            predicted = np.clip(predicted, [-16]*4 + [-9]*2, [16]*4 + [9]*2)
            errors = np.stack([np.max(abs(predicted - raw[ids + lag]), axis=1) for lag in range(-2, 3)], axis=1)
            best = errors.min(axis=1)
            out = dict(name=name, file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rows=len(ids), matches_below_1e_minus_4=int(np.sum(best < 1e-4)),
                minimum_nearby_action_max_absolute_error=quantile(best),
                time_alignment="Best of CSV row offsets -2..+2, accounting for asynchronous inference and logging; not a logged model hash")
            out["unmatched_csv_indices_first_32"] = a[ids[best >= 1e-4][:32], 0].astype(int).tolist()
            result["onnx_output_signature"].append(out)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "real_start_seeds.json").write_text(json.dumps(dict(
        schema="real-start-diagnostic-seeds-v1", source_sha256=source_sha,
        root_height_unidentified=True, cases=cases), indent=2, allow_nan=False) + "\n")
    # These are hardware-log figures, not a separate source of training curves.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s, e = spans[0]
    t = clock[s:e] - clock[s]
    colors = ["#2471a3", "#c0392b", "#239b56", "#8e44ad"]
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True, constrained_layout=True)
    fig.suptitle("Unified115 real robot: first zero-command start\n200 Hz policy; legs initially unpowered, body resting on ground", fontsize=14)
    axes[0].plot(t, pitch[s:e], label="IMU pitch", color="#c0392b")
    axes[0].plot(t, roll[s:e], label="IMU roll", color="#2471a3")
    axes[0].set_ylabel("IMU angle (deg)")
    for j in range(4):
        panel = axes[1 + (j % 2)]
        panel.plot(t, q[s:e, j], color=colors[j], label=NAMES[j].replace("_joint", ""))
        panel.plot(t, targets[s:e, j], "--", color=colors[j], alpha=.7)
        axes[3].plot(t, tau[s:e, j], color=colors[j], lw=.9, label=NAMES[j].replace("_joint", ""))
    axes[1].set_ylabel("Hip angle (rad)\nsolid actual / dashed target")
    axes[2].set_ylabel("Aux angle (rad)\nsolid actual / dashed target")
    axes[3].set_ylabel("Leg API torque (Nm)")
    axes[3].axhline(40, color="black", ls=":", lw=.8)
    axes[3].axhline(-40, color="black", ls=":", lw=.8)
    for j, label, color in [(4, "left", colors[0]), (5, "right", colors[1])]:
        axes[4].plot(t, .06 * dq[s:e, j], color=color, label=label + " wheel rim speed")
        axes[4].plot(t, .6 * raw[s:e, j], "--", color=color, alpha=.6, label=label + " target")
    axes[4].set_ylabel("Wheel rim speed (m/s)\nnot chassis velocity")
    axes[5].plot(t, hinge[s:e, 0], label="left hinge", color=colors[0])
    axes[5].plot(t, hinge[s:e, 1], label="right hinge", color=colors[1])
    axes[5].axhline(HINGE_MIN - HINGE_MARGIN, color="black", ls="--", label="lower rejection bound")
    axes[5].scatter([clock[e]-clock[s]], [hardware_hinge[e, 1]], color="black", zorder=5, label="fault edge, hardware-mapped right hinge")
    axes[5].set_ylabel("Hinge angle (rad)")
    axes[5].set_xlabel("Seconds since RL entry (nearby wheel feedback clock)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.savefig(args.output / "first_zero_command_start.png", dpi=150)
    plt.close(fig)
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == source_sha
    print(json.dumps(dict(source_sha256=source_sha, rows=len(a), episodes=len(spans),
        fault_edges=sum(x["fault_latched_on_exit"] for x in result["episodes"]),
        below_right_hinge_bound_at_fault=sum(x["right_hinge_below_rejection_at_exit"] and x["fault_latched_on_exit"] for x in result["episodes"]),
        output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()

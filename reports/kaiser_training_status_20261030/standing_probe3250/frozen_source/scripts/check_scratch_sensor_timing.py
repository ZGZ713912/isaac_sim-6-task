#!/usr/bin/env python3
"""Validate recorded actor-input timing against the same native trajectory.

This reads existing evidence only. It does not execute a policy, construct an
optimizer, or change the standing acceptance. Gravity kinematics constrain
angular velocity perpendicular to gravity; they cannot independently certify
the yaw component parallel to gravity or any real-robot sensor.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POLICY_FROM_CONTROL = (0, 1, 3, 4, 2, 5)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residual(actual, expected, mask):
    delta = np.asarray(actual, dtype=np.float64)[mask] - np.asarray(expected, dtype=np.float64)[mask]
    if not delta.size or not np.isfinite(delta).all():
        raise ValueError("Need finite, aligned, valid samples")
    return dict(maximum_absolute=float(np.abs(delta).max()), rms=float(np.sqrt(np.mean(delta**2))))


def initial_channels(feedback, nominal):
    names = ("initial_root_pose_local", "initial_gravity", "initial_omega",
             "initial_motor_position_control", "initial_motor_velocity_control", "initial_action_control")
    if not all(name in feedback for name in names):
        return None
    raw = feedback["observation_policy"][0]
    valid = feedback["valid"][0]
    pose = feedback["initial_root_pose_local"]
    if pose.shape != (len(raw), 7) or not np.isfinite(pose).all():
        raise ValueError("Need a finite final reset pose for every fixed case")
    x, y, z, w = pose[:, 3:].astype(np.float64).T
    if np.abs(x*x+y*y+z*z+w*w-1.).max() > 1e-5:
        raise ValueError("Initial root quaternion is not normalized")
    gravity_from_pose = np.stack((2*(w*y-x*z), -2*(w*x+y*z), -1+2*(x*x+y*y)), -1)
    q_delta = feedback["initial_motor_position_control"][:, [0, 1, 3, 4]] - nominal
    q_delta = np.arctan2(np.sin(q_delta), np.cos(q_delta))
    return dict(
        gravity_from_native=residual(raw[:, 7:10], feedback["initial_gravity"], valid),
        gravity_from_root_quaternion=residual(raw[:, 7:10], gravity_from_pose, valid),
        native_gravity_from_root_quaternion=residual(feedback["initial_gravity"], gravity_from_pose, valid),
        gyro_rad_s=residual(2*raw[:, 4:7], feedback["initial_omega"], valid),
        leg_position_delta_rad=residual(raw[:, 10:14], q_delta, valid),
        motor_velocity_rad_s=residual(10*raw[:, 16:22],
            np.take(feedback["initial_motor_velocity_control"], POLICY_FROM_CONTROL, axis=-1), valid),
        previous_applied_action=residual(raw[:, 22:28],
            np.take(feedback["initial_action_control"], POLICY_FROM_CONTROL, axis=-1), valid))


def check(evaluation):
    receipt_file = evaluation / "receipt.json"
    result_file = evaluation / "evaluation/result.json"
    native_file = evaluation / "evaluation/native_trace.npz"
    feedback_file = evaluation / "evaluation/feedback_trace.npz"
    receipt, result = json.loads(receipt_file.read_text()), json.loads(result_file.read_text())
    if receipt["status"] != "completed_unqualified" or receipt["new_PPO"] != 0:
        raise ValueError("Need a completed read-only evaluation")
    checkpoint = Path(receipt["checkpoint"])
    if (sha(native_file) != result["trace_sha256"] or sha(feedback_file) != result["feedback_trace"]["sha256"]
            or sha(checkpoint) != receipt["checkpoint_sha256"]):
        raise ValueError("Evidence checksum mismatch")
    config_file = checkpoint.parent / "config/contract.json"
    config = json.loads(config_file.read_text())
    manifest_file = ROOT / config["asset_directory"] / "manifest.json"
    if sha(manifest_file) != config["asset_manifest_sha256"]:
        raise ValueError("The encoder nominal-angle manifest changed")
    manifest = json.loads(manifest_file.read_text())
    legs = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")
    nominal = np.array([manifest["nominal_joint_pos"][name] for name in legs])
    with np.load(native_file) as native, np.load(feedback_file) as feedback:
        valid = native["valid"]
        raw = feedback["observation_policy"]
        if (valid.ndim != 2 or raw.shape != (*valid.shape, 35)
                or not np.array_equal(valid, feedback["valid"]) or valid.shape[1] != result["case_count"]
                or np.any(np.diff(valid.astype(np.int8), axis=0) > 0)):
            raise ValueError("Unexpected actor shape, alignment or non-prefix validity")
        pair_valid = valid[1:] & valid[:-1]
        g = native["gravity"].astype(np.float64)
        omega = feedback["omega"].astype(np.float64)
        q_delta = native["q"][:-1] - nominal
        q_delta = np.arctan2(np.sin(q_delta), np.cos(q_delta))
        channels = dict(
            gravity=residual(raw[1:, :, 7:10], g[:-1], pair_valid),
            gyro_rad_s=residual(2. * raw[1:, :, 4:7], omega[:-1], pair_valid),
            motor_velocity_rad_s=residual(10. * raw[1:, :, 16:22],
                np.take(feedback["motor_velocity_control"][:-1], POLICY_FROM_CONTROL, axis=-1), pair_valid),
            leg_position_delta_rad=residual(raw[1:, :, 10:14], q_delta, pair_valid),
            previous_applied_action=residual(raw[1:, :, 22:28],
                np.take(feedback["action_applied_control"][:-1], POLICY_FROM_CONTROL, axis=-1), pair_valid))
        expected_commands = np.zeros((*valid.shape, 4))
        expected_commands[:, :, 3] = 5. * np.array([case["height_m"] for case in result["cases"]])
        channels["command_features"] = residual(raw[:, :, :4], expected_commands, valid)
        expected_context = np.zeros((*valid.shape, 7))
        expected_context[:, :, 0] = 1.
        channels["flat_command_context"] = residual(raw[:, :, 28:35], expected_context, valid)
        gdot = np.diff(g, axis=0) / .005
        prediction = -np.cross((omega[1:] + omega[:-1]) * .5, (g[1:] + g[:-1]) * .5)
        error = (gdot - prediction)[pair_valid]
        reversed_error = (gdot + prediction)[pair_valid]
        gyro_relation = dict(
            gravity_derivative_rms_per_s=float(np.sqrt(np.mean(gdot[pair_valid]**2))),
            measured_sign_residual_rms_per_s=float(np.sqrt(np.mean(error**2))),
            reversed_sign_residual_rms_per_s=float(np.sqrt(np.mean(reversed_error**2))),
            measured_sign_residual_norm_p95_per_s=float(np.quantile(np.linalg.norm(error, axis=-1), .95)),
            gravity_norm_maximum_error=float(np.abs(np.linalg.norm(g[valid], axis=-1) - 1.).max()),
            scope="Trapezoidal 200Hz finite-difference check, including impacts; not an exact continuous-time identity or independent yaw-axis certification")
        checks = {name: value["maximum_absolute"] < 1e-5 for name, value in channels.items()}
        initial = initial_channels(feedback, nominal)
        initial_checks = (None if initial is None else
                          {name: value["maximum_absolute"] < 1e-5 for name, value in initial.items()})
    files = [receipt_file, result_file, native_file, feedback_file, config_file, manifest_file, Path(__file__)]
    return dict(schema="scratch-standing-sensor-timing-validation-v1", new_PPO=0,
        native_physics_steps=0, policy_executed=False, optimizer_constructed=False,
        checkpoint_sha256=receipt["checkpoint_sha256"], channels=channels,
        pair_count=int(pair_valid.sum()), checks=checks, all_consecutive_input_alignment_checks_pass=all(checks.values()),
        initial_native_state_recorded=initial is not None, initial_channels=initial, initial_checks=initial_checks,
        all_input_alignment_checks_pass=all(checks.values()) and initial_checks is not None and all(initial_checks.values()),
        gyro_gravity_relation=gyro_relation,
        scope="Consecutive-frame comparisons start at the second actor call. Full alignment additionally requires a recorded native final reset state and matching first actor input. Standing acceptance starts at the first physical millisecond.",
        hardware_ready=False, sources_sha256={str(path.resolve()): sha(path) for path in files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("pair_count", "checks", "initial_native_state_recorded", "initial_checks", "gyro_gravity_relation")}))
    if not report["all_input_alignment_checks_pass"]:
        raise SystemExit("Full input alignment failed or lacks the native reset-boundary recording; see initial_checks")

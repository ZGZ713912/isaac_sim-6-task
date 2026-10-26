#!/usr/bin/env python3
"""Explain recorded native standing failures; does not redefine acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_traces(physical, feedback):
    valid = physical["valid"]
    if valid.ndim != 2 or not np.array_equal(valid, feedback["valid"]):
        raise ValueError("Feedback and physical valid masks disagree")
    steps, lanes = valid.shape
    if not valid[0].all() or np.any(np.diff(valid.astype(int), axis=0) > 0):
        raise ValueError("A fixed standing case must have one uninterrupted valid prefix")
    shapes = dict(observation_policy=(steps, lanes, 35), action_raw_policy=(steps, lanes, 6),
        action_applied_control=(steps, lanes, 6), omega=(steps, lanes, 3),
        motor_velocity_control=(steps, lanes, 6), motor_effort_control=(steps, lanes, 6),
        velocity_world=(steps, lanes, 3), time_s=(steps, lanes), initial_position_local=(lanes, 3))
    for name, shape in shapes.items():
        value = feedback[name]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Bad feedback shape or nonfinite state: " + name)
    for lane in range(lanes):
        time = feedback["time_s"][valid[:, lane], lane]
        if not np.allclose(time, np.arange(1, len(time) + 1) * .005, rtol=0., atol=2e-6):
            raise ValueError("Feedback clock differs from the native 200 Hz fixed-case clock")
    if physical["force"].shape != (steps, 5, lanes, 2, 3):
        raise ValueError("Missing one of the five native physical contact samples")


def explain(physical, feedback, cases):
    validate_traces(physical, feedback)
    if len(cases) != physical["valid"].shape[1]:
        raise ValueError("Case labels and feedback lanes disagree")
    rows = []
    for lane, case in enumerate(cases):
        valid = physical["valid"][:, lane]
        count = int(valid.sum())
        if case["physical_ms"] != count * 5:
            raise ValueError("Original fixed evaluation denominator changed")
        time = feedback["time_s"][valid, lane]
        force = physical["force"][valid, :, lane]
        supported = (np.linalg.norm(force, axis=-1) > 2.).all(-1).reshape(-1)
        missing = np.flatnonzero(~supported)
        gravity = physical["gravity"][valid, lane]
        tilt = np.degrees(np.arccos(np.clip(-gravity[:, 2], -1., 1.)))
        raw = feedback["action_raw_policy"][valid, lane]
        applied = feedback["action_applied_control"][valid, lane]
        wheel_target = 10. * applied[:, [2, 5]]
        wheel_velocity = feedback["motor_velocity_control"][valid, lane][:, [2, 5]]
        omega = feedback["omega"][valid, lane]
        xy = physical["position"][valid, lane, :2]
        row = dict(height_m=case["height_m"], pitch_deg=case["pitch_deg"],
            observed_policy_frames=count, original_standing_pass=case["standing_pass"],
            first_missing_support_s=float((missing[0] + 1) * .001) if len(missing) else None,
            wheel_target_rms_rad_s=float(np.sqrt(np.mean(wheel_target**2))),
            wheel_speed_rms_rad_s=float(np.sqrt(np.mean(wheel_velocity**2))),
            wheel_target_peak_rad_s=float(np.abs(wheel_target).max()),
            wheel_speed_peak_rad_s=float(np.abs(wheel_velocity).max()),
            roll_pitch_gyro_rms_rad_s=float(np.sqrt(np.mean(np.sum(omega[:, :2]**2, axis=-1)))),
            roll_pitch_gyro_peak_rad_s=float(np.abs(omega[:, :2]).max()),
            yaw_rms_rad_s=float(np.sqrt(np.mean(omega[:, 2]**2))),
            body_xy_max_excursion_m=float(np.linalg.norm(xy-feedback["initial_position_local"][lane, :2], axis=-1).max()),
            raw_to_applied_maximum_difference=float(np.abs(raw[:, [0, 1, 4, 2, 3, 5]]-applied).max()),
            first_tilt_crossings={})
        for threshold in (5, 10, 20):
            indices = np.flatnonzero(tilt > threshold)
            if not len(indices):
                row["first_tilt_crossings"][str(threshold)] = None
                continue
            i = int(indices[0])
            row["first_tilt_crossings"][str(threshold)] = dict(time_s=float(time[i]),
                tilt_deg=float(tilt[i]), omega_rad_s=omega[i].tolist(),
                wheel_target_rad_s=wheel_target[i].tolist(), wheel_velocity_rad_s=wheel_velocity[i].tolist(),
                actor_gravity_input=feedback["observation_policy"][valid, lane][i, 7:10].tolist(),
                raw_leg_target_delta_rad=(raw[i, :4]*.25).tolist())
        rows.append(row)
    return dict(schema="scratch-standing-feedback-explanation-v1", cases=rows,
        qualification=False,
        scope="Measured wheel response and failure order; no counterfactual control, acceptance change or causal attribution")


def analyze(directory):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    if (digest(directory / "native_trace.npz") != result["trace_sha256"]
            or digest(directory / "feedback_trace.npz") != result["feedback_trace"]["sha256"]):
        raise ValueError("Recorded evaluation trace SHA changed")
    with np.load(directory / "native_trace.npz") as physical, np.load(directory / "feedback_trace.npz") as feedback:
        report = explain(physical, feedback, result["cases"])
    report.update(evaluation=str(directory.resolve()), result_sha256=digest(directory / "result.json"),
        source_sha256=digest(Path(__file__)), completed_updates=result["completed_updates"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps([r for r in report["cases"] if r["height_m"] == .305], allow_nan=False))

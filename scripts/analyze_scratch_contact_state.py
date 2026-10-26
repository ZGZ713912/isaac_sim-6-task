#!/usr/bin/env python3
"""Classify three recorded contact readouts without changing standing acceptance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(physical, state, cases, metadata):
    valid = np.asarray(physical["valid"])
    if (valid.dtype != np.bool_ or valid.ndim != 2 or not valid[0].all()
            or np.any(np.diff(valid.astype(int), axis=0) > 0)):
        raise ValueError("Expected one uninterrupted valid prefix for each fixed case")
    steps, lanes = valid.shape
    if len(cases) != lanes:
        raise ValueError("Case labels and physical lanes disagree")
    fine_valid = np.repeat(valid, 5, axis=0)
    expected_force_shape = (steps, 5, lanes, 2, 3)
    if (physical["force"].shape != expected_force_shape
            or not np.array_equal(state["valid"], fine_valid)
            or not np.array_equal(state["filtered_wheel_force"], physical["force"].reshape(-1, lanes, 2, 3))):
        raise ValueError("Contact-state force or clock differs from original evaluation")
    joint_names, wheel_names = metadata["joint_names"], metadata["wheel_names"]
    if len(set(joint_names)) != len(joint_names) or len(wheel_names) != 2:
        raise ValueError("Ambiguous joint or wheel labels")
    shapes = dict(joint_position=(steps*5, lanes, len(joint_names)),
        joint_velocity=(steps*5, lanes, len(joint_names)), root_link_pose_local=(steps*5, lanes, 7),
        root_com_velocity_world=(steps*5, lanes, 6), wheel_link_pose_local=(steps*5, lanes, 2, 7),
        wheel_net_force=(steps*5, lanes, 2, 3), wheel_point_force=(steps*5, lanes, 2, 3),
        wheel_contact_point_count=(steps*5, lanes, 2),
        wheel_minimum_contact_separation=(steps*5, lanes, 2),
        contact_buffer_overflow=(steps*5, lanes, 2))
    for name, shape in shapes.items():
        if state[name].shape != shape:
            raise ValueError("Unexpected native contact-state shape: " + name)
        if name != "wheel_minimum_contact_separation" and not np.isfinite(state[name]).all():
            raise ValueError("Nonfinite native contact-state sample: " + name)
    counts = state["wheel_contact_point_count"]
    separation = state["wheel_minimum_contact_separation"]
    if (not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any()
            or state["contact_buffer_overflow"].any()
            or not np.isfinite(separation[counts > 0]).all()
            or not np.isposinf(separation[counts == 0]).all()):
        raise ValueError("Contact points are missing, nonfinite or truncated")
    if not np.isfinite(physical["force"]).all():
        raise ValueError("Nonfinite original contact force")
    rows = []
    for lane, case in enumerate(cases):
        selected = fine_valid[:, lane]
        denominator = int(selected.sum())
        if case["physical_ms"] != denominator:
            raise ValueError("Original physical denominator changed")
        forces = {label: np.linalg.norm(state[name][selected, lane], axis=-1)
                  for label, name in (("filtered", "filtered_wheel_force"),
                                      ("net", "wheel_net_force"), ("points", "wheel_point_force"))}
        missing = forces["filtered"] <= 2.
        dual_missing = missing.any(-1)
        if int(dual_missing.sum()) != case["missing_physical_ms"]:
            raise ValueError("Original strict support outcome changed")
        point_counts = counts[selected, lane]
        net_support, point_support = forces["net"] > 2., forces["points"] > 2.
        categories = dict(
            filter_missing_both_other_readouts_supported=missing & net_support & point_support,
            filter_missing_net_supported_points_below_threshold=missing & net_support & ~point_support,
            filter_missing_net_below_threshold_points_supported=missing & ~net_support & point_support,
            all_three_readouts_below_threshold=missing & ~net_support & ~point_support)
        category_counts = {name: int(mask.sum()) for name, mask in categories.items()}
        if sum(category_counts.values()) != int(missing.sum()):
            raise ValueError("Missing wheel samples are not exhaustively classified")
        events = []
        if case["height_m"] == .305:
            starts = np.flatnonzero(dual_missing & ~np.r_[False, dual_missing[:-1]])
            q = state["joint_position"][selected, lane]
            dq = state["joint_velocity"][selected, lane]
            for start in starts:
                end = int(start + 1)
                while end < denominator and dual_missing[end]:
                    end += 1
                previous = max(0, int(start)-1)
                details = []
                for index in range(max(0, int(start)-2), min(denominator, end+2)):
                    details.append(dict(physical_time_s=(index+1)*.001,
                        filtered_force_n=state["filtered_wheel_force"][index, lane].tolist(),
                        net_force_n=state["wheel_net_force"][index, lane].tolist(),
                        point_normal_force_n=state["wheel_point_force"][index, lane].tolist(),
                        contact_point_count=counts[index, lane].tolist(),
                        joint_position=q[index].tolist(), joint_velocity=dq[index].tolist(),
                        root_link_pose_local=state["root_link_pose_local"][index, lane].tolist(),
                        root_com_velocity_world=state["root_com_velocity_world"][index, lane].tolist(),
                        wheel_link_pose_local=state["wheel_link_pose_local"][index, lane].tolist()))
                events.append(dict(start_physical_ms=int(start+1), duration_ms=end-int(start),
                    joint_position_step_at_onset=(q[start]-q[previous]).tolist() if start else None,
                    onset_joint_velocity=dq[start].tolist(), neighborhood=details))
        rows.append(dict(height_m=case["height_m"], pitch_deg=case["pitch_deg"],
            physical_ms=denominator, original_missing_physical_ms=int(dual_missing.sum()),
            original_standing_pass=case["standing_pass"], missing_wheel_samples=int(missing.sum()),
            missing_wheel_samples_without_any_reported_point=int((missing & (point_counts == 0)).sum()),
            missing_wheel_samples_with_reported_point=int((missing & (point_counts > 0)).sum()),
            missing_wheel_readout_categories=category_counts, mid_height_break_events=events))
    return dict(cases=rows, joint_names=joint_names, wheel_names=wheel_names,
        joint_coordinate_units="Native generalized coordinates: rad for revolute joints, m for prismatic joints; "
            "corresponding velocities are rad/s and m/s. Per-joint types were not captured by this observer.",
        native_joint_properties=metadata.get("native_joint_properties", {}), qualification=False,
        scope="Recorded force-readout classification with original2N threshold and original valid samples. "
            "Point forces contain normal forces; net forces can contain other contact contributions. "
            "All three below threshold does not prove airborne geometry. Position increments and reported "
            "velocities are recorded separately; their difference alone does not establish a solver defect.")


def analyze(evaluation, baseline):
    evaluation, baseline = Path(evaluation), Path(baseline)
    result = json.loads((evaluation / "result.json").read_text())
    reference = json.loads((baseline / "result.json").read_text())
    metadata = result["contact_state_trace"]
    trace_hash = digest(evaluation / "native_trace.npz")
    if (trace_hash != result["trace_sha256"] or trace_hash != reference["trace_sha256"]
            or digest(baseline / "native_trace.npz") != trace_hash
            or result["cases"] != reference["cases"]
            or result["initial_root_positions"] != reference["initial_root_positions"]):
        raise ValueError("Contact observer failed to reproduce the original baseline")
    if (digest(evaluation / "contact_state_trace.npz") != metadata["sha256"]
            or not metadata["original_force_and_valid_samples_exact"] or metadata["buffer_overflow"]
            or metadata["physical_hz"] != 1000
            or digest(evaluation.parent / "diagnostic_sources/capture_scratch_contact_state.py")
               != metadata["source_sha256"]):
        raise ValueError("Contact observer provenance is incomplete or changed")
    with np.load(evaluation / "native_trace.npz") as physical, np.load(evaluation / "contact_state_trace.npz") as state:
        # NpzFile does not cache decoded arrays. Materialize once before the
        # per-event neighborhoods, rather than repeatedly decompressing them.
        report = classify({name: physical[name] for name in physical.files},
            {name: state[name] for name in state.files}, result["cases"], metadata)
    report.update(schema="scratch-contact-state-classification-v1", observed_utc=datetime.now(timezone.utc).isoformat(),
        evaluation=str(evaluation.resolve()), baseline=str(baseline.resolve()), exact_baseline_reproduction=True,
        artifacts_sha256={str(path.resolve()): digest(path) for path in (
            evaluation / "result.json", evaluation / "native_trace.npz", evaluation / "contact_state_trace.npz",
            baseline / "result.json", baseline / "native_trace.npz")}, source_sha256=digest(__file__),
        new_PPO=0, native_steps_by_reader=0, hardware_ready=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous contact-state classification")
    report = analyze(args.evaluation, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps([dict(height_m=row["height_m"], pitch_deg=row["pitch_deg"],
        missing_ms=row["original_missing_physical_ms"], categories=row["missing_wheel_readout_categories"])
        for row in report["cases"] if row["height_m"] == .305]))

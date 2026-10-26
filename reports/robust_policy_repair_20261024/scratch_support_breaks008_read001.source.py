#!/usr/bin/env python3
"""Locate strict support failures in an existing, authenticated native trace.

This is an evidence reader. It does not run a policy or change acceptance.
Times refer to the original one-based 1ms samples, with no startup exclusion.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals(missing):
    if missing.ndim != 1 or missing.dtype != np.bool_ or not len(missing):
        raise ValueError("Need a nonempty Boolean physical-sample sequence")
    edges = np.diff(np.r_[False, missing, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def inspect(directory, height):
    directory = Path(directory).resolve()
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    trace_path = directory / "native_trace.npz"
    if digest(trace_path) != result["trace_sha256"]:
        raise ValueError("The original native trace SHA changed")
    rows = []
    with np.load(trace_path) as trace:
        valid = trace["valid"]
        if (valid.ndim != 2 or valid.dtype != np.bool_ or not valid[0].all()
                or np.any(np.diff(valid.astype(np.int8), axis=0) > 0)
                or valid.shape[1] != len(result["cases"])):
            raise ValueError("Fixed cases need a single uninterrupted valid prefix")
        frames, lanes = valid.shape
        shapes = dict(force=(frames, 5, lanes, 2, 3), clearance=(frames, 5, lanes, 2),
                      dq=(frames, 5, lanes, 4), torque=(frames, 5, lanes, 4))
        if any(trace[name].shape != shape or not np.isfinite(trace[name]).all()
               for name, shape in shapes.items()):
            raise ValueError("Missing or nonfinite original1ms samples")
        for lane, case in enumerate(result["cases"]):
            if case["height_m"] != height:
                continue
            active = valid[:, lane]
            force = trace["force"][active, :, lane].reshape(-1, 2, 3)
            clearance = trace["clearance"][active, :, lane].reshape(-1, 2)
            speed = trace["dq"][active, :, lane].reshape(-1, 4)
            torque = trace["torque"][active, :, lane].reshape(-1, 4)
            supported = (np.linalg.norm(force, axis=-1) > 2.).all(-1)
            missing = ~supported
            breaks = intervals(missing)
            if (len(supported) != case["physical_ms"]
                    or int(missing.sum()) != case["missing_physical_ms"]
                    or float(supported.mean()) != case["dual_support_fraction"]
                    or bool(supported[:20].all()) != case["first20ms_continuous"]
                    or float(np.abs(speed).max()) != case["leg_speed_peak_rad_s"]
                    or sum(end - start for start, end in breaks) != int(missing.sum())):
                raise ValueError("Native samples disagree with original acceptance receipt")
            events = []
            for start, end in breaks:
                before = slice(max(0, start - 10), start)
                neighborhood = slice(max(0, start - 10), min(len(speed), end + 10))
                events.append(dict(
                    first_missing_physical_ms=start + 1,
                    last_missing_physical_ms=end,
                    duration_ms=end - start,
                    containing_policy_frame=start // 5 + 1,
                    substep_in_policy_frame=start % 5 + 1,
                    force_at_onset_n=force[start].tolist(),
                    clearance_at_onset_m=clearance[start].tolist(),
                    leg_velocity_at_onset_rad_s=speed[start].tolist(),
                    leg_speed_peak_previous10ms_rad_s=(float(np.abs(speed[before]).max())
                        if start else None),
                    leg_speed_peak_neighborhood_rad_s=float(np.abs(speed[neighborhood]).max()),
                    leg_effort_peak_neighborhood_nm=float(np.abs(torque[neighborhood]).max())))
            rows.append(dict(case_index=lane, height_m=height, pitch_deg=case["pitch_deg"],
                original_standing_pass=case["standing_pass"],
                physical_ms=len(supported), missing_physical_ms=int(missing.sum()),
                interruption_events=len(events), longest_interruption_ms=max(
                    (event["duration_ms"] for event in events), default=0),
                events=events))
    if len(rows) != 7:
        raise ValueError("Need all seven original pitch cases for the selected height")
    return dict(schema="scratch-native-support-break-evidence-v1", evaluation=str(directory),
        completed_updates=result["completed_updates"], height_m=height,
        total_missing_physical_ms=sum(row["missing_physical_ms"] for row in rows),
        total_interruption_events=sum(row["interruption_events"] for row in rows), cases=rows,
        source_sha256=digest(__file__), result_sha256=digest(result_path),
        native_trace_sha256=digest(trace_path), new_PPO=0, native_steps=0,
        behavior_qualification=False, hardware_ready=False,
        scope="Original strict2N support classification, every1ms from the first step. "
              "Event neighborhoods describe ordering and do not identify a unique cause.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--height", type=float, choices=(.23, .305, .42), default=.305)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".source.py").exists():
        raise ValueError("Preserve previous evidence and its exact reader")
    report = inspect(args.evaluation, args.height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(total_missing_physical_ms=report["total_missing_physical_ms"],
        total_interruption_events=report["total_interruption_events"],
        cases=[{key: row[key] for key in ("pitch_deg", "missing_physical_ms", "interruption_events",
               "longest_interruption_ms")} for row in report["cases"]])))

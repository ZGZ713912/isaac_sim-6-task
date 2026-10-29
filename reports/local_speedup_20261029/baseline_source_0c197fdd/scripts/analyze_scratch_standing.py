#!/usr/bin/env python3
"""Read the native event stream and behavior receipts; never run a policy."""
import argparse
import hashlib
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def analyze(run):
    progress = json.loads((run / "progress.json").read_text())
    evaluation = json.loads((run / "evaluation/result.json").read_text())
    events = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
    available = events.Tags()["scalars"]
    curves = {}
    for tag in ("UnifiedGetup/upright_fraction", "UnifiedGetup/vx_mae_m_s",
                "ScratchStanding/dual_support_physical_fraction", "ScratchStanding/leg_speed_rms_rad_s",
                "ScratchStanding/speed_cost_density_per_second", "ScratchStanding/support_cost_density_per_second"):
        if tag in available:
            values = events.Scalars(tag)
            curves[tag] = dict(count=len(values), first8_mean=sum(x.value for x in values[:8])/len(values[:8]),
                              last8_mean=sum(x.value for x in values[-8:])/len(values[-8:]))
    groups = []
    for height in sorted({row["height_m"] for row in evaluation["cases"]}):
        rows = [row for row in evaluation["cases"] if row["height_m"] == height]
        groups.append(dict(height_m=height, cases=len(rows), full_horizon=sum(row["full_horizon"] for row in rows),
            continuous_support_pass=sum(row["continuous_support_pass"] for row in rows),
            standing_pass=sum(row["standing_pass"] for row in rows),
            first20ms_continuous=sum(row["first20ms_continuous"] for row in rows),
            total_missing_ms=sum(row["missing_physical_ms"] for row in rows),
            worst_dual_support_fraction=min(row["dual_support_fraction"] for row in rows),
            mean_leg_vibration_mrad=sum(row["leg_vibration_3_95hz_rms_mrad"] for row in rows)/len(rows),
            maximum_leg_speed_rad_s=max(row["leg_speed_peak_rad_s"] for row in rows),
            mean_height_mae_m=sum(row["height_mae_m"] for row in rows)/len(rows),
            maximum_tilt_deg=max(row["tilt_peak_deg"] for row in rows),
            mean_forward_speed_rms_m_s=sum(row["forward_speed_rms_m_s"] for row in rows)/len(rows)))
    paths = [run / "progress.json", run / "initialization.json", run / "evaluation/result.json",
             run / "model_initial.pt", run / "model_final.pt"] + list(run.glob("events.out.tfevents.*"))
    return dict(schema="scratch-standing-analysis-v1", run=str(run.resolve()),
        status=progress["status"], completed_updates=progress["completed_updates"],
        checkpoint_reads_during_training=progress["checkpoint_reads"],
        fixed_target_probe=evaluation["fixed_target_probe"], protocol=evaluation["protocol"],
        groups=groups, training_event_summary=curves,
        artifacts_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.exists()},
        hardware_ready=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)

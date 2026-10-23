"""Record completed standing traces to TensorBoard, then plot only those events."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
CASES = ("stand_low", "stand", "stand_high")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=["composed_static_original115", "composed_static_holdout115"])
    args = parser.parse_args()
    run = RUNS / args.run
    report = json.loads((run / "robust/report.json").read_text())
    assert json.loads((run / "progress.json").read_text())["status"] == "completed"
    assert sha(run / "robust/trajectory.npz") == report["trajectory_sha256"]
    protocol = json.loads((REPORTS / "acceptance_protocol004.json").read_text())
    output = REPORTS / ("standing117_" + args.run)
    assert not output.with_suffix(".png").exists() and not output.with_suffix(".json").exists()
    event_path = run / "standing_trace117_events.json"
    assert not event_path.exists()
    # Decimated XY is for slow-drift visualization; exact full200Hz peaks are
    # recorded separately. ActualHF remains the existing200Hz analyzer metric.
    with np.load(run / "robust/trajectory.npz") as data, SummaryWriter(str(run), filename_suffix=".standing_trace117") as writer:
        for case in CASES:
            lanes = [row["lane"] for row in report["lanes"] if row["name"] == case]
            for lane in lanes:
                selected = data["alive"][:, lane] & ~data["passive"][:, lane]
                time = data["time"][selected, lane]
                xy = data["local_position"][selected, lane, :2]
                xy = xy - xy[:1]
                radius = np.linalg.norm(xy, axis=1)
                peak_index = int(radius.argmax())
                prefix = f"StandingTrace/{case}/lane{lane}/"
                writer.add_scalar(prefix + "full200Hz_peak_radius_m", float(radius[peak_index]), int(round(float(time[peak_index]) * 200.)))
                for i in range(0, len(time), 20):
                    step = int(round(float(time[i]) * 200.))
                    writer.add_scalar(prefix + "body_x_m", float(xy[i, 0]), step)
                    writer.add_scalar(prefix + "body_y_m", float(xy[i, 1]), step)
    event_files = {str(p.relative_to(ROOT)): sha(p) for p in run.glob("events.out.tfevents.*.standing_trace117")}
    assert event_files
    event_path.write_text(json.dumps(dict(schema="standing117-event-source-v1", events=event_files,
        report_sha256=sha(run / "robust/report.json"), trajectory_sha256=report["trajectory_sha256"],
        source_scope="TrueXY relative first active frame, 10Hz visual trace and exact200Hz peak; no new physical run or learning"), indent=2) + "\n")
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), layout="constrained")
    source_values = {}
    for case, axis, height in zip(CASES, axes, (230, 305, 420)):
        lanes = [row["lane"] for row in report["lanes"] if row["name"] == case]
        peaks = {lane: acc.Scalars(f"StandingTrace/{case}/lane{lane}/full200Hz_peak_radius_m")[-1] for lane in lanes}
        worst_lane = max(lanes, key=lambda lane: peaks[lane].value)
        source_values[case] = dict(worst_lane=worst_lane, exact200Hz_max_radius_m=peaks[worst_lane].value)
        for lane in lanes:
            prefix = f"StandingTrace/{case}/lane{lane}/"
            xs, ys = acc.Scalars(prefix + "body_x_m"), acc.Scalars(prefix + "body_y_m")
            time = np.array([v.step for v in xs]) / 200.
            x, y = np.array([v.value for v in xs]), np.array([v.value for v in ys])
            highlighted = lane == worst_lane
            axis.plot(time, x * 100., color="#237CB0" if highlighted else "#B1BDC6",
                      alpha=1. if highlighted else .45, lw=1.7 if highlighted else .7,
                      label="X, largest-excursion lane" if highlighted else None)
            if highlighted:
                axis.plot(time, np.sqrt(x*x + y*y) * 100., color="#2B9B78", ls="--", lw=1.3, label="XY radius, same lane")
                axis.scatter([peaks[lane].step / 200.], [peaks[lane].value * 100.], color="#2B9B78", s=22, zorder=4)
        bound = protocol["stand_xy_excursion_max_m_by_case"][case] * 100.
        axis.axhline(bound, color="#B02C35", ls=":", lw=1.2)
        axis.axhline(-bound, color="#B02C35", ls=":", lw=1.2)
        axis.axhline(0., color="#657582", lw=.6)
        axis.set_title(f"{height} mm | exact XY maximum {peaks[worst_lane].value * 100.:.2f} cm")
        axis.set_xlabel("Simulated time (s)")
        extent = max(bound, peaks[worst_lane].value * 100.) * 1.12
        axis.set_ylim(-extent, extent)
        axis.grid(alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Body displacement from initial position (cm)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"30 s standing | seed {report['seed']} | DR 0.35 | 200/1000 Hz\nLow/mid 5 cm; high 10 cm with slow recentering; simulation evidence", fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + extension), dpi=160)
    output.with_suffix(".json").write_text(json.dumps(dict(schema="standing117-event-plot-v1", run=args.run,
        seed=report["seed"], events=event_files, values=source_values, visual_trace_hz=10., exact_peak_hz=200.,
        event_manifest_sha256=sha(event_path), protocol_sha256=sha(REPORTS / "acceptance_protocol004.json"),
        source="TensorBoard scalar readback only", new_macro_updates=0, behavior_qualified=False), indent=2) + "\n")
    print(str(output.with_suffix(".png")), flush=True)


if __name__ == "__main__":
    main()

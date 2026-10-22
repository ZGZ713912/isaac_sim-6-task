#!/usr/bin/env python3
"""Export quiet-standing comparisons directly from TensorBoard events."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    progress = json.loads((args.confirmation / "progress.json").read_text())
    if progress["status"] != "completed":
        raise ValueError("A completed confirmation is required")
    seeds, step = progress["seeds"], progress["completed_updates"]
    events = [EventAccumulator(str(args.confirmation / str(seed))).Reload() for seed in seeds]
    names, heights = ("stand_low", "stand", "stand_high"), (230, 305, 420)
    metrics = [("leg_target_fast_rms_rad_3_95", "Leg target RMS (deg)", 180 / np.pi),
               ("wheel_target_fast_rms_rad_s_3_95", "Wheel target RMS (rad/s)", 1.),
               ("leg_motion_fast_rms_rad_3_95", "Leg motion RMS (deg)", 180 / np.pi),
               ("wheel_motion_fast_rms_rad_s_3_95", "Wheel velocity RMS (rad/s)", 1.)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, (metric, title, scale) in zip(axes.flat, metrics, strict=True):
        for index, (variant, label, color) in enumerate((("parent", "1346 parent", "#8c939d"),
                                                       ("robust", "Candidate", "#168c83"))):
            values = []
            for event in events:
                row = []
                for name in names:
                    sample = event.Scalars(f"Comparison/{variant}/{name}/{metric}")[-1]
                    if sample.step != step:
                        raise ValueError("TensorBoard sample belongs to another checkpoint step")
                    row.append(sample.value * scale)
                values.append(row)
            values = np.asarray(values)
            mean = values.mean(0)
            bars = ax.bar(np.arange(3) + (index - .5) * .35, mean, .35, label=label, color=color,
                          yerr=np.stack((mean - values.min(0), values.max(0) - mean)), capsize=3)
            ax.bar_label(bars, fmt="%.3g", padding=4, fontsize=8)
        ax.set(title=title, xticks=np.arange(3), xticklabels=heights, xlabel="Requested height (mm)")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.margins(y=.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Quiet standing: 3–95 Hz vibration, deterministic policy\n"
                 "Two seeds; bars = mean across seeds, whiskers = seed range; 3–12 s windows, Isaac Sim", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    files = sorted(args.confirmation.glob("*/events.out.tfevents.*"))
    provenance = dict(source="TensorBoard events", checkpoint_sha256=progress["checkpoint_sha256"],
        checkpoint_step=step, seeds=seeds, band_hz=[3, 95], quiet_window_seconds=[3, 12],
        events={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        plot_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), hardware_tested=False)
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

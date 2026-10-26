#!/usr/bin/env python3
"""Static export from verified original TensorBoard event byte prefixes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from compare_scratch_training_events import TAGS


def export(comparison, output):
    report = json.loads(comparison.read_text())
    receipt = output.with_suffix(".receipt.json")
    if output.exists() or receipt.exists():
        raise ValueError("Preserve earlier event exports")
    sources, values = report["sources"], {}
    for name, source in sources.items():
        event = Path(source["path"])
        with event.open("rb") as stream:
            raw = stream.read(source["observed_prefix_bytes"])
        if hashlib.sha256(raw).hexdigest() != source["observed_prefix_sha256"]:
            raise ValueError("Original TensorBoard event prefix changed")
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / event.name
            prefix.write_bytes(raw)
            accumulator = EventAccumulator(str(prefix), size_guidance={"scalars": 0}).Reload()
            values[name] = {tag: {row.step: row.value for row in accumulator.Scalars(tag)} for tag in TAGS}
    # Derive the plotted values from the original events and independently
    # check the existing window receipt. No second curve-data source is used.
    windows = report["windows"]
    derived = {name: {tag: [] for tag in TAGS} for name in sources}
    for window in windows:
        first, last = window["PPO_update_numbers_inclusive"]
        for name in sources:
            for tag in TAGS:
                average = sum(values[name][tag][step] for step in range(first - 1, last)) / (last - first + 1)
                if not math.isfinite(average) or not math.isclose(average, window["runs"][name][tag],
                                                                 rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("Window receipt differs from the original event data")
                derived[name][tag].append(average)
    x = [window["PPO_update_numbers_inclusive"][1] for window in windows]
    metrics = [
        ("ScratchStanding/dual_support_physical_fraction", "Dual-wheel support [%]", 100.),
        ("ScratchStanding/leg_speed_rms_rad_s", "Logged leg speed RMS [rad/s]", 1.),
        ("PrecisionStanding/planar_speed_rms_m_s", "Planar speed RMS [m/s]", 1.),
        ("Train/mean_episode_length", "Logged completed-episode mean [s]", .005),
    ]
    labels = {"pilot005_static_load": "005 reference", "pilot006_contact_events": "006: break cost 0.1",
              "pilot007_contact_cost_zero": "007: break cost 0"}
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=False)
    for axis, (tag, label, scale) in zip(axes.flat, metrics):
        for name in sources:
            axis.plot(x, [scale * value for value in derived[name][tag]], marker="o", markersize=4,
                      linewidth=1.8, label=labels.get(name, name))
        axis.set_ylabel(label)
        axis.set_xlabel("Completed fresh PPO updates")
        axis.set_xticks(x)
        axis.grid(alpha=.22)
        axis.spines[["right", "top"]].set_visible(False)
    axes[0, 0].axhline(100., color="#666666", linewidth=1., linestyle="--")
    axes[0, 0].set_ylim(min(100 * value for row in derived.values()
                          for value in row[metrics[0][0]]) - .15, 100.08)
    axes[1, 1].set_ylim(0., 8.35)
    handles, legend = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend, loc="upper center", ncol=len(sources), bbox_to_anchor=(.5, .95), frameon=False)
    figure.suptitle("Scratch standing PPO: matched 32-update windows", y=.995, fontsize=14)
    figure.text(.5, .025, "Stochastic training with failure resets; window means do not certify uninterrupted standing.",
                ha="center", fontsize=9)
    figure.text(.5, .005, "006 vs 007 is the event-cost control. 005 is an earlier reference with a different reset cache.",
                ha="center", fontsize=8, color="#555555")
    figure.tight_layout(rect=(0., .045, 1., .91))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    receipt.write_text(json.dumps(dict(schema="scratch-event-plot-export-v1",
        comparison=str(comparison.resolve()), comparison_sha256=hashlib.sha256(comparison.read_bytes()).hexdigest(),
        original_event_prefixes=sources, every_plotted_window_verified_from_original_events=True,
        plot=str(output.resolve()), plot_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        plotted_updates_inclusive=[windows[0]["PPO_update_numbers_inclusive"][0], x[-1]],
        native_evaluation=False, new_PPO=0, behavior_qualification=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.comparison, args.output)

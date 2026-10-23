"""Render the decomposition directly from its original TensorBoard events."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent


def main():
    run = ROOT / "runs/unified_robust_20261022/fresh_nominal067_211131"
    source = sorted(run.glob("*quiet081_terms*"))[-1]
    events = EventAccumulator(str(source), size_guidance={"scalars": 0})
    events.Reload()
    cases = ["stand_low", "stand", "stand_high"]
    series = {"Complete output": "whole_wheel20_rms_rad_s",
              "Original feedback + history": "smoothing_term20_rms_rad_s",
              "Quiet neural correction": "neural_correction20_rms_rad_s"}
    figure, axes = plt.subplots(figsize=(8.7, 4.8))
    figure.subplots_adjust(left=.11, right=.98, bottom=.16, top=.91)
    x, values = np.arange(3), {}
    for index, (label, key) in enumerate(series.items()):
        y = [events.Scalars("Diagnosis/QuietHF081/" + case + "/" + key)[-1].value for case in cases]
        values[label] = y
        bars = axes.bar(x + (index - 1) * .23, y, .23, label=label)
        for bar, value in zip(bars, y):
            axes.annotate(f"{value:.2g}", (bar.get_x() + bar.get_width()/2, value),
                          xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    axes.set_yscale("log")
    axes.set_ylim(2e-6, .5)
    axes.set_xticks(x, ["230 mm", "305 mm", "420 mm"])
    axes.set_ylabel("Wheel target RMS (rad/s, log scale)")
    axes.set_title("20–95 Hz contribution on recorded policy inputs")
    axes.legend(loc="upper left", fontsize=8)
    axes.grid(axis="y", alpha=.2)
    axes.set_axisbelow(True)
    figure.text(.5, .025, "Recorded-input decomposition; changed-action balance remains untested.",
                ha="center", fontsize=8)
    output = REPORTS / "quiet_hf081_terms.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    receipt = dict(schema="quiet081-original-event-figure-v1",
        sources={str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()},
        series_rad_s=values, figure_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        new_macro_updates=0, tensorboard_server_started=False)
    (REPORTS / "quiet_hf081_terms_plot.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

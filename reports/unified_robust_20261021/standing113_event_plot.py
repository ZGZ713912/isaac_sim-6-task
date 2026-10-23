"""Export paired standing displacement/HF comparison from TensorBoard only."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"


def main():
    assert json.loads((REPORTS / "iteration110_pipeline.json").read_text())["status"] == "completed"
    names = ["quiet_standing_fresh100", "position_gain_fresh106", "position_gain_tuning110"]
    labels = ["All heights 1.0", "Low/high 1.5; mid 1.0", "Low 1.5 / mid 1.0 / high 2.0"]
    heights = ["stand_low", "stand", "stand_high"]
    fields = ["full_window/worst_episode_body_xy_max_excursion_m", "leg_motion_fast_rms_rad_3_95",
              "wheel_motion_fast_rms_rad_s_3_95"]
    def read(name):
        directory = RUNS / name
        acc = EventAccumulator(str(directory)).Reload()
        values = np.array([[acc.Scalars(f"Comparison/robust/{height}/{field}")[-1].value
                            for height in heights] for field in fields])
        sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in directory.glob("events.out.tfevents.*")}
        assert sources
        return values, sources
    reference, ref_sources = read("quiet_velocity_baseline036")
    values, sources = [], {"quiet_velocity_baseline036": ref_sources}
    for name in names:
        row, event_sources = read(name)
        values.append(row)
        sources[name] = event_sources
    values = np.array(values)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), layout="constrained")
    x, width = np.arange(3), .24
    colors = ["#78899B", "#237CB0", "#359B7D"]
    for metric, axis in enumerate(axes):
        for version, (label, color) in enumerate(zip(labels, colors)):
            y = values[version, metric] * 100. if metric == 0 else values[version, metric] / reference[metric] * 100.
            bars = axis.bar(x + (version - 1) * width, y, width, color=color, label=label)
            if metric == 0:
                axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        axis.axhline(5. if metric == 0 else 100., ls="--", color="#B02C35", lw=1.2,
                    label="5 cm limit" if metric == 0 else "036 reference limit")
        axis.set_xticks(x, ["230 mm", "305 mm", "420 mm"])
        axis.set_ylim(0., (values[:, metric] * 100.).max() * 1.22 if metric == 0 else 125.)
        axis.set_ylabel("Full-window maximum (cm)" if metric == 0 else "RMS / 036 reference (%)")
        axis.set_title(["Body displacement", "Actual leg motion, 3–95 Hz", "Actual wheel motion, 3–95 Hz"][metric])
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Wheel-angle standing feedback: matched 48 × 12 s, seed 211425 / DR 0.35\nSimulation diagnostic; full-window 5 cm and all actual HF gates remain mandatory", fontsize=12)
    for suffix in ("png", "pdf"):
        output = REPORTS / ("standing113." + suffix)
        assert not output.exists()
        fig.savefig(output, dpi=160)
    output = REPORTS / "standing113_sources.json"
    assert not output.exists()
    output.write_text(json.dumps(dict(schema="standing113-event-plot-v1", events=sources,
        source="Existing TensorBoard events only; no server", run_order=names,
        qualified=False, full_training_admitted=False, new_macro_updates=0), indent=2) + "\n")
    print(str(REPORTS / "standing113.png"), flush=True)


if __name__ == "__main__":
    main()

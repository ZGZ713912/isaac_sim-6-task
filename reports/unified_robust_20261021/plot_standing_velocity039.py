"""Export matched static comparisons from the original TensorBoard events only."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/unified_robust_20261021"
RUNS = ROOT / "runs/unified_robust_20261022"


def main():
    final = json.loads((REPORTS / "iteration039_final.json").read_text())
    assert all(final["pairing"].values())
    sources = [("Initial quiet", "quiet_velocity_baseline036", 441),
               ("038: shared exploration", "wheel_velocity_static038", 473),
               ("039: split exploration", "axis_exploration_static039", 473)]
    metrics = [("full_window/worst_episode_body_xy_max_excursion_m", "0-12 s maximum displacement", "cm", 100., 5.),
               ("body_xy_worst_episode_excursion_m", "3-12 s maximum displacement", "cm", 100., 5.),
               ("leg_motion_fast_rms_rad_3_95", "Actual leg motion, 3-95 Hz", "mrad RMS", 1000., None),
               ("wheel_motion_fast_rms_rad_s_3_95", "Actual wheel speed, 3-95 Hz", "rad/s RMS", 1., None)]
    cases = ("stand_low", "stand", "stand_high")
    provenance, values = [], []
    for label, name, step in sources:
        directory = RUNS / name
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        accumulator.Reload()
        row = []
        for metric, _, _, scale, _ in metrics:
            readings = []
            for case in cases:
                tag = f"Comparison/robust/{case}/{metric}"
                candidates = [r.value for r in accumulator.Scalars(tag) if r.step == step]
                if not candidates or max(candidates) - min(candidates) > 1e-7:
                    raise ValueError("Missing or conflicting original TensorBoard readings: " + tag)
                readings.append(candidates[-1] * scale)
            row.append(readings)
        values.append(row)
        provenance.append(dict(label=label, run=str(directory.relative_to(ROOT)), step=step,
            files=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in sorted(directory.glob("events.out.tfevents.*"))]))
    values = np.asarray(values)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=True)
    colors = ("#68788a", "#db9053", "#317f94")
    x = np.arange(3)
    for j, (metric, title, unit, _, limit) in enumerate(metrics):
        ax = axes.flat[j]
        for i, (label, _, _) in enumerate(sources):
            ax.bar(x + (i - 1) * .24, values[i, j], width=.22, label=label, color=colors[i])
        if limit is not None:
            ax.axhline(limit, color="#ac3b3b", linestyle="--", linewidth=1.2, label="5 cm envelope")
        ax.set(title=title, ylabel=unit)
        ax.set_xticks(x, ["Low 230 mm", "Mid 305 mm", "High 420 mm"])
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Matched Isaac Sim static test: 16 episodes per height\n"
                 "200 Hz policy / 1 kHz physics; moderate domains; ground friction 0.3-0.5", fontsize=13)
    output = REPORTS / "standing_velocity039.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    receipt = dict(schema="quiet039-original-event-figure-v1", figure=str(output.relative_to(ROOT)),
        figure_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), sources=provenance,
        metrics=[m[0] for m in metrics], plotted_values=values.tolist(), paired_static_conditions=True,
        data_source="original RSL-RL run TensorBoard event files; no alternate curve source")
    (REPORTS / "standing_velocity039.provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()

"""Export standing feedback results from their original TensorBoard events."""
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
    result = json.loads((REPORTS / "iteration048_final.json").read_text())
    assert all(all(row["paired"].values()) for row in result["tests"].values())
    sources = [("Retained baseline", "quiet_velocity_baseline036"),
               ("Gain -2 (rejected)", "command_feedback_negative048"),
               ("Gain +2", "command_feedback_positive048")]
    metrics = [("full_window/worst_episode_body_xy_max_excursion_m", "Full-window maximum displacement", "cm", 100., 5.),
               ("body_xy_worst_episode_excursion_m", "Steady-window maximum displacement", "cm", 100., 5.),
               ("leg_motion_fast_rms_rad_3_95", "Actual leg motion, 3-95 Hz", "mrad RMS", 1000., None),
               ("wheel_motion_fast_rms_rad_s_3_95", "Actual wheel speed, 3-95 Hz", "rad/s RMS", 1., None)]
    values, provenance = [], []
    for label, name in sources:
        directory = RUNS / name
        events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        events.Reload()
        rows = []
        for metric, _, _, scale, _ in metrics:
            row = []
            for case in ("stand_low", "stand", "stand_high"):
                readings = [r.value for r in events.Scalars(f"Comparison/robust/{case}/{metric}") if r.step == 441]
                assert readings and max(readings) - min(readings) < 1e-7
                row.append(readings[-1] * scale)
            rows.append(row)
        values.append(rows)
        provenance.append(dict(label=label, run=str(directory.relative_to(ROOT)), step=441,
            files=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in sorted(directory.glob("events.out.tfevents.*"))]))
    values = np.asarray(values)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.7), constrained_layout=True)
    x = np.arange(3)
    for j, (_, title, unit, _, limit) in enumerate(metrics):
        ax = axes.flat[j]
        for i, (label, _) in enumerate(sources):
            ax.bar(x + (i - 1) * .24, values[i, j], width=.22, label=label,
                   color=("#68788a", "#ba6657", "#317f94")[i])
        if limit:
            ax.axhline(limit, color="#ac3b3b", linestyle="--", label="5 cm envelope")
        ax.set(title=title, ylabel=unit, yscale="log")
        ax.set_xticks(x, ["Low 230 mm", "Mid 305 mm", "High 420 mm"])
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Matched zero-PPO standing command feedback: 16 episodes per height\n"
                 "200 Hz policy / 1 kHz physics; moderate domains; ground friction 0.3-0.5", fontsize=13)
    path = REPORTS / "standing_command048.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    receipt = dict(schema="quiet048-original-event-figure-v1", figure=str(path.relative_to(ROOT)),
        figure_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), sources=provenance,
        metrics=[m[0] for m in metrics], plotted_values=values.tolist(),
        data_source="Original TensorBoard event files; logarithmic y axes; no separate curve store")
    (REPORTS / "standing_command048.provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()

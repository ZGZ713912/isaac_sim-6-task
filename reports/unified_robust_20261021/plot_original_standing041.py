"""Matched original/retained-quiet comparison from original TensorBoard events."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
receipt = json.loads((REPORTS / "iteration041_final.json").read_text())
assert all(receipt["pairing"].values()) and receipt["optimizer_updates"] == 0
sources = [("Original ONNX", "deployed_static041", "deployed", 473),
           ("Retained quiet policy", "quiet_velocity_baseline036", "robust", 441)]
metrics = [("full_window/worst_episode_body_xy_max_excursion_m", "0-12 s maximum displacement", "cm", 100.),
           ("body_xy_worst_episode_excursion_m", "3-12 s maximum displacement", "cm", 100.),
           ("leg_motion_fast_rms_rad_3_95", "Actual leg motion, 3-95 Hz (log scale)", "mrad RMS", 1000.),
           ("wheel_motion_fast_rms_rad_s_3_95", "Actual wheel speed, 3-95 Hz (log scale)", "rad/s RMS", 1.)]
cases = ("stand_low", "stand", "stand_high")
values, provenance = [], []
for label, name, variant, step in sources:
    directory = RUNS / name
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    events.Reload()
    row = []
    for metric, _, _, scale in metrics:
        readings = []
        for case in cases:
            tag = f"Comparison/{variant}/{case}/{metric}"
            found = [x.value for x in events.Scalars(tag) if x.step == step]
            if not found or max(found) - min(found) > 1e-7:
                raise ValueError("Missing or conflicting event values: " + tag)
            readings.append(found[-1] * scale)
        row.append(readings)
    values.append(row)
    provenance.append(dict(label=label, run=str(directory.relative_to(ROOT)), event_step=step,
        files=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in sorted(directory.glob("events.out.tfevents.*"))]))
values = np.asarray(values)
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=True)
for j, (_, title, unit, _) in enumerate(metrics):
    ax = axes.flat[j]
    for i, (label, _, _, _) in enumerate(sources):
        ax.bar(np.arange(3) + (i - .5) * .32, values[i, j], width=.30,
               label=label, color=("#68788a", "#317f94")[i])
    if j < 2:
        ax.axhline(5., color="#ac3b3b", ls="--", lw=1.2, label="5 cm envelope")
    else:
        ax.set_yscale("log")
    ax.set(title=title, ylabel=unit)
    ax.set_xticks(np.arange(3), ["Low 230 mm", "Mid 305 mm", "High 420 mm"])
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
fig.suptitle("Matched Isaac Sim test: 16 episodes per height; no new PPO\n"
             "200 Hz / 1 kHz; moderate domains; ground friction 0.3-0.5", fontsize=13)
output = REPORTS / "original_standing041.png"
fig.savefig(output, dpi=160)
plt.close(fig)
data = dict(schema="original041-original-event-figure-v1", figure=str(output.relative_to(ROOT)),
    figure_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), sources=provenance,
    metrics=[m[0] for m in metrics], plotted_values=values.tolist(),
    data_source="original TensorBoard events; no separate curve data source",
    scope=receipt["scope"], pairing=receipt["pairing"], new_optimizer_updates=0)
(REPORTS / "original_standing041.provenance.json").write_text(json.dumps(data, indent=2) + "\n")
print(output)

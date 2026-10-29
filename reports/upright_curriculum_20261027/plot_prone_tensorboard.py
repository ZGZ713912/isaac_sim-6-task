"""Render training diagnostics from TensorBoard events; never starts a server."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(__file__).resolve().parents[2]
report = Path(__file__).resolve().parent
names = sys.argv[1:] or ["pilot002_belly_encoder_dr025_u16_e256", "pilot003_prelift42_dense_dr025_u64_e256"]
metrics = [("Prone/knee_angle_mean_deg", "Mean physical knee angle (deg)"),
           ("Prone/support_com_alignment_mean_m", "Wheel support / COM alignment error (m)"),
           ("Prone/xy_excursion_peak_m", "Rollout mean of per-step population max XY (m)"),
           ("ProneReward/stable_standing_per_second", "Stable-standing reward density (1/s)")]
fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
provenance = []
for name in names:
    run = root / "runs/standard_prone_20261027" / name
    events = list(run.glob("events.out.tfevents.*"))
    # Hash the exact event prefix consumed even while training appends to it.
    prefixes = []
    with tempfile.TemporaryDirectory(prefix="prone-event-prefix-") as snapshot:
        for event in events:
            data = event.read_bytes()
            prefixes.append(dict(path=str(event), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            (Path(snapshot) / event.name).write_bytes(data)
        accumulator = EventAccumulator(snapshot, size_guidance={"scalars": 0})
        accumulator.Reload()
    points = {}
    for axis, (tag, title) in zip(axes.flat, metrics):
        values = accumulator.Scalars(tag)
        points[tag] = [dict(step=v.step, value=v.value, wall_time=v.wall_time) for v in values]
        axis.plot([v.step + 1 for v in values], [v.value for v in values], label=name.split("_")[0])
        axis.set(title=title, xlabel="Fresh PPO update")
        axis.grid(alpha=.25)
    provenance.append(dict(run=str(run), event_prefixes=prefixes, points=points))
axes[0, 0].axhspan(40., 43., alpha=.15, color="green", label="Fold guidance band")
axes[0, 0].legend(fontsize=9)
fig.suptitle("Standard prone: separate fresh generations; curves do not qualify get-up", fontsize=13)
fig.savefig(report / "prone_tensorboard_comparison.png", dpi=150)
(report / "prone_tensorboard_comparison.provenance.json").write_text(json.dumps(dict(
    source="RSL-RL TensorBoard event scalars only", training_generations_are_separate=True,
    confounds="Initial bias and dense route rewards differ; no single-factor causality claim",
    aggregation="Each scalar is RSL logger aggregation across the rollout; population max is not a whole-run true peak",
    data=provenance), indent=2) + "\n")
print(report / "prone_tensorboard_comparison.png")

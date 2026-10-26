"""Publish native trace metrics to existing event logs, then plot those events."""
from pathlib import Path
import hashlib
import json
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "runs/robust_policy_repair_20261024"
R = Path(__file__).resolve().parent
evaluations = [
    ("Before correction", B / "high_common0125_nominal211131_001/diagnostic_robust"),
    ("Circular paired targets", B / "diagnosis010_circular_nominal211131/robust"),
]
tags = ["tilt_deg", "height_m", "left_leg_common_rad", "right_leg_common_rad", "vx_m_s", "yaw_rad_s"]
colors = ["#c4692f", "#186f9a"]
receipts = []
for label, path in evaluations:
    report = json.loads((path / "report.json").read_text())
    raw = path / "trajectory.npz"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == report["trajectory_sha256"]
    with np.load(raw) as data:
        lane = next(row["lane"] for row in report["lanes"] if row["name"] == "right_asymmetric")
        assert report["lanes"][lane]["post_settle_family"] == "left"
        obs = data["observations"][:, lane]
        gravity = data["gravity"][:, lane]
        height = data["height"][:, lane]
        velocity = data["velocity"][:, lane, 0]
        yaw = data["omega"][:, lane, 2]
        pair = obs[:, 10:14].reshape(-1, 2, 2)
        difference = pair[:, :, 1] - pair[:, :, 0]
        common = pair[:, :, 0] + .5 * np.arctan2(np.sin(difference), np.cos(difference))
        common = np.arctan2(np.sin(common), np.cos(common))
        values = [np.degrees(np.arccos(np.clip(-gravity[:, 2], -1, 1))), height,
                  common[:, 0], common[:, 1], velocity, yaw]
    with SummaryWriter(str(path / "metrics"), filename_suffix=".circular-native-trace") as writer:
        for tag, series in zip(tags, values):
            for step, value in enumerate(series, 1):
                writer.add_scalar("NativeTrajectory/settled_left/" + tag, float(value), step)
    receipts.append(dict(evaluation=str(path), trajectory_sha256=report["trajectory_sha256"], lane=lane,
                         native_step_seconds=.005, run_status=report.get("native_run_status", "completed")))

fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
units = ["Body tilt (deg)", "Base height (m)", "Left common phase (rad)",
         "Right common phase (rad)", "Forward speed (m/s)", "Yaw rate (rad/s)"]
for color, (label, path) in zip(colors, evaluations):
    event = EventAccumulator(str(path / "metrics"), size_guidance={"scalars": 0})
    event.Reload()
    for ax, tag, unit in zip(axes.flat, tags, units):
        values = event.Scalars("NativeTrajectory/settled_left/" + tag)
        # A repeated plotting invocation does not alter trajectory evidence.
        by_step = {v.step: v.value for v in values}
        steps = np.array(sorted(by_step))
        ax.plot(steps * .005, [by_step[s] for s in steps], label=label, color=color, lw=1.25)
        ax.set(ylabel=unit, xlabel="Native simulation time (s)", xlim=(0, 12))
        ax.grid(alpha=.22)
axes[0, 1].axhline(.305, ls=":", color="#555555", lw=1, label="Height request")
axes[2, 0].axhline(.5, ls=":", color="#555555", lw=1)
axes[2, 1].axhline(1., ls=":", color="#555555", lw=1)
axes[0, 0].legend(loc="upper right", fontsize=9)
fig.suptitle("Same settled-left pose: paired target winding correction, zero PPO\n"
             "200 Hz policy / 1000 Hz native physics; seed 211131", fontsize=14)
fig.supxlabel("Before: full trace retained from cleanup-failed run (diagnostic only). "
               "After: completed 12 s run. This is simulation evidence.", fontsize=9)
output = R / "circular_previous_native001.png"
fig.savefig(output, dpi=170)
plt.close(fig)
(R / "circular_previous_native_plot001.json").write_text(json.dumps(dict(
    schema="native-tensorboard-trace-plot-v1", output=str(output), source="existing evaluation event files",
    inputs=receipts, new_PPO=0, hardware_started=False), indent=2) + "\n")
print(output)

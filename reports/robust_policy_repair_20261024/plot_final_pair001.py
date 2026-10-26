"""Write authenticated native samples to their existing events and plot them."""
from pathlib import Path
import hashlib
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "runs/robust_policy_repair_20261024"
R = Path(__file__).resolve().parent
cases = ["real_start_6__pitch_plus2", "left_symmetric__combined"]
runs = [("Before final-pair correction", B / "high_common0125_eval001/evaluation", "#c4692f"),
        ("After final-pair correction", B / "diagnosis012_final_arithmetic_fixed/evaluation", "#186f9a")]
tags = ["leg_speed_peak_rad_s", "leg_effort_peak_nm", "body_tilt_deg"]
receipts = []
for label, path, color in runs:
    report = json.loads((path / "report.json").read_text())
    raw = path / "trajectory.npz"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == report["trajectory_sha256"]
    data = np.load(raw)
    with SummaryWriter(str(path / "metrics"), filename_suffix=".final-pair-native-trace") as writer:
        for case in cases:
            lane = next(x for x in report["lanes"] if x["name"] == case)
            i = lane["lane"]
            alive = data["alive"][:, i]
            values = [data["leg_speed_peak_rad_s"][:, i], data["leg_effort_peak_nm"][:, i],
                      np.degrees(np.arccos(np.clip(-data["gravity"][:, i, 2], -1, 1)))]
            for tag, series in zip(tags, values):
                for step in np.flatnonzero(alive):
                    writer.add_scalar("NativeTrajectory/" + case + "/" + tag, float(series[step]), int(step + 1))
            receipts.append(dict(evaluation=str(path), trajectory_sha256=report["trajectory_sha256"],
                case=case, lane=i, survived=lane["survived"], recovered=lane["recovered"],
                termination_causes=lane["termination_causes"]))

fig, axes = plt.subplots(3, 2, figsize=(12, 8), constrained_layout=True)
for label, path, color in runs:
    event = EventAccumulator(str(path / "metrics"), size_guidance={"scalars": 0})
    event.Reload()
    for col, case in enumerate(cases):
        for row, tag in enumerate(tags):
            values = event.Scalars("NativeTrajectory/" + case + "/" + tag)
            by_step = {v.step: v.value for v in values}
            steps = np.array(sorted(by_step))
            ax = axes[row, col]
            ax.plot(steps * .005, [by_step[s] for s in steps], color=color, label=label, lw=1.1)
            ax.set(xlabel="Native simulation time (s)", xlim=(0, 8))
            ax.grid(alpha=.2)
for col, title in enumerate(["Floor entry with pitch bias", "Side-fall recovery (combined domain)"]):
    axes[0, col].set_title(title)
for row, unit in enumerate(["Leg speed, 1 ms peak (rad/s)", "Leg effort, 1 ms peak (Nm)", "Body tilt (deg)"]):
    for ax in axes[row]:
        ax.set_ylabel(unit)
axes[0, 0].legend(fontsize=8)
fig.suptitle("Final paired target representation: lower speed spikes, recovery still unqualified\n"
             "Same learned weights / seed241201 / full DR / 200 Hz policy, 1000 Hz physics", fontsize=13)
fig.supxlabel("After-correction side-fall trace ends at an impact guard. Neither policy is hardware-qualified.", fontsize=9)
output = R / "final_pair_native001.png"
fig.savefig(output, dpi=170)
plt.close(fig)
(R / "final_pair_native_plot001.json").write_text(json.dumps(dict(
    schema="native-tensorboard-plot-v1", output=str(output), source="existing native evaluation event files",
    policy_step_seconds=.005, inputs=receipts, new_PPO=0, qualification=None, hardware_started=False), indent=2) + "\n")
print(output)

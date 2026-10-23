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


def main(iteration=48):
    result = json.loads((REPORTS / f"iteration{iteration:03d}_final.json").read_text())
    if iteration == 48:
        assert all(all(row["paired"].values()) for row in result["tests"].values())
    elif iteration == 49:
        assert all(result["baseline_pairing"].values()) and all(result["wheel_proxy_pairing"].values())
    elif iteration == 50:
        assert all(result["pairing"].values())
    elif iteration == 51:
        assert all(result["pairing"]["checks"].values())
    elif iteration == 52:
        assert all(result["pairing_to050"].values()) and all(result["pairing_to_retained"].values())
    elif iteration == 53:
        assert all(result["pairing"].values()) and all(result["pairing_to052"].values())
    elif iteration in (57, 58):
        assert all(result["pairing"].values()) and all(result["zero_abi_pairing"].values())
    elif iteration in (59, 60, 61, 62):
        assert all(result["pairing_to036"].values())
        assert all(result[f"pairing_to{iteration-1:03d}"].values())
    elif iteration == 64:
        assert all(result["pairing"].values()) and all(result["parent_pairing"].values())
    else:
        assert all(result["pairing"].values()) and all(result["pairing_to052"].values()) and all(result["pairing_to053"].values())
    sources = [("Retained baseline", "quiet_velocity_baseline036"),
               ("Gain -2 (rejected)", "command_feedback_negative048"),
               ("Gain +2", "command_feedback_positive048")]
    if iteration == 49:
        sources = [("Retained baseline", "quiet_velocity_baseline036"),
                   ("Wheel proxy, gain +2", "command_feedback_positive048"),
                   ("Rolling estimate, gain +2", "rolling_command_feedback049")]
    if iteration == 50:
        sources = [("Retained baseline", "quiet_velocity_baseline036"),
                   ("Learned rolling feedback, 64 updates", "learned_rolling_static050")]
    if iteration == 51:
        sources = [("050 with original delays", "learned_rolling_static050"),
                   ("050, diagnostic delay0", "instant_policy_delay051")]
    if iteration == 52:
        sources = [("Retained baseline", "quiet_velocity_baseline036"),
                   ("Learned rolling feedback (050)", "learned_rolling_static050"),
                   ("All-height q prediction (052)", "all_prediction_static052")]
    if iteration == 53:
        sources = [("Retained baseline", "quiet_velocity_baseline036"),
                   ("All-height prediction (052)", "all_prediction_static052"),
                   ("Adapted feedback, 32 updates (053)", "adapted_prediction_static053")]
    if iteration == 54:
        sources = [("Retained baseline", "quiet_velocity_baseline036"),
                   ("Original wheel exploration (053)", "adapted_prediction_static053"),
                   ("Lower wheel exploration (054)", "precision_sampling_static054")]
    if iteration in (57, 58):
        sources = [("Retained baseline", "quiet_velocity_baseline036"),
                   (f"Learned position feedback, 32 updates ({iteration:03d})", f"learned_position_static{iteration:03d}")]
    if iteration == 59:
        sources = [("Retained baseline", "quiet_velocity_baseline036"),
                   ("Learned position feedback (058)", "learned_position_static058"),
                   ("Mid-to-low transfer (059)", "low_transfer_static059")]
    if iteration == 60:
        sources = [("Retained baseline", "quiet_velocity_baseline036"),
                   ("Low transfer (059)", "low_transfer_static059"),
                   ("Mid-to-high transfer (060)", "high_transfer_static060")]
    if iteration == 61:
        sources = [("Retained baseline", "quiet_velocity_baseline036"),
                   ("Position filter 50 ms (060)", "high_transfer_static060"),
                   ("Position filter 150 ms (061)", "position_filter_static061")]
    if iteration == 62:
        sources = [("Retained baseline", "quiet_velocity_baseline036"),
                   ("Capture after 200 ms (061)", "position_filter_static061"),
                   ("Capture after 50 ms (062)", "position_reference_static062")]
    if iteration == 64:
        sources = [("Retained baseline", "quiet_velocity_baseline036"),
                   ("Position feedback parent (062)", "position_reference_static062"),
                   ("Joint quiet adaptation, 32 updates (064)", "joint_quiet_static064")]
    metrics = [("full_window/worst_episode_body_xy_max_excursion_m", "Full-window maximum displacement", "cm", 100., 5.),
               ("body_xy_worst_episode_excursion_m", "Steady-window maximum displacement", "cm", 100., 5.),
               ("leg_motion_fast_rms_rad_3_95", "Actual leg motion, 3-95 Hz", "mrad RMS", 1000., None),
               ("wheel_motion_fast_rms_rad_s_3_95", "Actual wheel speed, 3-95 Hz", "rad/s RMS", 1., None)]
    values, provenance = [], []
    for label, name in sources:
        directory = RUNS / name
        step = json.loads((directory / "progress.json").read_text())["completed_updates"]
        events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        events.Reload()
        rows = []
        for metric, _, _, scale, _ in metrics:
            row = []
            for case in ("stand_low", "stand", "stand_high"):
                readings = [r.value for r in events.Scalars(f"Comparison/robust/{case}/{metric}") if r.step == step]
                assert readings and max(readings) - min(readings) < 1e-7
                row.append(readings[-1] * scale)
            rows.append(row)
        values.append(rows)
        provenance.append(dict(label=label, run=str(directory.relative_to(ROOT)), step=step,
            files=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in sorted(directory.glob("events.out.tfevents.*"))]))
    values = np.asarray(values)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.7), constrained_layout=True)
    x = np.arange(3)
    for j, (_, title, unit, _, limit) in enumerate(metrics):
        ax = axes.flat[j]
        for i, (label, _) in enumerate(sources):
            ax.bar(x + (i - (len(sources) - 1) / 2) * .24, values[i, j], width=.22, label=label,
                   color=("#68788a", "#ba6657", "#317f94")[i])
        if limit:
            ax.axhline(limit, color="#ac3b3b", linestyle="--", label="5 cm envelope")
        ax.set(title=title, ylabel=unit, yscale="log")
        ax.set_xticks(x, ["Low 230 mm", "Mid 305 mm", "High 420 mm"])
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    title = "Matched learned rolling feedback" if iteration == 50 else "Matched zero-PPO standing command feedback"
    if iteration == 51:
        title = "Observation-delay counterfactual (051), no PPO"
    if iteration == 52:
        title = "Matched all-height leg prediction (052), no PPO; original delays retained"
    if iteration == 53:
        title = "Adapted rolling feedback (053); original randomized delays retained"
    if iteration == 54:
        title = "Equal-budget quiet wheel exploration (053 vs054); sensor domains retained"
    if iteration in (57, 58):
        title = f"Learned encoder-relative position feedback ({iteration:03d}); sensor domains retained"
    if iteration in (59, 60):
        title = f"Transferred encoder-relative position feedback ({iteration:03d}), no PPO"
    if iteration == 61:
        title = "Slower encoder-relative position observation (061), no PPO"
    if iteration == 62:
        title = "Earlier quiet position-reference capture (062), no PPO"
    if iteration == 64:
        title = "Low-LR joint quiet adaptation (064); sensor domains retained"
    fig.suptitle(title + ": 16 episodes per height\n"
                 "200 Hz policy / 1 kHz physics; moderate domains; ground friction 0.3-0.5", fontsize=13)
    basename = f"standing_rolling{iteration:03d}" if iteration == 50 else f"standing_command{iteration:03d}"
    if iteration in (51, 52):
        basename = f"standing_{'delay' if iteration == 51 else 'prediction'}{iteration:03d}"
    if iteration == 53:
        basename = "standing_adaptation053"
    if iteration == 54:
        basename = "standing_sampling054"
    if iteration in (57, 58, 59, 60, 61, 62, 64):
        basename = f"standing_position{iteration:03d}"
    path = REPORTS / (basename + ".png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    receipt = dict(schema=f"quiet{iteration:03d}-original-event-figure-v1", figure=str(path.relative_to(ROOT)),
        figure_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), sources=provenance,
        metrics=[m[0] for m in metrics], plotted_values=values.tolist(),
        data_source="Original TensorBoard event files; logarithmic y axes; no separate curve store")
    (REPORTS / (basename + ".provenance.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, choices=[48, 49, 50, 51, 52, 53, 54, 57, 58, 59, 60, 61, 62, 64], default=48)
    main(parser.parse_args().iteration)

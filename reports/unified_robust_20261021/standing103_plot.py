"""Static summary exported only from the existing TensorBoard events."""
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


def main():
    runs = ["constant_wheel_random087", "high_bias_static093", "high_bias_static097", "high_bias_static095"]
    bias = np.array([0., .002, .003, .004])
    heights = ["stand_low", "stand", "stand_high"]
    labels = ["230 mm", "305 mm", "420 mm"]
    colors = ["#2477B3", "#DB7C18", "#37955A"]
    fields = ["full_window/worst_episode_body_xy_max_excursion_m", "leg_motion_fast_rms_rad_3_95", "wheel_motion_fast_rms_rad_s_3_95"]
    factors = [100., 1000., 1.]
    titles = ["Full-window displacement", "Actual leg motion, 3–95 Hz", "Actual wheel motion, 3–95 Hz"]
    ylabels = ["Maximum displacement (cm)", "RMS angle (mrad)", "RMS angular speed (rad/s)"]
    limits = [[5.] * 3, [.2827848, .6616216, 1.2038746], [.0577383, .1259506, .2259019]]
    values = []
    source = {}
    for name in runs:
        directory = RUNS / name
        accumulator = EventAccumulator(str(directory)).Reload()
        values.append([[accumulator.Scalars(f"Comparison/robust/{height}/{field}")[-1].value
                        for height in heights] for field in fields])
        source[name] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in directory.glob("events.out.tfevents.*")}
    values = np.array(values)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), layout="constrained")
    for metric, axis in enumerate(axes):
        for height, (label, color) in enumerate(zip(labels, colors)):
            axis.plot(bias * 1000., values[:, metric, height] * factors[metric], "o-", color=color, label=label, lw=1.7)
            if metric:
                axis.axhline(limits[metric][height], color=color, ls="--", alpha=.55, lw=1.)
        if not metric:
            axis.axhline(5., color="#B32E35", ls="--", label="5 cm limit")
        axis.set(title=titles[metric], xlabel="High quiet common bias (×10⁻³)", ylabel=ylabels[metric])
        axis.set_xticks(bias * 1000.)
        axis.grid(alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8, loc="best")
    fig.suptitle("Paired random standing: 48 × 12 s per candidate\nOnly 0.003 passes this batch; longer/fresh confirmation remains pending", fontsize=12)
    fig.savefig(REPORTS / "standing103.png", dpi=160)
    fig.savefig(REPORTS / "standing103.pdf")
    (REPORTS / "standing103_sources.json").write_text(json.dumps(dict(
        schema="standing103-event-plot-v1", source="Existing TensorBoard events only; no server",
        events=source, candidate_order=runs, source_bias_steps=bias.tolist(),
        qualified=False, new_macro_updates=0), indent=2) + "\n")
    print(str(REPORTS / "standing103.png"), flush=True)


if __name__ == "__main__":
    main()

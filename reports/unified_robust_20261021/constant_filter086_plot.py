"""Export the086 comparison from original TensorBoard scalar events only."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUN = ROOT / "runs/unified_robust_20261022/constant_wheel_static086"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from evaluate_robust_getup_batch import quiet_filter_variants


def main():
    assert json.loads((REPORTS / "iteration086_pipeline.json").read_text())["status"] == "completed"
    events = EventAccumulator(str(RUN), size_guidance={"scalars": 0}).Reload()
    variants = quiet_filter_variants([1., .4, .32, .25])
    panels = [("full_window/worst_episode_body_xy_max_excursion_m", 100., "Full-window drift (cm)"),
              ("wheel_target_hf_rms_rad_s", 1., "Wheel command 20-95 Hz (rad/s)"),
              ("leg_motion_fast_rms_rad_3_95", 1000., "Leg motion 3-95 Hz (mrad)")]
    fig, axes = plt.subplots(1, 3, figsize=(11., 3.8))
    tags = []
    for axis, (metric, scale, title) in zip(axes, panels, strict=True):
        for case, label, color in (("stand_low", "23 cm", "#0072B2"), ("stand", "30.5 cm", "#D55E00"),
                                  ("stand_high", "42 cm", "#009E73")):
            values = []
            for variant in variants:
                tag = f"Comparison/{variant}/{case}/{metric}"
                entry = events.Scalars(tag)[-1]
                assert entry.step == 505
                values.append(entry.value * scale)
                tags.append(tag)
            axis.plot(range(4), values, "o-", color=color, label=label, linewidth=1.5, markersize=5)
        axis.set_xticks(range(4), ["1", "0.4", "0.32", "0.25"])
        axis.set_xlabel("Complete wheel output alpha")
        axis.set_title(title, fontsize=11)
        axis.grid(alpha=.2)
        axis.set_ylim(bottom=0.)
    axes[0].axhline(5., color="#444444", linestyle="--", linewidth=1., label="5 cm limit")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Constant filtering: fresh same-batch nominal control at 200 Hz", fontsize=13)
    fig.text(.5, .02, "One nominal condition; repeated lanes do not add randomized coverage.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0., .05, 1., .92])
    output = REPORTS / "constant_filter086.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = dict(schema="quiet086-original-event-plot-v1", plot=str(output.relative_to(ROOT)),
        plot_sha256=sha(output), tags=tags, step=505,
        event_sources={str(p.relative_to(ROOT)): sha(p) for p in sorted(RUN.glob("events.out.tfevents.*"))})
    (REPORTS / "constant_filter086_plot.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

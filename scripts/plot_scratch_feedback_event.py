#!/usr/bin/env python3
"""Export a native evaluation plot from its RSL-RL TensorBoard event only."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.evaluation_run / "receipt.json").read_text())
    if receipt["status"] != "completed_unqualified":
        raise ValueError("Need a completed read-only evaluation; this plot grants no qualification")
    result = receipt["evaluation"]
    window = result["tensorboard_native_mid_trace"]
    event = EventAccumulator(str(args.evaluation_run), size_guidance={"scalars": 0})
    event.Reload()

    def channel(rate, name):
        rows = event.Scalars(f"EvalScratchTrace{rate}Hz/mid_zero_pitch/{name}")
        expected = window["physical_frames" if rate == 1000 else "policy_frames"]
        steps = np.array([r.step for r in rows])
        if not np.array_equal(steps, np.arange(1, expected+1)):
            raise ValueError("TensorBoard native trace has duplicate or missing physical samples: " + name)
        return steps / rate, np.array([r.value for r in rows])

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    t, tilt = channel(200, "tilt_deg")
    axes[0].plot(t, tilt, color="tab:purple")
    axes[0].axhline(10., color="gray", linestyle=":", label="8 s screen limit")
    axes[0].set_ylabel("Body tilt (deg)")
    axes[0].legend(loc="upper left")
    t, force = channel(1000, "minimum_wheel_force_n")
    _, support = channel(1000, "dual_support")
    axes[1].plot(t, force, color="tab:blue", linewidth=.8)
    axes[1].axhline(2., color="gray", linestyle=":")
    axes[1].scatter(t[support < .5], force[support < .5], color="tab:red", s=6,
                    label=f"Lost dual support: {int((support < .5).sum())} ms")
    axes[1].set_ylabel("Min wheel force (N)")
    axes[1].legend(loc="upper left")
    t, speed = channel(1000, "leg_speed_peak_rad_s")
    axes[2].plot(t, speed, color="tab:green", linewidth=.8)
    axes[2].axhline(3., color="gray", linestyle=":")
    axes[2].set_ylabel("Peak |leg speed| (rad/s)")
    for side, color in (("left", "tab:blue"), ("right", "tab:orange")):
        for kind, style in (("target", "--"), ("velocity", "-")):
            t, value = channel(200, f"{side}_wheel_{kind}_rad_s")
            axes[3].plot(t, value, color=color, linestyle=style, linewidth=.8, label=f"{side} {kind}")
    axes[3].set_ylabel("Wheel speed (rad/s)")
    axes[3].set_xlabel("Native time from grounded reset (s); no startup grace")
    axes[3].legend(loc="upper left", ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    label = Path(receipt["checkpoint"]).parent.name
    fig.suptitle(f"{label}: from-zero update {result['completed_updates']}, 0.305 m / 0 deg\n"
                 "Nominal deterministic evaluation; unqualified")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = dict(schema="scratch-standing-event-figure-v1", image=str(args.output.resolve()),
        image_sha256=sha(args.output), checkpoint_sha256=receipt["checkpoint_sha256"],
        evaluator_receipt_sha256=sha(args.evaluation_run / "receipt.json"),
        tensorboard_event_sha256={str(p): sha(p) for p in args.evaluation_run.glob("events.out.tfevents.*")},
        trace_window=window, graph_source="RSL-RL TensorBoard events; no NPZ used by this plot",
        qualified=False, hardware_ready=False)
    args.output.with_suffix(".receipt.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

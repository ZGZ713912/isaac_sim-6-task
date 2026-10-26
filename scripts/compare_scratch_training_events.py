#!/usr/bin/env python3
"""Compare complete from-zero PPO windows from immutable event byte prefixes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = (
    "Train/mean_episode_length", "ScratchStanding/dual_support_physical_fraction",
    "ScratchStanding/leg_speed_rms_rad_s", "PrecisionStanding/height_mae_m",
    "PrecisionStanding/tilt_mean_deg", "PrecisionStanding/planar_speed_rms_m_s",
    "StandingFailure/episode_events",
)


def compare(runs, window):
    if window <= 0 or len({run.name for run in runs}) != len(runs):
        raise ValueError("Need positive window length and distinct run names")
    sources, data = {}, {}
    for run in runs:
        event, = list(run.glob("events.out.tfevents.*"))
        raw = event.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / event.name
            copy.write_bytes(raw)
            accumulator = EventAccumulator(str(copy), size_guidance={"scalars": 0}).Reload()
            data[run.name] = {}
            for tag in TAGS:
                rows = accumulator.Scalars(tag)
                values = {row.step: row.value for row in rows}
                if len(values) != len(rows) or not all(math.isfinite(v) for v in values.values()):
                    raise ValueError("Duplicate steps or nonfinite event values")
                data[run.name][tag] = values
        sources[run.name] = dict(path=str(event.resolve()), observed_prefix_bytes=len(raw),
            observed_prefix_sha256=hashlib.sha256(raw).hexdigest())
    common = 0
    while all(common in data[run.name][tag] for run in runs for tag in TAGS):
        common += 1
    windows = []
    for start in range(0, common // window * window, window):
        values = {run.name: {tag: sum(data[run.name][tag][i] for i in range(start, start+window))/window
                            for tag in TAGS} for run in runs}
        for stats in values.values():
            stats["logged_episode_mean_seconds"] = stats["Train/mean_episode_length"]*.005
        windows.append(dict(PPO_update_numbers_inclusive=[start+1, start+window], runs=values))
    if not windows:
        raise ValueError("No complete common window; keep observing the same live training handle")
    return dict(schema="scratch-training-event-window-comparison-v1",
        observed_utc=datetime.now(timezone.utc).isoformat(), new_PPO=0, sources=sources,
        common_complete_updates=common, windows=windows, hardware_ready=False,
        scope="Stochastic training with failure autoresets. Native episode length is a rolling mean of completed "
              "episodes, not a deterministic standing pass. No new curve data source or TensorBoard server.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve older event-prefix observations")
    report = compare(args.runs, args.window)
    report["reader_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(common_complete_updates=report["common_complete_updates"], windows=report["windows"])))

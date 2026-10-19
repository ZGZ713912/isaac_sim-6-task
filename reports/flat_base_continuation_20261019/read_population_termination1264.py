"""Read unchanged native termination flags from all sixteen completed updates."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / "runs/flat_base_20261019/pilot024"
OUT = HERE / "population_live_termination1264.json"
if OUT.exists():
    raise SystemExit("Preserve the completed termination readout")
progress = json.loads((RUN / "progress.json").read_text())
if progress["status"] != "block_complete" or progress["successful_updates_total"] != 1264:
    raise SystemExit("Wait for all sixteen completed updates")
events = EventAccumulator(str(RUN), size_guidance={"scalars": 0})
events.Reload()
tags = set(events.Tags()["scalars"])
steps = set(range(1249, 1265))
views = []
unique_count = 0
for tag in sorted(tags):
    if not tag.startswith(("FlatBirth/", "FlatBehavior/")) or not tag.endswith("/terminated_frame_fraction"):
        continue
    prefix = tag.removesuffix("/terminated_frame_fraction")
    frames = {p.step: p.value for p in events.Scalars(prefix + "/sample_frames")}
    counts = {}
    for p in events.Scalars(tag):
        if p.step not in steps:
            continue
        count = frames[p.step] * p.value
        if not math.isclose(count, round(count), abs_tol=1e-5):
            raise ValueError("Noninteger native flag count: " + tag)
        if round(count):
            counts[str(p.step)] = round(count)
    if counts:
        views.append(dict(view=prefix, terminated_events_by_completed_update=counts))
        if prefix.count("/") == 1:
            unique_count += sum(counts.values())
failure = {p.step + 1: p.value for p in events.Scalars("/reward/failure_event")}
if set(failure) != steps:
    raise ValueError("Missing native failure-event points")
reward_count = -sum(failure.values()) * 4096 * 96 / 200
matched = math.isclose(unique_count, reward_count, abs_tol=1e-5)
if not matched:
    raise ValueError("Native termination flags and reward accounting disagree")
result = dict(schema="flat-live-termination-count-readout-v1",
    recorded_at=datetime.now(timezone.utc).isoformat(), completed_update=1264,
    completed_update_steps=sorted(steps), unique_group_terminated_flagged_frames=unique_count,
    native_failure_reward_values_by_completed_update=failure,
    reward200_equivalent_events=reward_count, matching_counts=matched, views=views,
    event_file_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in RUN.glob("events.out.tfevents.*")},
    scope="Disjoint first1s and after1s windows and unsplit groups count unique pre-auto-reset terminated flags. Mode and height views overlap. Pure boundary truncation is excluded by terminated. The old logger has no reason flags, so failure types and co-occurring boundary causes cannot be assigned.",
    new_native_steps=0, new_optimizer_updates=0, behavior_qualified=False)
OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(unique_termination_frames=unique_count, reward_equivalent=reward_count,
                      views=views), indent=2))

#!/usr/bin/env python3
"""Compare matched PPO update windows, using TensorBoard events as the source."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics


def summarize(values):
    if not values:
        return None
    return dict(samples=len(values),mean=statistics.mean(values),median=statistics.median(values),
        minimum=min(values),maximum=max(values),population_std=statistics.pstdev(values))


def read_run(root):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    root = root.resolve()
    path = root/"receipt.json"
    receipt = json.loads(path.read_text())
    events = EventAccumulator(str(root),size_guidance={"scalars":0})
    events.Reload()
    tags = ("Perf/collection time","Perf/learning_time","Perf/total_fps")
    scalars = {tag:{entry.step:entry for entry in events.Scalars(tag)} for tag in tags}
    return receipt,scalars,dict(receipt=str(path),receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status=receipt["status"],completed_updates=receipt["completed_updates"],
        physics_hz=receipt["arguments"]["physics_hz"],
        constraint_gap_limit_m=receipt.get("constraint_gap_limit_m",.003),
        max_constraint_gap_m=receipt.get("max_constraint_gap_m"),
        event_files=[str(p) for p in sorted(root.glob("events.out.tfevents.*"))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--first-update",type=int,default=6)
    parser.add_argument("--last-update",type=int,default=30)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.first_update<1 or args.last_update<args.first_update:
        parser.error("Use an inclusive, positive PPO parameter-update window")
    first,a,baseline = read_run(args.baseline)
    second,b,candidate = read_run(args.candidate)
    fields = ("task","num_envs","seed","randomization_profile","randomization_strength",
        "solver_position_iterations","solver_velocity_iterations","external_forces_every_iteration")
    for field in fields:
        if first["arguments"][field]!=second["arguments"][field]:
            raise ValueError("Timing comparison changed another setting: "+field)
    for field in ("source_snapshot_sha256","asset_manifest_sha256","reset_lookup_sha256",
            "collision_support_sha256","ground_plane_sha256"):
        if first[field]!=second[field]:
            raise ValueError("Timing comparison changed model/source data: "+field)
    common_source = first["implementation_files_sha256"].keys()&second["implementation_files_sha256"].keys()
    changed = [name for name in common_source if name.startswith("src/") and
        first["implementation_files_sha256"][name]!=second["implementation_files_sha256"][name]]
    if changed:
        raise ValueError("Timing comparison changed the model/control implementation: "+str(changed))
    steps = sorted(set(range(args.first_update-1,args.last_update))&
        set.intersection(*(set(values) for values in [*a.values(),*b.values()])))
    def window(scalars):
        collection = scalars["Perf/collection time"]
        learning = scalars["Perf/learning_time"]
        fps = scalars["Perf/total_fps"]
        return dict(collection_time_s=summarize([collection[s].value for s in steps]),
            learning_time_s=summarize([learning[s].value for s in steps]),
            iteration_time_s=summarize([collection[s].value+learning[s].value for s in steps]),
            event_wall_interval_s=summarize([fps[s].wall_time-fps[s-1].wall_time for s in steps if s-1 in fps]),
            policy_samples_per_s=summarize([fps[s].value for s in steps]))
    baseline["window"] = window(a)
    candidate["window"] = window(b)
    result = dict(schema="wheelbipe-event-timing-comparison-v1",observed_at=datetime.now(timezone.utc).isoformat(),
        units="seconds per PPO parameter update; 24 policy steps per environment",policy_hz=50,
        matched_settings={k:first["arguments"][k] for k in fields},
        requested_update_window_inclusive=[args.first_update,args.last_update],
        actual_update_window_inclusive=[steps[0]+1,steps[-1]+1] if steps else None,
        compared_updates=len(steps),complete_requested_window=len(steps)==args.last_update-args.first_update+1,
        baseline=baseline,candidate=candidate,
        notes=["Setup time excluded; first five updates excluded by the default window",
            "Iteration time is the original RSL collection_time + learn_time",
            "Event wall intervals also include intervening logging/checkpoint overhead",
            "Same seed/configuration does not imply identical trajectories across integration clocks",
            "Timing is independent of behavior acceptance; failed runs stay failed"])
    if steps:
        old = baseline["window"]["iteration_time_s"]["mean"]
        new = candidate["window"]["iteration_time_s"]["mean"]
        result["mean_iteration_speedup"] = old/new
        result["mean_iteration_time_reduction_fraction"] = 1.-new/old
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("baseline","candidate","notes")},indent=2))


if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Review native inverted trials, separating birth pose from settled RL entry."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from train_robust_getup import digest, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    progress = json.loads((args.run / "progress.json").read_text())
    if progress["status"] != "completed":
        raise ValueError("Require a completed native evaluation")
    report = json.loads((args.run / "evaluation/report.json").read_text())
    summary = json.loads((args.run / "evaluation/gap_summary.json").read_text())
    trace_path = args.run / "evaluation/trajectory.npz"
    if digest(trace_path) != report["trajectory_sha256"] or digest(trace_path) != summary["trajectory_sha256"]:
        raise ValueError("Native trace hash differs")
    checkpoint = Path(progress["checkpoint"])
    if digest(checkpoint) != progress["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs")
    step = torch.load(checkpoint, map_location="cpu", weights_only=False)["infos"]["completed_updates"]
    trace = np.load(trace_path)
    rows = []
    writer = SummaryWriter(str(args.run / "evaluation/metrics"), filename_suffix=".inverted-review")
    for lane, measured in zip(report["lanes"], summary["rows"], strict=True):
        if not lane["behavior_case"].startswith("rmcs_") or lane["name"] != measured["name"]:
            raise ValueError("Not the declared RMCS inverted population")
        i = lane["lane"]
        controlled = np.flatnonzero(trace["alive"][:, i] & ~trace["passive"][:, i])
        first = int(controlled[0]) if len(controlled) else None
        row = dict(name=lane["name"], pose_index=lane["pose_index"],
                   post_settle_family=lane["post_settle_family"], survived=lane["survived"],
                   recovered=lane["recovered"], followup_success=lane["followup_success"],
                   termination_causes=lane["termination_causes"],
                   termination_time_s=lane["termination_time_s"],
                   gravity_z_at_first_RL=(float(trace["gravity"][first, i, 2]) if first is not None else None),
                   height_at_first_RL=(float(trace["height"][first, i]) if first is not None else None),
                   first_RL_time_s=(float(trace["time"][first, i]) if first is not None else None),
                   tail_stable_fraction=measured.get("tail_stable_fraction", 0.),
                   max_leg_speed_rad_s=measured.get("max_leg_speed_rad_s"),
                   max_gap_m=lane["max_gap_m"], max_contact_impulse_ns=lane["max_contact_impulse_ns"])
        row["inverted_at_RL_entry"] = (first is not None and row["gravity_z_at_first_RL"] > .8)
        row["tail_stable90"] = row["tail_stable_fraction"] >= .9
        for key, value in row.items():
            if isinstance(value, (int, float)):
                writer.add_scalar("RMCSInverted/" + row["name"] + "/" + key, value, step)
        rows.append(row)
    writer.close()

    def counts(selected):
        return dict(episodes=len(selected), **{key: sum(r[key] for r in selected) for key in
                    ("survived", "recovered", "followup_success", "tail_stable90")})

    out = dict(schema="rmcs-inverted-native-review-v2", run=str(args.run),
               checkpoint_sha256=digest(checkpoint), learning_clock=step, seconds=report["seconds"],
               all=counts(rows), actually_inverted_at_first_RL=counts([r for r in rows if r["inverted_at_RL_entry"]]),
               post_settle_family_counts=dict(Counter(r["post_settle_family"] for r in rows)),
               termination_counts=dict(Counter(c for r in rows for c in r["termination_causes"])),
               max_leg_speed_rad_s=max(r["max_leg_speed_rad_s"] or 0. for r in rows), rows=rows,
               trajectory_sha256=digest(trace_path), new_PPO=0, qualification=None, hardware_ready=False,
               scope="Birth orientation alone is not policy recovery; examine actual settled class and first controlled gravity. Short tests are preliminary.")
    write_json(args.output, out)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

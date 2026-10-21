#!/usr/bin/env python3
"""Summarize frozen native episodes, separating spawn from settled attitude."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def analyze(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    trajectories = np.load(directory / "trajectory.npz")
    groups = defaultdict(list)
    for lane in report["lanes"]:
        groups[lane["name"]].append(lane)
    cases = {}
    for name, rows in groups.items():
        result = dict(episodes=len(rows), survived=sum(r["survived"] for r in rows),
            recovered=sum(r["recovered"] for r in rows),
            recovery_and_motion_success=sum(r["followup_success"] and r["survived"] for r in rows),
            post_settle_families=dict(), max_gap_mm=max(r["max_gap_m"] for r in rows) * 1000.)
        for row in rows:
            family = row["post_settle_family"] or "not_enabled"
            result["post_settle_families"][family] = result["post_settle_families"].get(family, 0) + 1
        for key in ("vx_mean_m_s", "yaw_mean_rad_s", "vx_mae_m_s", "yaw_mae_rad_s"):
            values = [r[key] for r in rows if r[key] is not None]
            result[key] = float(np.mean(values)) if values else None
        durations, enabled, final_gz, best_gz, best_height = [], [], [], [], []
        reasons = defaultdict(int)
        for row in rows:
            lane = row["lane"]
            mask = trajectories["alive"][:, lane]
            times = trajectories["time"][:, lane]
            indices = np.flatnonzero(mask)
            if len(indices):
                last = indices[-1]
                durations.append(float(times[last]))
                final_gz.append(float(trajectories["gravity"][last, lane, 2]))
                best_gz.append(float(trajectories["gravity"][indices, lane, 2].min()))
                best_height.append(float(trajectories["height"][indices, lane].max()))
            times_enabled = times[mask & ~trajectories["passive"][:, lane]]
            if len(times_enabled):
                enabled.append(float(times_enabled[0]))
            for key in ("mechanical", "closure_gap", "impact_bad"):
                if key in trajectories and (trajectories[key][:, lane] & mask).any():
                    reasons[key] += 1
        result.update(duration_mean_s=float(np.mean(durations)),
            enabled_mean_s=float(np.mean(enabled)) if enabled else None,
            final_gravity_z_mean=float(np.mean(final_gz)), best_gravity_z_mean=float(np.mean(best_gz)),
            best_height_mean_m=float(np.mean(best_height)), sampled_termination_causes=dict(reasons))
        cases[name] = result
    settled = defaultdict(lambda: dict(episodes=0, recovered=0, recovery_and_motion_success=0))
    for row in report["lanes"]:
        if row["recovery"]:
            bucket = settled[row["post_settle_family"] or "not_enabled"]
            bucket["episodes"] += 1
            bucket["recovered"] += int(row["recovered"])
            bucket["recovery_and_motion_success"] += int(row["followup_success"] and row["survived"])
    return dict(evaluation=str(directory.resolve()), policy_hz=report["policy_hz"],
        physics_hz=report["physics_hz"], cases=cases, actual_settled_attitudes=dict(settled),
        passive_motor_peak_nm=report["passive_motor_peak_nm"], max_gap_m=report["max_gap_m"],
        recovered=report["recovered"], recovery_episodes=report["recovery_episodes"],
        recovery_and_motion_success=report["recovery_and_motion_success"],
        motion_survived=report["motion_survived"], motion_episodes=report["motion_episodes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.evaluation)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()

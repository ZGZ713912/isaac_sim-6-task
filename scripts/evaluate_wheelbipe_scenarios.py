#!/usr/bin/env python3
"""Read-only fixed scenarios for a Wheelbipe actor: 20 cm step-up/drive-off and flat jumps.

Each scenario runs scripts/validate_wheelbipe_mixed.py in its own process with a
course override (platform-dominant quotas, fixed rise/speed/mode, everything
unlocked). Arc driving uses scripts/audit_wheelbipe_behavior.py --cases arc_*.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("stand", "forward", "reverse", "height", "turn", "dash", "spin", "jump", "airborne", "platform")


def scenario(rise, speed, over, count):
    # Every task keeps at least one lane (contract rule); the rest are platform lanes.
    other = max(.01, 1.5/count)
    fractions = {name: other for name in TASKS}
    fractions["platform"] = 1.-other*(len(TASKS)-1)
    return dict(task_fractions=fractions, platform=dict(lane_heights_m=[rise], speed_range_m_s=[speed, speed],
        over_fraction=1. if over else 0., unlock=dict(start_update=0, full_update=0, start_height_m=rise,
        final_height_m=rise)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--rise", type=float, default=.2)
    p.add_argument("--speeds", type=float, nargs="+", default=[.5, .75, 1.])
    p.add_argument("--randomization-strength", type=float, default=0.)
    p.add_argument("--training-profile", choices=("foundation-jump-v7", "foundation-jump-v8"), default="foundation-jump-v7")
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for speed in a.speeds:
        for over in (True, False):
            name = f"{'step_up' if over else 'drive_off'}_{a.rise:g}m_{speed:g}mps"
            out = a.output/name
            command = [a.python, "-u", str(ROOT/"scripts/validate_wheelbipe_mixed.py"), "--checkpoint", str(a.checkpoint),
                "--output", str(out), "--num-envs", str(a.num_envs), "--steps", str(a.steps), "--course-update", "20000",
                "--training-profile", a.training_profile, "--randomization-profile",
                "requested" if a.randomization_strength > 0 else "nominal",
                "--randomization-strength", str(a.randomization_strength),
                "--spec-override", json.dumps(scenario(a.rise, speed, over, a.num_envs))]
            with (a.output/(name+".log")).open("w") as log:
                code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
            receipt = json.loads((out/"receipt.json").read_text()) if (out/"receipt.json").exists() else {}
            events = receipt.get("platform_events", {})
            lanes = receipt.get("allocated_envs", {}).get("platform", 0)
            rows.append(dict(scenario=name, exit_code=code, status=receipt.get("status"), platform_lanes=lanes,
                events=events, rates_per_lane={k: (v/lanes if lanes else None) for k, v in events.items()}))
            print(json.dumps(rows[-1]), flush=True)
    (a.output/"summary.json").write_text(json.dumps(dict(checkpoint=str(a.checkpoint), rise_m=a.rise,
        scenarios=rows, interpretation="read-only; rates are events per platform lane over one episode window"), indent=2)+"\n")


if __name__ == "__main__":
    main()

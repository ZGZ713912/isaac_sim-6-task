#!/usr/bin/env python3
"""Read-only fixed scenarios for a Wheelbipe actor: 20 cm step-up/drive-off and flat jumps.

Each scenario runs scripts/validate_wheelbipe_mixed.py in its own process with a
course override (platform-dominant quotas, fixed rise/speed/mode, everything
unlocked). Arc driving uses scripts/audit_wheelbipe_behavior.py --cases arc_*.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
    p.add_argument("--suite",choices=("platform","slope_spin","slope_still","step_down","bump","slope_jump","balls","motion","regression","retention"),default="platform")
    p.add_argument("--course-update",type=int,default=20000)
    p.add_argument("--episodes",type=int,default=64)
    p.add_argument("--seeds",type=int,nargs="+",default=[173,293])
    p.add_argument("--max-cells",type=int,default=None,help="Smoke check only; partial grids cannot qualify")
    p.add_argument("--retention-csv",type=Path,default=None)
    p.add_argument("--reference-summary",type=Path,default=None,
        help="Matched V8 regression summary for the 90% retention comparison")
    p.add_argument("--randomization-profile",choices=("both","nominal","requested"),default="both")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--rise", type=float, default=.2)
    p.add_argument("--speeds", type=float, nargs="+", default=[.5, .75, 1.])
    p.add_argument("--randomization-strength", type=float, default=0.)
    p.add_argument("--training-profile", choices=("foundation-jump-v7", "foundation-jump-v8","foundation-jump-v9","foundation-jump-v10"), default="foundation-jump-v7")
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()
    if a.suite != "platform":
        return evaluate_suite(a)
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


def evaluate_suite(a):
    sys.path.insert(0,str(ROOT/"src"))
    from wheeled_tasks.wheelbipe_port.training_profile import specification
    from wheeled_tasks.wheelbipe_port.scenario_evaluation import scenario_grid,scenario_override,retention_warnings
    from torch.utils.tensorboard import SummaryWriter
    if a.training_profile != "foundation-jump-v9" and not (a.suite == "regression" and a.training_profile == "foundation-jump-v8"):
        raise ValueError("V9 suites require --training-profile foundation-jump-v9")
    if a.episodes < 1 or a.num_envs < 1:
        raise ValueError("Positive episode and target-lane counts are required")
    a.output.mkdir(parents=True,exist_ok=False)
    spec = specification(a.training_profile)
    grid = scenario_grid(a.suite,update=a.course_update)
    expected_cells = len(grid)
    if a.max_cells is not None: grid = grid[:a.max_cells]
    domains = ["nominal","requested"] if a.randomization_profile == "both" else [a.randomization_profile]
    writer = SummaryWriter(str(a.output/"events"))
    rows = []
    reference = json.loads(a.reference_summary.read_text())["scenarios"] if a.reference_summary else []
    try:
        for index,cell in enumerate(grid):
            for domain in domains:
                for seed in a.seeds:
                    target = 16 if cell["task"] == "balls" or a.suite == "retention" else a.num_envs
                    patch,count = scenario_override(spec,cell,target)
                    # Include enough full windows for the requested sample count.
                    episodes = 16 if a.suite == "retention" else a.episodes
                    duration = patch["evaluation"]["duration_s"]
                    steps = max(a.steps,round((duration+.04)/.02)*math.ceil(episodes/target))
                    name = f"{index:03d}_{cell['kind']}_{domain}_seed{seed}"
                    out = a.output/name
                    cmd = [a.python,"-u",str(ROOT/"scripts/validate_wheelbipe_mixed.py"),"--checkpoint",str(a.checkpoint),
                        "--output",str(out),"--num-envs",str(count),"--steps",str(steps),"--seed",str(seed),
                        "--course-update",str(a.course_update),"--training-profile",a.training_profile,
                        "--randomization-profile",domain,"--randomization-strength","1" if domain == "requested" else "0",
                        "--spec-override",json.dumps(patch)]
                    with (a.output/(name+".log")).open("w") as log:
                        code = subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
                    receipt = json.loads((out/"receipt.json").read_text()) if (out/"receipt.json").exists() else {}
                    row = receipt.get("scenario",dict(cell=cell,episodes=0,successes=0,metrics={},passed=False,missing=["native scenario receipt"]))
                    row.update(domain=domain,seed=seed,exit_code=code,native_status=receipt.get("status"),
                        output=str(out),course_update=a.course_update)
                    if code or receipt.get("status") != "complete":
                        row["missing"].append("completed native evaluation")
                        row["passed"] = False
                    if row["episodes"] < episodes:
                        row["missing"].append("requested completed episode population")
                        row["passed"] = False
                    if a.suite == "regression" and a.training_profile == "foundation-jump-v9":
                        matched = [r for r in reference if r["cell"]["name"] == cell["name"]
                            and r["domain"] == domain and r["seed"] == seed]
                        if not matched:
                            row["missing"].append("matched V8 checkpoint evaluation at the same update")
                            row["passed"] = False
                        else:
                            baseline = matched[0]["metrics"].get("success_fraction")
                            value = row["metrics"].get("success_fraction")
                            row["v8_reference_success_fraction"] = baseline
                            row["regression_at_least_90_percent_of_v8"] = (baseline is not None and value is not None and value >= .9*baseline)
                            row["passed"] &= row["regression_at_least_90_percent_of_v8"]
                    rows.append(row)
                    for key,value in row["metrics"].items():
                        if isinstance(value,(int,float)) and math.isfinite(value):
                            writer.add_scalar("Retention/"+cell["name"]+"/"+domain+"/"+key,value,a.course_update)
                    print(json.dumps(row,allow_nan=False),flush=True)
                    (a.output/"summary.json").write_text(json.dumps(dict(suite=a.suite,expected_cells=expected_cells,
                        sampled_cells=len(grid),checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                        course_update=a.course_update,scenarios=rows,behavior_qualified=False),indent=2,allow_nan=False)+"\n")
    finally:
        writer.flush();writer.close()
    if a.suite == "retention":
        csv_path = a.retention_csv or a.output.parent/"retention.csv"
        csv_path.parent.mkdir(parents=True,exist_ok=True)
        fresh = not csv_path.exists()
        with csv_path.open("a",newline="") as stream:
            fields = ["update","cell","domain","seed","episodes","success_fraction","radius_max_m","speed_error_mean_m_s","roll_max_deg","pitch_max_deg","receipt"]
            writer_csv = csv.DictWriter(stream,fieldnames=fields)
            if fresh: writer_csv.writeheader()
            for row in rows:
                record = dict(update=a.course_update,cell=row["cell"]["name"],domain=row["domain"],seed=row["seed"],episodes=row["episodes"],receipt=row["output"])
                record.update({key:row["metrics"].get(key) for key in fields if key in row["metrics"]})
                writer_csv.writerow(record)
        history = []
        for path in sorted(a.output.parent.glob("*/summary.json")):
            if path == a.output/"summary.json": continue
            old = json.loads(path.read_text())
            if old.get("suite") == "retention" and old.get("course_update",0) < a.course_update:
                history.append(old["scenarios"])
        warnings = retention_warnings(history,rows)
        (a.output/"warnings.json").write_text(json.dumps(warnings,indent=2)+"\n")
        if warnings: print("RETENTION WARNING "+json.dumps(warnings),flush=True)


if __name__ == "__main__":
    main()

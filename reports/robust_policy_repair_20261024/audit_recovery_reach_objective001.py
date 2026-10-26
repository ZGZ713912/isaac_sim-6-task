"""Replay a proposed cost; do not change a policy, native state, or reward."""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from recovery_support_candidate001 import recovery_support_cost

torch.set_num_threads(1)
base = Path("runs/robust_policy_repair_20261024")
rows = []
for name in ("diagnosis012_final_circular_fixed", "diagnosis014_recovery_bias015_fixed",
             "diagnosis014_recovery_bias030_fixed", "diagnosis016_entry_bias015_fixed",
             "diagnosis012_final_arithmetic_nominal/robust", "diagnosis015_bias015_nominal/robust"):
    directory = base / name
    if (directory / "evaluation").exists():
        directory = directory / "evaluation"
    report = json.loads((directory / "report.json").read_text())
    trajectory = directory / "trajectory.npz"
    assert hashlib.sha256(trajectory.read_bytes()).hexdigest() == report["trajectory_sha256"]
    z = np.load(trajectory)
    active = z["alive"] & ~z["passive"]
    # Explicitly restricted to the first recovery episode of designated lanes.
    # No claim that this reconstructs all internal recovery_active transitions.
    lane_recovery = np.array([lane["recovery"] for lane in report["lanes"]])
    unrecovered = lane_recovery[None] & ~z["recovered"]
    cost = recovery_support_cost(torch.tensor(z["wheel_clearance"]),
        torch.tensor(z["gravity"][..., 2]), torch.tensor(unrecovered), 1.).numpy()
    stalled = active & unrecovered & (z["gravity"][..., 2] < -.9) & (z["wheel_clearance"].min(-1) > .3)
    writer = SummaryWriter(str(directory / "metrics"), filename_suffix=".recovery-reach-audit")
    selected = []
    for lane in report["lanes"]:
        if not lane["recovery"]:
            continue
        i = lane["lane"]
        mask = active[:, i] & unrecovered[:, i]
        row = dict(name=lane["name"], family=lane["post_settle_family"],
            recovered=lane["recovered"], followup=lane["followup_success"],
            survived=lane["survived"], termination=lane["termination_causes"],
            unrecovered_active_seconds=float(mask.sum() * .005),
            added_cost_integral=float((cost[:, i] * mask).sum() * .005),
            added_cost_max_density=float(cost[mask, i].max()) if mask.any() else 0.,
            stalled_seconds=float(stalled[:, i].sum() * .005))
        selected.append(row)
        for key, value in row.items():
            if isinstance(value, (float, int)):
                writer.add_scalar("RecoveryReachAudit/" + row["name"] + "/" + key, value, 1501)
    writer.close()
    rows.append(dict(run=name, trajectory_sha256=report["trajectory_sha256"],
                     rows=selected, total_cost_integral=sum(r["added_cost_integral"] for r in selected)))
    print(name, "stalled seconds", round(sum(r["stalled_seconds"] for r in selected), 3),
          "cost integral", round(rows[-1]["total_cost_integral"], 3))
out = dict(schema="recovery-reach-objective-counterfactual-v1", proposed_weight=1.,
    coordinate="Native cylinder bottom clearance,15cm scale, continuous upright gate:gz-.7..-.9",
    candidate_source_sha256=hashlib.sha256(Path(__file__).with_name("recovery_support_candidate001.py").read_bytes()).hexdigest(),
    evaluations=rows, scope="Only added density cost before first recovery is replayed. These are not total returns, policy gradients, or evidence of learned improvement. Short runs do not qualify a policy.",
    current_policy_changed=False, reward_changed_in_native_simulation=False,
    new_PPO=0, actual_experiment_consumed=2587, qualification=None,
    hardware_started=False, created_unix=time.time())
Path("reports/robust_policy_repair_20261024/recovery_reach_objective_audit001.json").write_text(json.dumps(out, indent=2) + "\n")

"""Read-only roll-specific early-authority hypothesis; no actor/physics change."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo.robust_getup import tilt_gate


def early_roll_gate(obs):
    baseline = tilt_gate(obs, True, True)
    gravity = obs[:, 7:10]
    lateral = gravity[:, 1:2].abs() / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    start, full = .4067366430758002, .7071067811865475  # sin24deg, sin45deg
    roll = ((lateral - start) / (full - start)).clamp(0., 1.)
    return torch.maximum(baseline, roll)


def evidence(name):
    directory = RUNS / name / "robust"
    report = json.loads((directory / "report.json").read_text())
    path = directory / "trajectory.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["trajectory_sha256"]
    assert json.loads((directory.parent / "progress.json").read_text())["status"] == "completed"
    return directory, report, np.load(path)


def main():
    torch.set_num_threads(1)
    result = dict(schema="roll072-authority-readonly-v1", start_degrees=24., full_degrees=45.,
        hypothesis="Current stance-aware severe branch begins at60deg; add IMU lateral gravity authority before over-center roll",
        ordinary={}, disturbances=[], new_macro_updates=0, native_policy_changed=False,
        long_training_admitted=False, hardware_ready=False,
        limitation="Timing on recorded observations only; changed actions may alter the trajectory and cannot establish benefit")
    directory, report, data = evidence("high_recenter_full066")
    selected = [row["lane"] for row in report["lanes"] if not row["recovery"] and not row.get("push")]
    obs = data["observations"][:, selected]
    alive = data["alive"][:, selected]
    x = torch.from_numpy(obs.reshape(-1, 35).copy())
    with torch.inference_mode():
        old, new = tilt_gate(x, True, True).ravel(), early_roll_gate(x).ravel()
    active = torch.from_numpy(alive.reshape(-1).copy())
    delta = (new - old)[active]
    result["ordinary"] = dict(valid_frames=int(active.sum()), changed_gate_frames=int((delta != 0).sum()),
        maximum_gate_delta=float(delta.max()), trajectory_sha256=report["trajectory_sha256"])
    assert result["ordinary"]["changed_gate_frames"] == 0
    for name in ("stationary_reference_after065", "stationary_reference_after071"):
        directory, report, data = evidence(name)
        rows = []
        for row in report["lanes"]:
            i = row["lane"]
            alive = data["alive"][:, i] & (data["time"][:, i] >= 5.)
            x = torch.from_numpy(data["observations"][:, i].copy())
            with torch.inference_mode():
                old, new = [v.numpy().ravel() for v in (tilt_gate(x, True, True), early_roll_gate(x))]
            old_idx, new_idx = [np.flatnonzero((gate >= .9) & alive) for gate in (old, new)]
            timing = dict(lane=i, case=row["name"], push_scale=row.get("diagnostic_push_scale", 1.),
                actual_fall=row["in_motion_fall"], survived=row["survived"], causes=row["termination_causes"],
                original_90_time_s=float(data["time"][old_idx[0], i]) if len(old_idx) else None,
                proposed_90_time_s=float(data["time"][new_idx[0], i]) if len(new_idx) else None)
            if len(old_idx) and len(new_idx):
                timing["earlier_s"] = timing["original_90_time_s"] - timing["proposed_90_time_s"]
            rows.append(timing)
        changed = [row["earlier_s"] for row in rows if row.get("earlier_s", 0.) > 0.]
        summary = dict(evaluation=name, trajectory_sha256=report["trajectory_sha256"], lanes=rows,
            earlier_90_lanes=len(changed), earlier_s_min=min(changed) if changed else None,
            earlier_s_max=max(changed) if changed else None)
        result["disturbances"].append(summary)
        writer = SummaryWriter(str(directory.parent), filename_suffix=".roll_gate072")
        writer.add_scalar("Diagnosis/RollGate072/earlier90_lanes", len(changed), 505)
        if changed:
            writer.add_scalar("Diagnosis/RollGate072/earlier90_seconds_mean", float(np.mean(changed)), 505)
        writer.close()
    path = REPORTS / "roll_gate072_probe.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(ordinary=result["ordinary"], disturbances=[{key: row[key] for key in (
        "evaluation", "earlier_90_lanes", "earlier_s_min", "earlier_s_max")} for row in result["disturbances"]])), flush=True)


if __name__ == "__main__":
    main()

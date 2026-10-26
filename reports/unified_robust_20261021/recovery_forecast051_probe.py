"""Read-only forecast of the whole recovery gate, including the phase branch.

The earlier046 probe raised only the severe branch. This probe can change
partial authority before60deg, but only while already beyond30deg and moving
outward. Native source, parameters and thresholds remain untouched.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo.robust_getup import tilt_gate


def main():
    torch.set_num_threads(1)
    run = ROOT / "runs/unified_robust_20261022/side_confirmation032/robust"
    report = json.loads((run / "report.json").read_text())
    assert hashlib.sha256((run / "trajectory.npz").read_bytes()).hexdigest() == report["trajectory_sha256"]
    data = np.load(run / "trajectory.npz")
    obs = data["observations"].astype(np.float64)
    gyro, gravity = obs[:, :, 4:7] * 2., obs[:, :, 7:10]
    norm = np.linalg.norm(gyro, axis=-1, keepdims=True)
    axis = gyro / np.maximum(norm, 1e-12)
    old = tilt_gate(torch.from_numpy(obs.reshape(-1, 35)), True, True).numpy().reshape(obs.shape[:2])
    active = data["alive"] & ~data["passive"]
    ordinary = np.array([c["lane"] for c in report["lanes"] if not c["recovery"] and not c.get("push")])
    results = []
    for horizon in (.05, .1, .15):
        angle = norm * horizon
        future = (gravity * np.cos(angle) - np.cross(axis, gravity) * np.sin(angle)
                  + axis * (axis * gravity).sum(-1, keepdims=True) * (1. - np.cos(angle)))
        eligible = (gravity[:, :, 2] > -np.cos(np.pi / 6)) & (future[:, :, 2] > gravity[:, :, 2])
        future_obs = obs.copy()
        future_obs[:, :, 7:10] = future
        whole = tilt_gate(torch.from_numpy(future_obs.reshape(-1, 35)), True, True).numpy().reshape(old.shape)
        forecast = np.where(eligible, np.maximum(old, whole), old)
        changed = (forecast - old)[:, ordinary][active[:, ordinary]]
        rows = []
        for case in report["lanes"]:
            if not case.get("push"):
                continue
            i = case["lane"]
            mask = active[:, i] & (data["time"][:, i] >= 5.)

            def first(predicate):
                indices = np.flatnonzero(mask & predicate)
                return float(data["time"][indices[0], i]) if len(indices) else None

            before, after = first(old[:, i] >= .9), first(forecast[:, i] >= .9)
            before_half, after_half = first(old[:, i] >= .5), first(forecast[:, i] >= .5)
            no_contact = first(~data["wheel_contact"][:, i].any(-1))
            rows.append(dict(lane=i, case=case["name"],
                gate09_earlier_ms=(before-after)*1000 if before is not None and after is not None else None,
                gate05_earlier_ms=(before_half-after_half)*1000 if before_half is not None and after_half is not None else None,
                gate09_before_no_contact_ms=(no_contact-after)*1000 if no_contact is not None and after is not None else None,
                changed_frames=int((mask & (forecast[:, i] > old[:, i] + 1e-7)).sum())))
        results.append(dict(horizon_s=horizon, ordinary_changed_frames=int((changed > 1e-7).sum()),
            ordinary_max_gate_change=float(changed.max()), rows=rows))
    output = dict(schema="recovery051-whole-gate-forecast-offline-v1", optimizer_updates=0, native_runs=0,
        checkpoint_sha256=report["checkpoint_sha256"], trajectory_sha256=report["trajectory_sha256"],
        ordinary_active_frames=int(active[:, ordinary].sum()), trigger="measured tilt>30deg and predicted tilt increases",
        results=results, scope="Report-only counterfactual on old trajectories. No native behavior improvement or deployment admission follows; closed-loop states would change.")
    step = json.loads((run.parent / "progress.json").read_text())["completed_updates"]
    with SummaryWriter(str(run.parent), filename_suffix=".forecast_probe051") as writer:
        for row in results:
            prefix = f"Diagnosis/recovery_forecast051/{round(row['horizon_s']*1000)}ms/"
            writer.add_scalar(prefix + "ordinary_changed_frames", row["ordinary_changed_frames"], step)
            writer.add_scalar(prefix + "push_gate09_earlier_max_ms", max(r["gate09_earlier_ms"] for r in row["rows"]), step)
    Path(__file__).with_suffix(".json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()

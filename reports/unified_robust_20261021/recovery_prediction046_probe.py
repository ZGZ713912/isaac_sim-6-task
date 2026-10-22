"""Offline50ms IMU gravity forecast probe; no native policy intervention."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "src")]
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
    angle = norm * .05
    # g_dot=-omega cross g in the body frame. The forecast is a constant
    # measured-rate hypothesis, not a prediction of an actual next trajectory.
    future = (gravity * np.cos(angle) - np.cross(axis, gravity) * np.sin(angle)
              + axis * (axis * gravity).sum(-1, keepdims=True) * (1. - np.cos(angle)))
    old = tilt_gate(torch.from_numpy(obs.reshape(-1, 35)), True, True).numpy().reshape(obs.shape[:2])
    severe = np.clip((future[:, :, 2] + .5) / (.5 - .08715574274765817), 0., 1.)
    predicted = np.maximum(old, severe)
    delta = predicted - old
    active = data["alive"] & ~data["passive"]
    ordinary = np.array([c["lane"] for c in report["lanes"] if not c["recovery"] and not c.get("push")])
    rows = []
    for case in report["lanes"]:
        if not case.get("push"):
            continue
        lane = case["lane"]
        selected = active[:, lane] & (data["time"][:, lane] >= 5.)

        def first(mask):
            indices = np.flatnonzero(selected & mask)
            return float(data["time"][indices[0], lane]) if len(indices) else None

        old_time = first(old[:, lane] >= .9)
        new_time = first(predicted[:, lane] >= .9)
        contact_time = first(~data["wheel_contact"][:, lane].any(-1))
        rows.append(dict(lane=lane, name=case["name"], observed_old_gate09_s=old_time,
            counterfactual_gate09_s=new_time,
            earlier_gate09_ms=(old_time - new_time) * 1000. if old_time is not None and new_time is not None else None,
            first_both_wheels_no_contact_s=contact_time,
            counterfactual_gate09_before_no_contact_ms=(contact_time - new_time) * 1000. if contact_time is not None and new_time is not None else None,
            changed_input_frames=int((selected & (delta[:, lane] > 1e-7)).sum())))
    changed = delta[:, ordinary][active[:, ordinary]]
    result = dict(schema="recovery046-forecast-offline-probe-v1", lookahead_seconds=.05,
        source_trajectory_sha256=report["trajectory_sha256"], checkpoint_sha256=report["checkpoint_sha256"],
        method="Raise only the severe-tilt branch from forecast gravity; keep the existing phase/shape branch and present authority unchanged",
        input="existing35D measured gyro and gravity only", additional_state=False,
        ordinary_nonpush_active_frames=len(changed), ordinary_nonpush_changed_gate_frames=int((changed > 1e-7).sum()),
        ordinary_nonpush_maximum_gate_change=float(changed.max()), rows=rows,
        native_runs=0, optimizer_updates=0, implementation_scope="report-only offline forecast; not installed in native actor or deployed model",
        limits="The old trajectory is fixed. Changed authority would change future states; no native recovery improvement or closed-loop equivalence follows.",
        hardware_ready=False)
    from torch.utils.tensorboard import SummaryWriter
    step = json.loads((run.parent / "progress.json").read_text())["completed_updates"]
    metrics = dict(ordinary_nonpush_changed_gate_frames=result["ordinary_nonpush_changed_gate_frames"],
                   ordinary_nonpush_maximum_gate_change=result["ordinary_nonpush_maximum_gate_change"],
                   push_gate09_earlier_max_ms=max(r["earlier_gate09_ms"] for r in rows))
    with SummaryWriter(log_dir=str(run.parent), filename_suffix=".forecast_probe046") as writer:
        for tag, value in metrics.items():
            writer.add_scalar("Diagnosis/recovery_prediction046/" + tag, value, step)
    result["tensorboard"] = dict(run=str(run.parent.relative_to(ROOT)), step=step,
                                 tag_prefix="Diagnosis/recovery_prediction046/", scalar_values=metrics)
    (REPORTS / "recovery_prediction046_probe.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit old/new geometry validity on recorded policy sensor frames, zero PPO."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from train_robust_getup import write_json, digest
    from wheeled_algo.rolling_velocity import RollingBaseVelocity, TABLE_SHA256
    from wheeled_algo.standing_displacement import WheelAngleCoordinate
    from wheeled_tasks.chassis.standing_geometry107 import table_path, EXTENDED_SHA256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "evaluation/report.json").read_text())
    progress = json.loads((args.run / "progress.json").read_text())
    if progress["status"] != "completed":
        raise ValueError("Geometry audit requires a complete recorded evaluation")
    trace_path = args.run / "evaluation/trajectory.npz"
    if digest(trace_path) != report["trajectory_sha256"]:
        raise ValueError("Recorded trajectory hash differs")
    data = np.load(trace_path)
    step = torch.load(progress["checkpoint"], map_location="cpu", weights_only=False)["infos"]["completed_updates"]
    legacy, extended = WheelAngleCoordinate(), WheelAngleCoordinate()
    extended.geometry = RollingBaseVelocity(json.loads(table_path().read_text()))
    rows = []
    writer = SummaryWriter(str(args.run / "evaluation/metrics"), filename_suffix=".geometry-audit")
    for lane in report["lanes"]:
        if not lane["behavior_case"].startswith("stand"):
            continue
        i = lane["lane"]
        sensor = torch.from_numpy(data["observations"][:, i])
        # Validity uses four measured leg encoders and IMU gravity only. Wheel
        # angles here are dummy finite values; no simulator pose substitutes
        # for the actual delayed/noisy policy measurements.
        q = torch.zeros(len(sensor), 6)
        q[:, :4] = sensor[:, 10:14] + legacy.geometry.nominal
        old = legacy(q, sensor[:, 7:10])[1][:, 0].numpy()
        new = extended(q, sensor[:, 7:10])[1][:, 0].numpy()
        tail = data["alive"][:, i] & ~data["passive"][:, i] & (data["time"][:, i] >= report["seconds"] - 4.)
        row = dict(name=lane["name"], tail_frames=int(tail.sum()),
                   legacy_coordinate_valid_fraction=float(old[tail].mean()),
                   extended_coordinate_valid_fraction=float(new[tail].mean()),
                   newly_supported_fraction=float((new & ~old)[tail].mean()))
        for key, value in row.items():
            if isinstance(value, float):
                writer.add_scalar("GapStanding/" + lane["name"] + "/" + key, value, step)
        rows.append(row)
    writer.close()
    write_json(args.output, dict(schema="standing-geometry-validity-audit-v1", run=str(args.run),
        trajectory_sha256=digest(trace_path), old_table_sha256=TABLE_SHA256,
        extended_table_sha256=EXTENDED_SHA256, rows=rows, optimizer_updates=0,
        scope="Recorded policy sensor-frame validity only; no counterfactual behavior claim"))
    print(json.dumps(rows))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure sensor-range rejection near the unchanged physical knee stops."""
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
    from train_robust_getup import digest, write_json
    from wheeled_tasks.chassis.standing_geometry107 import table_path
    from wheeled_algo.rolling_velocity import RollingBaseVelocity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    progress = json.loads((args.run / "progress.json").read_text())
    if progress["status"] != "completed":
        raise ValueError("Require a complete native trace")
    report = json.loads((args.run / "evaluation/report.json").read_text())
    path = args.run / "evaluation/trajectory.npz"
    if digest(path) != report["trajectory_sha256"]:
        raise ValueError("Trace hash differs")
    trace = np.load(path)
    settings = report["robust_domains"]["domains"]
    bias, sigma = settings["encoder_bias_rad"], settings["encoder_noise_rad"]
    relative_error_bound = 2. * bias + 4. * np.sqrt(2.) * sigma
    geom = RollingBaseVelocity(json.loads(table_path().read_text()))
    limits = np.array([[getattr(geom, f"side{i}_delta_rad")[j].item() for j in (0, -1)] for i in (0, 1)])
    nominal = geom.nominal.numpy()
    step = torch.load(progress["checkpoint"], map_location="cpu", weights_only=False)["infos"]["completed_updates"]
    rows = []
    writer = SummaryWriter(str(args.run / "evaluation/metrics"), filename_suffix=".sensor-limits")
    for lane in report["lanes"]:
        if not lane["behavior_case"].startswith("stand"):
            continue
        i = lane["lane"]
        q = trace["observations"][:, i, 10:14] + nominal
        difference = q[:, [1, 3]] - q[:, [0, 2]]
        difference = np.arctan2(np.sin(difference), np.cos(difference))
        overshoot = np.maximum(np.maximum(limits[:, 0] - difference, difference - limits[:, 1]), 0.)
        gravity = trace["observations"][:, i, 7:10]
        gravity = gravity / np.linalg.norm(gravity, axis=-1, keepdims=True)
        tail = trace["alive"][:, i] & ~trace["passive"][:, i] & (trace["time"][:, i] >= report["seconds"] - 4.)
        # Align the stored post-step mechanical sample with the earlier policy
        # frame and its declared integer sensor delay. This remains a recorded
        # comparison; it does not initialize or correct the policy observer.
        delay = lane["first_reset_domain"]["sensor_delay_policy_ticks"]
        previous = np.maximum(np.arange(len(q)) - 1 - delay, 0)
        actual = trace["motor_position"][previous, i][:, [0, 1, 3, 4]]
        actual_difference = actual[:, [1, 3]] - actual[:, [0, 2]]
        actual_difference = np.arctan2(np.sin(actual_difference), np.cos(actual_difference))
        actual_overshoot = np.maximum(np.maximum(limits[:, 0] - actual_difference,
                                                actual_difference - limits[:, 1]), 0.)
        row = dict(name=lane["name"], frames=int(tail.sum()),
                   within_gravity_gate_fraction=float((gravity[tail, 2] < -.94).mean()),
                   sensor_relative_inside_fraction=float((overshoot[tail].max(-1) == 0.).mean()),
                   sensor_relative_within_uncertainty_fraction=float((overshoot[tail].max(-1) <= relative_error_bound).mean()),
                   sensor_overshoot_max_rad=float(overshoot[tail].max()),
                   mechanical_relative_inside_fraction=float((actual_overshoot[tail].max(-1) <= 1e-6).mean()),
                   mechanical_overshoot_max_rad=float(actual_overshoot[tail].max()),
                   displacement_observation_nonzero_fraction=float((np.abs(trace["observations"][tail, i, 14:16]).max(-1) > 1e-7).mean()))
        for key, value in row.items():
            if isinstance(value, (int, float)):
                writer.add_scalar("StandingSensorLimits/" + row["name"] + "/" + key, value, step)
        rows.append(row)
    writer.close()
    out = dict(schema="standing-sensor-limit-audit-v1", run=str(args.run), rows=rows,
               trajectory_sha256=digest(path), encoder_bias_bound_rad=bias,
               encoder_noise_sigma_rad=sigma, candidate_relative_uncertainty_bound_rad=relative_error_bound,
               derivation="Two bounded encoder biases plus four standard deviations of their independent noise difference",
               scope="Recorded measurements only. A prospective bounded sensor projection would retain the physical 40..107 degree knee and all mechanical guards.",
               new_PPO=0, hardware_ready=False)
    write_json(args.output, out)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

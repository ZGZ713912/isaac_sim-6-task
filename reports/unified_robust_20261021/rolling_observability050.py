"""Read-only per-episode observability check; no actor receives truth velocity."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).parent)]
from iteration036_compare import read, CASES
from wheeled_algo.rolling_velocity import frozen_rolling_estimator


def main():
    torch.set_num_threads(1)
    evidence = read("quiet_velocity_baseline036")
    directory, report, _, data = evidence
    estimator = frozen_rolling_estimator()
    rows = []
    for lane in report["lanes"]:
        if lane["name"] not in CASES:
            continue
        i = lane["lane"]
        inputs = torch.from_numpy(data["observations"][1:, i].copy())
        with torch.inference_mode():
            velocity, validity = estimator(inputs)
        estimated = velocity.numpy().ravel()
        valid = validity.numpy().ravel().astype(bool)
        # Observations are before the policy step; align against previous truth.
        actual = data["velocity"][:-1, i, 0]
        selected = data["alive"][1:, i] & (data["time"][1:, i] >= 3.) & valid
        error = estimated[selected] - actual[selected]
        dt = np.diff(data["time"][:, i])
        integrated_error = float(np.sum(error * dt[selected]))
        active = data["alive"][:, i] & ~data["passive"][:, i]
        xy = data["local_position"][active, i, :2]
        excursion = np.linalg.norm(xy - xy[:1], axis=-1)
        peak = int(excursion.argmax())
        rows.append(dict(lane=i, case=lane["name"], domain_enabled=lane["domain_enabled"],
            full_excursion_m=float(excursion[peak]), peak_displacement_xy_m=(xy[peak] - xy[0]).tolist(),
            steady_true_vx_m_s=float(actual[selected].mean()),
            steady_estimated_vx_m_s=float(estimated[selected].mean()),
            mean_velocity_error_m_s=float(error.mean()),
            velocity_mae_m_s=float(np.abs(error).mean()),
            integrated_body_forward_velocity_error_m=integrated_error,
            valid_fraction=float(valid.mean())))
    summary = {}
    for case in CASES:
        group = [r for r in rows if r["case"] == case]
        worst = max(group, key=lambda r: r["full_excursion_m"])
        summary[case] = dict(worst_excursion_lane=worst["lane"],
            worst_excursion_m=worst["full_excursion_m"],
            worst_lane_true_vx_m_s=worst["steady_true_vx_m_s"],
            worst_lane_estimated_vx_m_s=worst["steady_estimated_vx_m_s"],
            max_abs_mean_velocity_error_m_s=max(abs(r["mean_velocity_error_m_s"]) for r in group),
            max_abs_integrated_velocity_error_m=max(abs(r["integrated_body_forward_velocity_error_m"]) for r in group))
    output = dict(schema="rolling050-observability-v1", source_trajectory_sha256=report["trajectory_sha256"],
        source_report_sha256=hashlib.sha256((directory / "report.json").read_bytes()).hexdigest(),
        optimizer_updates=0, native_runs=0, summary=summary, lanes=rows,
        scope="Offline error labels only; integrated body-forward velocity error is not a deployed position observer or world-frame position error. No feedback gain, training state, domain or threshold changes.")
    with SummaryWriter(str(directory.parent), filename_suffix=".rolling_observability050") as writer:
        for case, values in summary.items():
            for name, value in values.items():
                if name != "worst_excursion_lane":
                    writer.add_scalar(f"Diagnosis/rolling_observability050/{case}/{name}", value, 441)
    target = Path(__file__).with_suffix(".json")
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

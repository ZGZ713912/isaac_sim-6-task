"""Separate startup excursion, steady translation and oscillation evidence."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(REPORTS)]
from iteration036_compare import read, standing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", default=["quiet_velocity_baseline036", "coupled_state_static045", "bias_ablation_static047"])
    args = parser.parse_args()
    results = {}
    for name in args.runs:
        evidence = read(name)  # Includes trajectory SHA authentication.
        directory, report, _, data = evidence
        reference = standing(evidence)
        progress = json.loads((directory.parent / "progress.json").read_text())
        rows = []
        for case in report["lanes"]:
            lane = case["lane"]
            time = data["time"][:, lane]
            full = data["alive"][:, lane] & ~data["passive"][:, lane]
            early = full & (time < 3.)
            late = full & (time >= 3.)
            xy = data["local_position"][:, lane, :2].astype(np.float64)
            full_max = float(np.linalg.norm(xy[full] - xy[full][:1], axis=-1).max())
            early_max = float(np.linalg.norm(xy[early] - xy[full][:1], axis=-1).max()) if early.any() else None
            row = dict(lane=lane, name=case["name"], domain_enabled=case["domain_enabled"],
                       survived=case["survived"], full_max_excursion_m=full_max,
                       startup_0_3s_max_excursion_m=early_max)
            if late.sum() >= 200:
                values = xy[late]
                t = time[late].astype(np.float64)
                centered_time = t - t.mean()
                slope = (centered_time[:, None] * (values - values.mean(0))).sum(0) / (centered_time ** 2).sum()
                row.update(steady_3_12s_net_displacement_m=float(np.linalg.norm(values[-1] - values[0])),
                           steady_3_12s_max_excursion_m=float(np.linalg.norm(values - values[:1], axis=-1).max()),
                           steady_3_12s_body_vx_mean_m_s=float(data["velocity"][late, lane, 0].mean()),
                           steady_3_12s_world_xy_drift_slope_m_s=float(np.linalg.norm(slope)))
            rows.append(row)
        groups = {}
        for case in ("stand_low", "stand", "stand_high"):
            selected = [r for r in rows if r["name"] == case]
            valid = [r for r in selected if "steady_3_12s_net_displacement_m" in r]
            assert sum(r["full_max_excursion_m"] <= .05 for r in selected) == reference[case]["full_window_within_5cm"]
            group = dict(episodes=len(selected), completed=sum(r["survived"] for r in selected),
                episodes_with_steady_data=len(valid),
                full_window_within5cm_count=sum(r["survived"] and r["full_max_excursion_m"] <= .05 for r in selected),
                steady_window_within5cm_count=sum(r["survived"] and r["steady_3_12s_max_excursion_m"] <= .05 for r in valid),
                startup_max_excursion_worst_m=max(r["startup_0_3s_max_excursion_m"] for r in selected),
                steady_net_displacement_median_m=float(np.median([r["steady_3_12s_net_displacement_m"] for r in valid])),
                steady_net_displacement_worst_m=max(r["steady_3_12s_net_displacement_m"] for r in valid),
                steady_drift_slope_worst_m_s=max(r["steady_3_12s_world_xy_drift_slope_m_s"] for r in valid),
                steady_body_vx_mean_median_m_s=float(np.median([r["steady_3_12s_body_vx_mean_m_s"] for r in valid])))
            groups[case] = group
        with SummaryWriter(log_dir=str(directory.parent), filename_suffix=".standing_segments047") as writer:
            for case, metrics in groups.items():
                for tag, value in metrics.items():
                    writer.add_scalar(f"Diagnosis/standing_segments/{case}/{tag}", value, progress["completed_updates"])
        results[name] = dict(checkpoint_sha256=report["checkpoint_sha256"], trajectory_sha256=report["trajectory_sha256"],
            tensorboard_run=str(directory.parent.relative_to(ROOT)), tensorboard_step=progress["completed_updates"],
            groups=groups, rows=rows)
    result = dict(schema="standing047-segmented-drift-v1", optimizer_updates=0, results=results,
        scope="0-3s excursion relative to first active position; 3-12s excursion relative to3s position and independent net translation/OLS slope. None replaces full-window5cm admission or frequency metrics.")
    (REPORTS / "standing_segments047.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v["groups"] for k, v in results.items()}))


if __name__ == "__main__":
    main()

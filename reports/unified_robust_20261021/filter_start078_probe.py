"""Read-only startup filter/position evidence after074 and076 failed static gates."""
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
from wheeled_algo.robust_getup import quiet_gate


def main():
    torch.set_num_threads(1)
    result = dict(schema="filter078-startup-readonly-v1", evaluations={}, new_macro_updates=0,
        hypothesis="Keep full wheel filtering within the small standing-position band; release extra delay during larger position corrections",
        changed_action_native_behavior_proven=False)
    for name in ("high_recenter_static066", "quiet_whole_static074", "filtered_quiet_static076"):
        directory = RUNS / name / "robust"
        report = json.loads((directory / "report.json").read_text())
        path = directory / "trajectory.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report["trajectory_sha256"]
        data = np.load(path)
        rows = {}
        for case in ("stand_low", "stand", "stand_high"):
            lanes = [row["lane"] for row in report["lanes"] if row["name"] == case]
            positions = data["local_position"][:, lanes, :2]
            excursion = np.linalg.norm(positions - positions[:1], axis=-1)
            flat = int(np.argmax(excursion))
            t, index = np.unravel_index(flat, excursion.shape)
            lane = lanes[index]
            obs = data["observations"][:, lanes]
            coords = np.abs(obs[:, :, 14:16] / 20.).max(-1)
            times = data["time"][:, lanes]
            alive = data["alive"][:, lanes]
            steady = alive & (times >= 3.)
            quiet = quiet_gate(torch.from_numpy(obs.reshape(-1, 35).copy())).numpy().reshape(times.shape)
            kneespeed = np.abs(obs[:, :, [17, 19]] - obs[:, :, [16, 18]]).max(-1) * 10.
            rows[case] = dict(worst_lane=lane, maximum_excursion_m=float(excursion[t, index]),
                peak_time_s=float(data["time"][t, lane]), peak_estimated_coordinate_m=float(coords[t, index]),
                peak_quiet_gate=float(quiet[t, index]), peak_knee_speed_rad_s=float(kneespeed[t, index]),
                peak_height_error_m=float(data["height"][t, lane] - data["commands"][t, lane, 2]),
                steady_coordinate_gt_1cm_fraction=float((coords[steady] > .01).mean()),
                steady_coordinate_gt_25mm_fraction=float((coords[steady] > .025).mean()),
                steady_coordinate_peak_m=float(coords[steady].max()),
                steady_knee_speed_gt_02_fraction=float((kneespeed[steady] > .2).mean()))
        result["evaluations"][name] = dict(cases=rows, trajectory_sha256=report["trajectory_sha256"])
        writer = SummaryWriter(str(directory.parent), filename_suffix=".filter_start078")
        for case, row in rows.items():
            for key, value in row.items():
                writer.add_scalar("Diagnosis/QuietFilter078/" + case + "/" + key, value,
                                  537 if name.endswith("076") else 505)
        writer.close()
    (REPORTS / "filter_start078_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: row["cases"] for name, row in result["evaluations"].items()}), flush=True)


if __name__ == "__main__":
    main()

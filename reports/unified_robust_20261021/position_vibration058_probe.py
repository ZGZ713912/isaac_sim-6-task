"""Split actual quiet motion into bands on completed, hash-checked trajectories."""
import json
from pathlib import Path
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "scripts"), str(REPORTS)]
from analyze_robust_getup import hf_rms
from iteration036_compare import CASES, paired, read


def main():
    names = ("quiet_velocity_baseline036", "learned_position_static058", "low_transfer_static059", "high_transfer_static060")
    baseline = read(names[0])
    result = dict(schema="position058-actual-motion-band-diagnosis-v1", runs={}, new_macro_updates=0,
                  window="alive, command zero, time>=3s; same Hann normalization as fixed analyzer",
                  scope="Frequency diagnosis only; does not alter the frozen 3-95 Hz acceptance gate")
    for name in names:
        evidence = read(name)
        directory, report, analysis, data = evidence
        matches = paired(baseline, evidence)
        lanes, summary = [], {}
        for row in report["lanes"]:
            i = row["lane"]
            t = data["time"][:, i]
            selected = data["alive"][:, i] & (t >= 3.) & (np.abs(data["commands"][:, i, :2]).max(-1) < .01)
            dt = float(np.median(np.diff(t)))
            values = dict(lane=i, case=row["name"], observation_delay_ticks=row["first_reset_domain"]["sensor_delay_policy_ticks"])
            full = data["alive"][:, i] & ~data["passive"][:, i]
            position = data["local_position"][:, i, :2]
            relative = position - position[np.flatnonzero(full)[:1]]
            distance = np.linalg.norm(relative, axis=-1)
            at = int(np.where(full, distance, -1.).argmax())
            values.update(full_excursion_peak_m=float(distance[at]), full_excursion_peak_time_s=float(t[at]),
                          xy_at_peak_m=relative[at].tolist(),
                          filtered_wheel_position_before_peak_action_m=(data["observations"][at, i, 14:16] / 20.).tolist())
            for first, last in ((3, 8), (8, 20), (20, 95), (3, 95)):
                for quantity, array, columns, unit in (("leg", "motor_position", [0, 1, 3, 4], "rad"),
                                                        ("wheel", "motor_velocity", [2, 5], "rad_s")):
                    values[f"{quantity}_motion_rms_{unit}_{first}_{last}"] = hf_rms(
                        data[array][selected, i][:, columns], dt, first, last)
            lanes.append(values)
        for case in CASES:
            selected = [r for r in lanes if r["case"] == case]
            summary[case] = {key: float(np.mean([r[key] for r in selected]))
                             for key in selected[0] if "_motion_rms_" in key}
            worst = max(selected, key=lambda row: row["full_excursion_peak_m"])
            summary[case].update(worst_excursion_lane=worst["lane"],
                                 worst_excursion_m=worst["full_excursion_peak_m"],
                                 worst_excursion_time_s=worst["full_excursion_peak_time_s"])
            for key, value in (("leg_motion_rms_rad_3_95", "leg_motion_fast_rms_rad_3_95"),
                               ("wheel_motion_rms_rad_s_3_95", "wheel_motion_fast_rms_rad_s_3_95")):
                assert abs(summary[case][key] - analysis["cases"][case]["quiet"][value]) < 1e-10
        step = json.loads((directory.parent / "progress.json").read_text())["completed_updates"]
        writer = SummaryWriter(str(directory.parent), filename_suffix=".position_band058")
        for case, values in summary.items():
            for key, value in values.items():
                writer.add_scalar(f"Diagnosis/position_band058/{case}/{key}", value, step)
        writer.flush()
        writer.close()
        result["runs"][name] = dict(trajectory_sha256=report["trajectory_sha256"], pairing=matches,
                                     summary=summary, lanes=lanes, tensorboard_step=step)
    (REPORTS / "position_vibration058_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: row["summary"]["stand"] for name, row in result["runs"].items()}))


if __name__ == "__main__":
    main()

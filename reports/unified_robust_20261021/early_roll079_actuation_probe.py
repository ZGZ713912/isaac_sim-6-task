"""Read-only action/timing evidence for the failed same-batch077 roll test."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUN = ROOT / "runs/unified_robust_20261022/early_roll_push077"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration045_runner import write
from wheeled_algo.evaluation_early_roll import roll_gate


def main():
    torch.set_num_threads(1)
    mean, _ = load_mean(ROOT / "runs/unified_robust_20261022/quiet_severe_reference065/model_migrated.pt")
    result = dict(schema="roll079-failure-actuation-readonly-v2", variants={}, new_macro_updates=0,
        correction="Saved actions use control order; wheel target indices are2,5. The old v1 near-raw9 field used4,5 and is superseded; original JSON remains preserved.",
        wheel_control_indices=[2, 5],
        interpretation="Representative deterministic lanes; timestamps are logged after-step times for input-aligned observations; no claim of unique impact cause")
    for variant, full in (("robust", None), ("early_roll20_to_45", 45.), ("early_roll20_to_25", 25.)):
        directory = RUN / variant
        report = json.loads((directory / "report.json").read_text())
        path = directory / "trajectory.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report["trajectory_sha256"]
        data = np.load(path)
        rows = {}
        for case in ("moving_push_left", "moving_push_right"):
            row = next(r for r in report["lanes"] if r["name"] == case)
            lane = row["lane"]
            obs = torch.from_numpy(data["observations"][:, lane].copy())
            original, proposed = roll_gate(mean, obs, 20., full or 45.)
            gate = original.numpy().ravel() if full is None else proposed.numpy().ravel()
            time = data["time"][:, lane]
            selected = (time >= 5.) & (time <= row["termination_time_s"] + 1e-4)
            def first(mask):
                times = time[selected & mask]
                return float(times[0]) if len(times) else None
            window = selected & (time >= row["termination_time_s"] - .2)
            dq = data["motor_velocity"][:, lane]
            tau = data["motor_effort"][:, lane]
            pos = data["motor_position"][:, lane][:, [0, 1, 3, 4]]
            travel = np.arctan2(np.sin(pos - pos[np.where(window)[0][0]]),
                                np.cos(pos - pos[np.where(window)[0][0]]))
            rows[case] = dict(termination_time_s=row["termination_time_s"],
                first_authority_01_s=first(gate > .01), first_authority_09_s=first(gate >= .9),
                first_actual_fall_s=first(data["in_motion_fall"][:, lane]),
                preimpact_200ms_leg_speed_peak_rad_s=float(np.abs(dq[window][:, [0, 1, 3, 4]]).max()),
                preimpact_200ms_leg_travel_peak_rad=float(np.abs(travel[window]).max()),
                leg_torque_near_cap_fraction=float((np.abs(tau[selected][:, [0, 1, 3, 4]]) >= 39.9).mean()),
                wheel_torque_near_cap_fraction=float((np.abs(tau[selected][:, [2, 5]]) >= 4.49).mean()),
                wheel_target_near_raw9_fraction=float((np.abs(data["actions"][selected, lane][:, [2, 5]]) >= 8.99).mean()),
                peak_contact_impulse_ns=float(data["impulse"][selected, lane].max()),
                peak_gyro_x_rad_s=float(np.abs(data["omega"][selected, lane, 0]).max()),
                reference_resets_peak=int(data["standing_reference_severe_reset_count"][selected, lane].max()))
        result["variants"][variant] = dict(cases=rows, trajectory_sha256=report["trajectory_sha256"])
        writer = SummaryWriter(str(RUN), filename_suffix=".roll079_actuation")
        for case, row in rows.items():
            for key, value in row.items():
                if value is not None:
                    writer.add_scalar("Diagnosis/EarlyRoll079/corrected_control_order/" + variant + "/" + case + "/" + key, value, 505)
        writer.close()
    write(REPORTS / "early_roll079_actuation_corrected.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

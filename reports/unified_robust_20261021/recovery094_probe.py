"""Read-only stage attribution for sampled right-side recovery092."""
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration045_runner import write
from wheeled_algo.robust_getup import tilt_gate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_time(time, mask):
    ids = np.flatnonzero(mask)
    return float(time[ids[0]]) if len(ids) else None


def inspect(directory, model, sampled):
    report = json.loads((directory / "report.json").read_text())
    assert sha(directory / "trajectory.npz") == report["trajectory_sha256"]
    analysis = json.loads((directory / "analysis.json").read_text())
    data = np.load(directory / "trajectory.npz")
    rows = []
    for row in report["lanes"]:
        if row["name"] not in ("moving_push_left", "moving_push_right"):
            continue
        lane = row["lane"]
        obs = torch.from_numpy(data["observations"][:, lane].copy())
        time = data["time"][:, lane]
        alive = data["alive"][:, lane]
        with torch.inference_mode():
            gate = tilt_gate(obs, model.stance_aware_recovery,
                             model.recovery_shape_aware).numpy()[:, 0]
            mean = model(obs).numpy()
        action = data["actions"][:, lane][:, [0, 1, 3, 4, 2, 5]]
        physical_leg_difference = (action[:, :4] - mean[:, :4]) * .25
        physical_leg_difference = np.arctan2(np.sin(physical_leg_difference),
                                           np.cos(physical_leg_difference))
        push = data["pushing"][:, lane]
        push_time = first_time(time, push & alive)
        assert push_time is not None
        prepush = alive & (time >= push_time - .25) & (time < push_time)
        recovery = alive & (gate >= .5) & (time >= push_time)
        fall = data["in_motion_fall"][:, lane]
        i = np.flatnonzero(alive & push)[0]
        dq = data["motor_velocity"][:, lane][:, [0, 1, 3, 4]].reshape(-1, 2, 2)
        values = dict(lane=lane, case=row["name"], survived=row["survived"],
            practical=analysis["practical_followup_by_lane"][lane],
            actual_fall=row["in_motion_fall"], recovered_after_push=row["recovered_after_push"],
            termination_time_s=row["termination_time_s"], impulse_peak_ns=row["max_contact_impulse_ns"],
            push_start_s=push_time,
            fall_start_s=first_time(time, alive & fall),
            recovery_gate_half_s=first_time(time, recovery),
            recovery_gate_full_s=first_time(time, alive & (gate >= .999) & (time >= push_time)),
            recovered_after_push_s=first_time(time, alive & data["recovered_after_push"][:, lane]),
            prepush_vx_mean_m_s=float(data["velocity"][prepush, lane, 0].mean()),
            prepush_roll_speed_rms_rad_s=float(np.sqrt(np.mean(data["omega"][prepush, lane, 0] ** 2))),
            at_push_gravity=data["gravity"][i, lane].tolist(),
            at_push_velocity_m_s=data["velocity"][i, lane].tolist(),
            at_push_omega_rad_s=data["omega"][i, lane].tolist(),
            at_push_relative_leg_angle_rad=obs[i, 10:14].tolist(),
            recovery_frames=int(recovery.sum()))
        if recovery.any():
            values.update(
                recovery_common_speed_peak_rad_s=float(np.abs(dq[recovery].mean(-1)).max()),
                recovery_knee_half_speed_peak_rad_s=float(np.abs(np.diff(dq[recovery], axis=-1) * .5).max()),
                recovery_leg_sample_mean_distance_rms_rad=float(np.sqrt(np.mean(physical_leg_difference[recovery] ** 2))),
                recovery_wheel_sample_mean_distance_rms_rad_s=float(np.sqrt(np.mean(((action - mean)[recovery, 4:] * 10.) ** 2))))
        if sampled:
            # Recorded applied targets can be clipped/projected. These residuals
            # are not the original Gaussian draws or an inferred optimal action.
            values["residual_scope"] = "applied action versus mean on recorded preaction input; not Gaussian epsilon"
        rows.append(values)
    return dict(trajectory_sha256=report["trajectory_sha256"], rows=rows)


def main():
    torch.set_num_threads(1)
    checkpoint = RUNS / "pilot033_dynamic_recovery079/model_final.pt"
    assert sha(checkpoint) == "5fcc13518d9fdd1cb4a3745ef84b38892bffbaeb90719336f66f5a15e6091018"
    model, _ = load_mean(checkpoint)
    sources = {
        "sampled092": (RUNS / "recovery_sample_exposure092/robust", True),
        "mean088": (RUNS / "recovery_common_step088/robust", False),
    }
    results = {name: inspect(path, model, sampled) for name, (path, sampled) in sources.items()}
    successful = [r for r in results["sampled092"]["rows"]
                  if r["case"] == "moving_push_right" and r["practical"]]
    assert len(successful) == 1 and successful[0]["actual_fall"] and successful[0]["recovered_after_push"]
    assert successful[0]["recovery_gate_full_s"] < successful[0]["recovered_after_push_s"]
    writer = SummaryWriter(str(RUNS / "recovery_sample_exposure092"), filename_suffix=".recovery094")
    for name, source in results.items():
        for row in source["rows"]:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"Diagnosis/Recovery094/{name}/lane{row['lane']}/{key}", value, 569)
    writer.close()
    value = dict(schema="recovery094-sampled-stage-diagnosis-v1", status="completed",
        checkpoint_sha256=sha(checkpoint), sources=results,
        verified="One original-noise right-side trajectory actually falls, reaches full learned recovery authority, and returns to practical motion",
        limitations=["Prepush Gaussian draws also differ; postfall noise is not uniquely established as the cause",
            "Successful applied targets are correlated with the resulting state and are not a validated teacher",
            "No learning or mean-policy improvement is established;092cannot qualify a deployment"],
        new_macro_updates=0, robust_consumed=1384, full_training_admitted=False)
    write(REPORTS / "recovery094.json", value)
    print(json.dumps(dict(successful_right=successful, limits=value["limitations"])), flush=True)


if __name__ == "__main__":
    main()

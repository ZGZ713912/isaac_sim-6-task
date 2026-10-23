"""Report-only sensor sensitivity on a completed trajectory, retaining delay/bias.

Nominal lanes first authenticate alignment. Each counterfactual substitutes a
single sensor group's recorded clean state plus the same fixed bias/delay.
These action differences are not a new closed-loop rollout or deployment.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration036_compare import read, CASES


def main(run_name):
    torch.set_num_threads(1)
    directory, report, _, data = read(run_name)
    progress = json.loads((directory.parent / "progress.json").read_text())
    assert progress["status"] == "completed"
    process = json.loads((REPORTS / (run_name + "_process.json")).read_text())
    assert process["returncode"] == 0 and process["native_status"] == "completed"
    command = process["command"]
    checkpoint = Path(command[command.index("--checkpoint") + 1])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == report["checkpoint_sha256"]
    mean, _ = load_mean(checkpoint)
    obs = data["observations"][1:].copy()
    q = data["motor_position"][:-1, :, [0, 1, 3, 4]] - np.array([-.42, .13742282595395358, .42, -.1374155762580851])
    q = np.arctan2(np.sin(q), np.cos(q))
    sensors = np.concatenate((data["omega"][:-1]*.5, data["gravity"][:-1], q,
        np.zeros((*q.shape[:2], 2)), data["motor_velocity"][:-1, :, [0, 1, 3, 4, 2, 5]]*.1), -1)
    aligned = sensors.copy()
    selected = data["alive"][1:] & (data["time"][1:] >= 3.)
    nominal_errors = []
    for case in report["lanes"]:
        i = case["lane"]
        delay = case["first_reset_domain"]["sensor_delay_policy_ticks"]
        if delay:
            aligned[delay:, i] = sensors[:-delay, i]
        aligned[:, i] += np.array(case["first_reset_domain"]["sensor_bias"])
        aligned[:, i, 3:6] /= np.linalg.norm(aligned[:, i, 3:6], axis=-1, keepdims=True)
        if not case["domain_enabled"]:
            nominal_errors.append(float(np.abs(obs[:, i, 4:22][selected[:, i]] - aligned[:, i][selected[:, i]]).max()))
    assert max(nominal_errors) < 2e-6
    x = torch.from_numpy(obs.reshape(-1, 35))
    clean = torch.from_numpy(aligned.astype(np.float32).reshape(-1, 18))
    masks = {case: (selected & np.array([lane["name"] == case for lane in report["lanes"]])[None]).ravel()
             for case in CASES}
    with torch.inference_mode():
        actual = mean(x)
    groups = dict(gyro=(4, 7), gravity=(7, 10), leg_position=(10, 14), leg_velocity=(16, 20), wheel_velocity=(20, 22))
    results = {}
    for group, (start, end) in groups.items():
        alternate = x.clone()
        alternate[:, start:end] = clean[:, start-4:end-4]
        with torch.inference_mode():
            difference = (mean(alternate) - actual).numpy() * np.array([.25]*4 + [10.]*2)
        results[group] = {}
        sensor_difference = (alternate[:, start:end] - x[:, start:end]).numpy()
        for case, mask in masks.items():
            results[group][case] = dict(
                target_difference_leg_rms_rad=float(np.sqrt(np.mean(difference[mask, :4]**2))),
                target_difference_wheel_rms_rad_s=float(np.sqrt(np.mean(difference[mask, 4:]**2))),
                observed_sensor_difference_rms_scaled_units=float(np.sqrt(np.mean(sensor_difference[mask]**2))))
    result = dict(schema="standing055-sensor-action-sensitivity-v1", run=run_name,
        checkpoint_sha256=report["checkpoint_sha256"], trajectory_sha256=report["trajectory_sha256"],
        nominal_alignment_max_error=max(nominal_errors), nominal_lane_count=len(nominal_errors),
        per_group_single_substitution=results, native_runs=0, optimizer_updates=0,
        scope="All sensor biases, delays, commands and previous applied actions retained. Only random sensor errors removed one group at a time on recorded states. Correlated feedback, actual closed-loop response and spectral qualification do not follow; no randomization removed from any policy or training config.")
    (REPORTS / ("standing_noise055_"+run_name+".json")).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    with SummaryWriter(str(directory.parent), filename_suffix=".standing_noise055") as writer:
        for group, cases in results.items():
            for case, metrics in cases.items():
                for name, value in metrics.items():
                    writer.add_scalar(f"Diagnosis/standing_noise055/{group}/{case}/{name}", value, progress["completed_updates"])
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    main(parser.parse_args().run)

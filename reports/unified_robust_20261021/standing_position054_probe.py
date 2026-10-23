"""Offline feasibility of angle odometry; this is not an existing policy input.

Reads native continuous wheel angles, adds an assumed encoder noise model,
aligns them to the existing delayed IMU/leg observation, and measures relative
forward-displacement error on completed standing runs. No actor or RMCS edit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "src"), str(REPORTS)]
from iteration036_compare import read, CASES
from wheeled_algo.rolling_velocity import frozen_table_path, TABLE_SHA256


def coordinate(leg_q, wheel_q, gravity, table):
    # A planar forward coordinate; heading/large roll/slip need separate tests.
    pitch = np.arctan2(gravity[:, 0], -gravity[:, 2])
    result = .06 * pitch
    for i, side in enumerate(("left", "right")):
        t = table["sides"][side]
        knots, points = np.asarray(t["delta_rad"]), np.asarray(t["wheel_center_b_m"])
        axis, origin = np.asarray(t["hip_axis_b"]), np.asarray(t["hip_origin_b_m"])
        difference = leg_q[:, 2*i+1] - leg_q[:, 2*i]
        delta = np.arctan2(np.sin(difference), np.cos(difference))
        assert ((delta >= knots[0]) & (delta <= knots[-1])).all()
        point = np.stack([np.interp(delta, knots, points[:, j]) for j in range(3)], -1)
        beta = np.interp(delta, knots, t["passive_inner_deg"]) * np.pi / 180.
        angle = leg_q[:, 2*i] - t["hip_reference_rad"]
        value = point - origin
        center = (np.cos(angle[:, None]) * value + np.sin(angle[:, None]) * np.cross(axis, value)
            + (1. - np.cos(angle[:, None])) * axis * (axis * value).sum(-1, keepdims=True) + origin)
        center_x = np.cos(pitch) * center[:, 0] + np.sin(pitch) * center[:, 2]
        spin = leg_q[:, 2*i] + (beta if i == 0 else -beta) + wheel_q[:, i]
        result += .5 * (.06 * np.sign(axis[1]) * spin - center_x)
    return result


def main(run_name):
    directory, report, _, data = read(run_name)
    assert json.loads((directory.parent / "progress.json").read_text())["status"] == "completed"
    table = json.loads(frozen_table_path().read_text())
    rng = np.random.default_rng(54054)
    nominal = np.array([-.42, .13742282595395358, .42, -.1374155762580851])
    rows = []
    for case in report["lanes"]:
        if case["name"] not in CASES:
            continue
        i = case["lane"]
        obs = data["observations"][1:, i].astype(np.float64)
        # Existing sensors precede the recorded action; wheel angles use the
        # same preceding state and the same policy-delay tick.
        wheels = data["motor_position"][:-1, i][:, [2, 5]].astype(np.float64)
        delay = case["first_reset_domain"]["sensor_delay_policy_ticks"]
        assert delay in (0, 1)
        if delay:
            wheels = np.concatenate((wheels[:1], wheels[:-1]))
        # Hypothetical wheel encoder noise, not measured hardware calibration.
        noise = rng.normal(0., .001, wheels.shape) + rng.uniform(-.005, .005, (1, 2))
        estimate = coordinate(obs[:, 10:14] + nominal, wheels + noise, obs[:, 7:10], table)
        selected = data["alive"][1:, i] & (data["time"][1:, i] >= .2)
        actual = data["local_position"][:-1, i, 0][selected].astype(np.float64)
        actual -= actual[0]
        estimate = estimate[selected] - estimate[selected][0]
        error = estimate - actual
        rows.append(dict(lane=i, case=case["name"], domain_enabled=case["domain_enabled"],
            sensor_delay_ticks=delay, body_x_final_m=float(actual[-1]), odometry_final_m=float(estimate[-1]),
            final_error_m=float(error[-1]), max_abs_error_m=float(np.abs(error).max()),
            mae_m=float(np.abs(error).mean()), body_x_max_excursion_m=float(np.abs(actual).max())))
    summary = {}
    for case in CASES:
        group = [row for row in rows if row["case"] == case]
        summary[case] = dict(episodes=len(group), worst_abs_error_m=max(row["max_abs_error_m"] for row in group),
            worst_final_abs_error_m=max(abs(row["final_error_m"]) for row in group),
            mean_episode_mae_m=float(np.mean([row["mae_m"] for row in group])))
    result = dict(schema="standing054-position-observability-feasibility-v1", run=run_name,
        source_trajectory_sha256=report["trajectory_sha256"], geometry_table_sha256=TABLE_SHA256,
        hypothetical_wheel_noise_sigma_rad=.001, hypothetical_wheel_constant_bias_bound_rad=.005,
        reference_seconds=.2, summaries=summary, lanes=rows, native_runs=0, optimizer_updates=0,
        installed_in_actor=False, installed_in_rmcs=False,
        limitations="Current35D zeroes wheel angles. This diagnostic adds recorded continuous wheel-angle measurements with assumed noise; it proves neither deployed sensor availability nor closed-loop behavior. Flat standing planar forward coordinate only; heading, large roll, air, slip and recovery resets require separate design and testing. True position is used only as an offline error label.")
    target = REPORTS / ("standing_position054_" + run_name + ".json")
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    step = json.loads((directory.parent / "progress.json").read_text())["completed_updates"]
    with SummaryWriter(str(directory.parent), filename_suffix=".standing_position054") as writer:
        for case, metrics in summary.items():
            for key, value in metrics.items():
                writer.add_scalar(f"Diagnosis/standing_position054/{case}/{key}", value, step)
    print(json.dumps(result["summaries"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    main(parser.parse_args().run)

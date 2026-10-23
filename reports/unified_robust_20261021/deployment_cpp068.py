"""Offline C++/Torch observation parity; no RMCS or motor process is changed."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
CPP = REPORTS / "deployment_candidate068"
OUTPUT = ROOT / "runs/unified_robust_20261022/standing_reference_cpp068"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from iteration036_compare import read
from wheeled_algo.standing_displacement import WheelAngleCoordinate
from wheeled_algo.standing_reference_lifecycle import observer_from_settings, observation_contract, reference_settings
from wheeled_algo.robust_getup import quiet_gate, tilt_gate
from wheeled_algo.rolling_velocity import frozen_table_path, TABLE_SHA256, SPEC_SHA256


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal(value):
    value = format(float(value), ".9g")
    return value + ("" if "." in value or "e" in value else ".0") + "F"


def vector(values):
    return "{" + ", ".join(literal(value) for value in values) + "}"


def generate_table(estimator):
    lines = ["#pragma once", '#include "standing_reference_v3.hpp"',
        "// Generated from frozen model table SHA256 " + TABLE_SHA256,
        "namespace wheel_leg_policy {", "inline const GeometryTable& frozen_geometry107() {",
        "    static const GeometryTable table{{"]
    for side in range(2):
        get = lambda key: getattr(estimator.geometry, f"side{side}_{key}").tolist()
        lines.append("        SideGeometry{" + vector(get("hip_origin_b_m")) + ", " + vector(get("hip_axis_b"))
            + ", " + literal(get("hip_reference_rad")) + ", " + literal(get("wheel_radius_m")) + ", {")
        for delta, center, passive in zip(get("delta_rad"), get("wheel_center_b_m"), get("passive_inner_deg"), strict=True):
            lines.append("            GeometryKnot{" + literal(delta) + ", " + vector(center) + ", " + literal(passive) + "},")
        lines.append("        }},")
    lines += ["    }};", "    return table;", "}", "} // namespace wheel_leg_policy", ""]
    (CPP / "model_geometry107.hpp").write_text("\n".join(lines))


def main():
    torch.set_num_threads(1)
    assert sha(frozen_table_path()) == TABLE_SHA256
    estimator = WheelAngleCoordinate().eval()
    generate_table(estimator)
    nominal = estimator.geometry.nominal.numpy()
    positions, observations, ticks, resets, enabled = [], [], [], [], []
    sources = []
    for name, chosen in (("high_recenter_static066", [0, 1, 2, 27, 28, 29]),
                         ("high_recenter_full066", [0, 2, 4, 6, 8, 10, 12, 30])):
        _, report, _, data = read(name)
        sources.append(dict(run=name, trajectory_sha256=report["trajectory_sha256"], lanes=chosen))
        for lane in chosen:
            obs = data["observations"][:, lane].copy()
            p = data["motor_position"][:, lane][:, [0, 1, 3, 4, 2, 5]].copy()
            # The old traces did not retain raw delayed wheel angles. Use
            # recorded post-step wheel angles and the actual observed leg/IMU
            # frame as an offline encoder fixture, not as a hardware replay.
            p[:, :4] = obs[:, 10:14] + nominal
            for index in range(len(obs)):
                if not data["alive"][index, lane]:
                    break
                repeats = 2 if index % 31 == 0 else 1
                for repeat in range(repeats):
                    positions.append(p[index]); observations.append(obs[index]); ticks.append(index)
                    resets.append(index == 0 and repeat == 0)
                    enabled.append(not data["passive"][index, lane])
    for tick in range(440):
        obs = np.zeros(35, dtype=np.float32)
        obs[3], obs[9] = .305 * 5., -1.
        p = np.r_[nominal, [.002 * tick, -.002 * tick]].astype(np.float32)
        if 80 <= tick < 160:
            obs[9] = 1.; p[4:] = [tick * 3., -tick * 3.]
        elif 160 <= tick < 170:
            obs[9] = -.4
        elif 170 <= tick < 200:
            obs[4] = 1.
        elif 280 <= tick < 300:
            obs[0] = .7
        elif 320 <= tick < 330:
            obs[9] = float("nan")
        elif 330 <= tick < 340:
            p[0] = obs[10] = float("nan")
        elif tick >= 340:
            p[4:] = [1000., -1000.]
        for repeat in range(2 if tick % 17 == 0 else 1):
            positions.append(p.copy()); observations.append(obs.copy()); ticks.append(tick)
            resets.append(tick == 0 and repeat == 0); enabled.append(True)
    p = torch.from_numpy(np.asarray(positions))
    obs = torch.from_numpy(np.asarray(observations))
    coordinates, valid = estimator(p, obs[:, 7:10])
    gates = (quiet_gate(obs).ravel() >= .999) & (tilt_gate(obs, True, True).ravel() <= .05)
    settings = dict(quiet_velocity_feedback=True, quiet_velocity_feedback_mode="standing_displacement",
        standing_position_filter_seconds=.15, standing_position_reference=reference_settings(.05, True),
        standing_position_observation=observation_contract(.15, .05, True))
    observer = observer_from_settings(settings, 1, "cpu")
    expected, entry_tick = [], 0
    for i, tick in enumerate(ticks):
        if resets[i]:
            observer.reset(torch.tensor([0])); entry_tick = tick
        severe = observer.disarm_severe_tilt(obs[i:i + 1, 7:10])
        if severe[0]: entry_tick = tick
        requested = (obs[i:i + 1, [0, 2]].abs() < torch.tensor([.1, .2])).all(-1) & enabled[i]
        age = torch.tensor(tick - entry_tick) * .005
        eligible = gates[i:i + 1] & (age >= .05)
        output, _ = observer.update(coordinates[i:i + 1], valid[i:i + 1], requested, eligible, tick)
        expected.append([float(valid[i, 0]), *coordinates[i].tolist(), float(observer.active[0]),
            *observer.anchor[0].tolist(), *observer.filtered[0].tolist(), int(observer.severe_reset_count[0]),
            *output[0].tolist()])
    expected = np.asarray(expected)
    OUTPUT.mkdir(exist_ok=False)
    fixture = OUTPUT / "inputs.txt"
    with fixture.open("w") as stream:
        for i, tick in enumerate(ticks):
            values = [str(tick), str(int(resets[i])), str(int(enabled[i]))]
            values += [format(float(value), ".9g") for value in np.r_[p[i].numpy(), obs[i].numpy()]]
            stream.write(" ".join(values) + "\n")
    binary = OUTPUT / "parity_driver"
    command = ["g++", "-std=c++20", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
               str(CPP / "parity_driver.cpp"), "-o", str(binary)]
    subprocess.run(command, cwd=ROOT, check=True)
    actual_path = OUTPUT / "actual.txt"
    with fixture.open() as input_stream, actual_path.open("w") as output_stream:
        subprocess.run([str(binary)], stdin=input_stream, stdout=output_stream, check=True)
    actual = np.loadtxt(actual_path)
    assert actual.shape == expected.shape
    discrete_exact = np.array_equal(actual[:, [0, 3, 8]], expected[:, [0, 3, 8]])
    errors = np.max(np.abs(actual - expected), axis=0)
    assert discrete_exact, "Geometry validity/reference activation/reset count mismatch"
    assert max(errors[[1, 2, 4, 5]]) < 2e-5, errors
    assert max(errors[[6, 7, 9, 10]]) * 20. < 1e-4, errors
    np.savez_compressed(OUTPUT / "parity.npz", expected=expected, actual=actual)
    result = dict(schema="standing-reference068-offline-cpp-parity-v1", frames=len(ticks),
        sources=sources, geometry_table_sha256=TABLE_SHA256, model_spec_sha256=SPEC_SHA256,
        contract=settings, build_command=command, discrete_valid_active_count_exact=discrete_exact,
        maximum_coordinate_error_m=float(max(errors[1:3])),
        maximum_anchor_error_m=float(max(errors[4:6])),
        maximum_filtered_error_m=float(max(errors[6:8])), maximum_observation_slot_error=float(max(errors[9:11]) * 20.),
        files={str(path.relative_to(ROOT)): sha(path) for path in [fixture, actual_path, OUTPUT / "parity.npz",
            CPP / "standing_reference_v3.hpp", CPP / "model_geometry107.hpp", CPP / "parity_driver.cpp"]},
        installed_in_rmcs=False, hardware_tested=False, long_training_admitted=False,
        scope="Standalone C++ candidate; native-derived encoder fixtures and synthetic lifecycle/invalid/air-spin/duplicate-tick cases")
    (REPORTS / "deployment_cpp068.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("frames", "discrete_valid_active_count_exact",
        "maximum_coordinate_error_m", "maximum_observation_slot_error", "installed_in_rmcs")}), flush=True)


if __name__ == "__main__":
    main()

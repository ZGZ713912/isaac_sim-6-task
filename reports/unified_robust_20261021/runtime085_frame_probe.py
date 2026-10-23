"""Compare the integrated RMCS full35 frame helper with Torch on saved fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from wheeled_algo.standing_displacement import WheelAngleCoordinate
from wheeled_algo.standing_reference_lifecycle import observer_from_settings, observation_contract, reference_settings
from wheeled_algo.robust_getup import quiet_gate, tilt_gate
from wheeled_algo.paired_recovery import paired_offsets


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="")
    args = parser.parse_args()
    assert not args.revision or args.revision.replace("_", "").isalnum()
    suffix = "_" + args.revision if args.revision else ""
    torch.set_num_threads(1)
    source = ROOT / "runs/unified_robust_20261022/standing_reference_cpp068/inputs.txt"
    output = ROOT / ("runs/unified_robust_20261022/runtime085_frames" + suffix)
    output.mkdir(exist_ok=False)
    raw = np.loadtxt(source)
    ticks, resets, enabled = raw[:, 0].astype(int), raw[:, 1].astype(bool), raw[:, 2].astype(bool)
    q = torch.tensor(raw[:, 3:9], dtype=torch.float32)
    old = torch.tensor(raw[:, 9:], dtype=torch.float32)
    # Exercise complete turns as well as the original native-derived fixtures.
    q[::7, :4] += 4 * torch.pi
    velocity, gyro, gravity = old[:, 16:22] * 10., old[:, 4:7] * 2., old[:, 7:10]
    commands, height, applied = old[:, :3], old[:, 3] / 5., old[:, 22:28]
    nominal = WheelAngleCoordinate().geometry.nominal
    frame = torch.zeros(len(raw), 35)
    frame[:, :3], frame[:, 3] = commands, height * 5.
    frame[:, 4:7], frame[:, 7:10] = gyro * .5, gravity
    delta = q[:, :4] - nominal
    frame[:, 10:14] = torch.atan2(delta.sin(), delta.cos())
    frame[:, 16:22], frame[:, 22:28], frame[:, 28] = velocity * .1, applied, 1.
    position = q.clone()
    position[:, :4] = frame[:, 10:14] + nominal
    estimator = WheelAngleCoordinate().eval()
    coordinate, valid = estimator(position, gravity)
    finite = torch.isfinite(frame).all(-1) & torch.isfinite(q).all(-1)
    gates = (quiet_gate(frame).ravel() >= .999) & (tilt_gate(frame, True, True).ravel() <= .05)
    settings = dict(quiet_velocity_feedback=True, quiet_velocity_feedback_mode="standing_displacement",
        standing_position_filter_seconds=.15, standing_position_reference=reference_settings(.05, True),
        standing_position_observation=observation_contract(.15, .05, True))
    observer = observer_from_settings(settings, 1, "cpu")
    expected = np.full((len(raw), 41), np.nan)
    history = paired_offsets(q[:, :4] - nominal) / .25
    entry_tick, previous_enabled = 0, False
    fixture = output / "inputs.txt"
    with fixture.open("w") as stream:
        for i, tick in enumerate(ticks):
            reset = bool(resets[i] or enabled[i] and not previous_enabled)
            resets[i] = reset
            previous_enabled = bool(enabled[i])
            if reset:
                observer.reset(torch.tensor([0])); entry_tick = tick
            if finite[i]:
                severe = observer.disarm_severe_tilt(gravity[i:i+1])
                if severe[0]: entry_tick = tick
                requested = (frame[i:i+1, [0, 2]].abs() < torch.tensor([.1, .2])).all(-1) & bool(enabled[i])
                eligible = gates[i:i+1] & (torch.tensor(tick - entry_tick) * .005 >= .05)
                displacement, _ = observer.update(coordinate[i:i+1], valid[i:i+1], requested, eligible, tick)
                row = frame[i].clone()
                row[14:16] = displacement[0] * 20.
                expected[i] = torch.cat((row.clamp(-100., 100.), history[i], torch.zeros(2))).numpy()
            values = torch.cat((q[i], velocity[i], gyro[i], gravity[i], commands[i], height[i:i+1], applied[i])).numpy()
            stream.write(" ".join([str(tick), str(int(reset)), str(int(enabled[i]))]
                + [format(float(v), ".9g") for v in values]) + "\n")
    actual_path = output / "actual.txt"
    command = ["docker", "exec", "-i", "rmcs_devcontainer-rmcs-develop-1",
        "/workspaces/RMCS/.codex/worktrees/unified-standing-v3/rmcs_ws/build/rmcs_core/wheel_leg_observation_parity"]
    with fixture.open() as input_stream, actual_path.open("w") as stream:
        subprocess.run(command, stdin=input_stream, stdout=stream, check=True)
    lines = actual_path.read_text().splitlines()
    assert len(lines) == len(raw)
    actual = np.full_like(expected, np.nan)
    accepted = np.zeros(len(raw), bool)
    for i, line in enumerate(lines):
        values = np.fromstring(line, sep=" ")
        accepted[i] = values[0] == 1.
        if accepted[i]:
            assert len(values) == 42
            actual[i] = values[1:]
    assert np.array_equal(accepted, finite.numpy())
    errors = np.max(np.abs(actual[accepted] - expected[accepted]), axis=0)
    assert errors[:35].max() < 1e-4, errors
    assert errors[35:].max() < 1e-4, errors
    np.savez_compressed(output / "parity.npz", expected=expected, actual=actual, accepted=accepted)
    core = Path("/home/noir/Documents/workspace/RMCS/.codex/worktrees/unified-standing-v3/rmcs_ws/src/rmcs_core")
    result = dict(schema="runtime085-full-frame-parity-v1", frames=len(raw), valid_frames=int(accepted.sum()),
        input_fixture068_sha256=sha(source), invalid_rejection_exact=True,
        full35_maximum_error=float(errors[:35].max()), position_slot_maximum_error=float(errors[14:16].max()),
        passive_history_maximum_error=float(errors[35:].max()), multiple_common_turns_injected=True,
        command=command, scope="native-derived encoder fixtures plus synthetic reset/invalid/duplicate cases; raw delayed wheel angles were not saved in068, so its post-step angles remain an offline fixture",
        files={str(p): sha(p) for p in (fixture, actual_path, output / "parity.npz",
            core / "src/controller/chassis/wheel_leg/unified_observation.hpp",
            core / "src/controller/chassis/wheel_leg/standing_reference_v3.hpp",
            core / "src/controller/chassis/wheel_leg/standing_model_geometry.hpp")},
        hardware_tested=False, behavior_qualified=False, new_ppo_updates=0)
    (Path(__file__).parent / ("runtime085_frame" + suffix + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("frames", "valid_frames", "full35_maximum_error", "position_slot_maximum_error", "passive_history_maximum_error")}), flush=True)


if __name__ == "__main__":
    main()

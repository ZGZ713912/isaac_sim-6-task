"""New-wheel units, default launch binding and frozen historical compatibility."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wheeled_tasks.chassis.control_contract import load_chassis_control, validate_chassis_control
from wheeled_tasks.chassis.preflight import design_preflight, validate_runtime_state

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


IDENTIFICATION = module("wheel_ratio_identification", "scripts/identify_wheel_response.py")
FREEZE = module("wheel_ratio_freeze", "scripts/freeze_wheel_reduction.py")
TRAIN = module("wheel_ratio_train", "scripts/train_scratch_standing.py")


def inputs():
    config = json.loads(TRAIN.BASE_CONFIG.read_text())
    design = json.loads((ROOT / config["design_preflight"]["file"]).read_text())
    row = design["stages"]["speed"]["roles"]["train"]
    control = load_chassis_control(ROOT / config["control_math_source"])
    return config, control, row


def preflight(config, control, row):
    return design_preflight(config, control, ROOT, stage="speed", role="train",
        seed=row["invocation"]["seed"]["value"], num_envs=row["invocation"]["num_envs"])


def test_default_new_training_authenticates_15p62_control_and_design():
    assert TRAIN.BASE_CONFIG.name == FREEZE.MECHANISM
    config, control, row = inputs()
    assert control["actuators"]["wheel"]["gear_ratio"] == 15.62
    receipt = preflight(config, control, row)
    assert receipt["status"] == "static_passed"
    assert receipt["decimation"] == 5
    assert receipt["wheel_reduction"]["gear_ratio"] == 15.62
    assert receipt["wheel_reduction"]["new_wheel_response_identified"] is False
    assert receipt["wheel_reduction"]["legacy_response_gear_ratio"] == 15.8


def test_new_contracts_rebuild_exactly_without_touching_their_parents(tmp_path):
    receipt = FREEZE.build(output=tmp_path)
    for relative, expected in receipt["parent_files_sha256"].items():
        assert FREEZE.digest((ROOT / relative).read_bytes()) == expected
    for relative, expected in receipt["files_sha256"].items():
        name = Path(relative).name
        assert (tmp_path / name).read_bytes() == (ROOT / relative).read_bytes()
        assert FREEZE.digest((tmp_path / name).read_bytes()) == expected


@pytest.mark.parametrize("ratio", [15.8, 11., 19.2032])
def test_new_control_rejects_historical_or_stock_ratios(ratio):
    _, control, _ = inputs()
    control["actuators"]["wheel"]["gear_ratio"] = ratio
    with pytest.raises(ValueError, match="gear_ratio"):
        validate_chassis_control(control)


def test_new_design_rejects_legacy_control_and_missing_reduction():
    config, control, row = inputs()
    old = load_chassis_control(ROOT / "contracts/flat_base_200hz_control_v1.json")
    with pytest.raises(ValueError, match="gear_ratio"):
        preflight(config, old, row)
    del config["wheel_reduction"]
    with pytest.raises(ValueError, match="wheel_reduction"):
        preflight(config, control, row)


def test_runtime_checks_the_effective_ratio_instead_of_echoing_the_design():
    config, control, row = inputs()
    expected = preflight(config, control, row)["runtime_expected"]
    actual = deepcopy(expected)
    # This test uses the effective seed, just as the native generator reports it.
    if isinstance(actual["seed"], dict):
        actual["seed"] = row["invocation"]["seed"]["value"]
    assert validate_runtime_state(expected, actual)["status"] == "runtime_configuration_passed"
    actual["wheel_reduction"]["gear_ratio"] = 15.8
    with pytest.raises(ValueError, match="wheel_reduction.gear_ratio"):
        validate_runtime_state(expected, actual)


def test_output_cap_is_not_multiplied_by_the_new_reduction():
    import torch
    from wheeled_tasks.v40.core import motor_torque_limit
    _, control, _ = inputs()
    speed = torch.tensor([-90., -1., 0., 1., 90.])
    assert torch.equal(motor_torque_limit(speed, control["actuators"]["wheel"]),
                       torch.full_like(speed, 4.5))


@pytest.mark.parametrize("recorded_ratio", [15.8, 15.62])
def test_raw_motor_rpm_and_api_outputs_use_declared_ratio_once(tmp_path, recorded_ratio):
    run = tmp_path / "recording"
    (run / "fit").mkdir(parents=True)
    (run / "fit/wheel_fit.json").write_text("{}")
    (run / "trajectory-plan.json").write_text('{"segments": []}')
    (run / "wheel_quality.json").write_text('{"status": "synthetic", "reasons": []}')
    n = 3
    rpm = np.tile([-9372, 4686], (n, 1))
    raw = np.zeros((n, 48), dtype=np.uint8)
    encoded = rpm % 65536
    for side in range(2):
        raw[:, 32 + side*8 + 2] = encoded[:, side] // 256
        raw[:, 32 + side*8 + 3] = encoded[:, side] % 256
    speed = np.ones((n, 6))
    speed[:, 4:6] = -rpm / recorded_ratio * 2*np.pi/60
    torque = np.full((n, 6), .5)
    clock = np.arange(n, dtype=np.int64)*1_000_000
    np.savez(run / "wheel_1khz.npz", phase=np.full(n, 2), control_steady_ns=clock,
        segment_id=np.zeros(n), q_api=np.ones((n, 6)), dq_api=speed,
        tau_frame_api=torque, torque_fb_api=torque,
        wheel_velocity_target_api=speed[:, 4:6],
        feedback_steady_ns=np.tile(clock[:, None], (1, 6)), feedback_frame_bytes=raw)
    data, _, provenance = IDENTIFICATION.load_run(tmp_path, "recording", recorded_gear_ratio=recorded_ratio)
    assert data["speed_consistent"].all()
    np.testing.assert_allclose(data["dq_api"][:, 4:6], np.tile([20*np.pi, -10*np.pi], (n, 1)))
    np.testing.assert_array_equal(data["dq_api"][:, :4], speed[:, :4])
    np.testing.assert_allclose(data["dq_api"][:, 4:6]*data["tau_frame_api"][:, 4:6],
                               speed[:, 4:6]*torque[:, 4:6])
    assert provenance["wheel_reduction"]["gear_ratio"] == 15.62


def test_recorded_controller_gain_and_cap_follow_the_same_unit_change():
    scale = IDENTIFICATION.recording_scale(15.8, 15.62)
    r = scale["torque"]
    t = np.arange(301)*.001
    target = np.full_like(t, 10.)
    command = np.zeros_like(t)
    old = IDENTIFICATION.replay(t, target, command, 0., [.0045, .05, .07, .001], .6, True)
    new = IDENTIFICATION.replay(t, target/r, command, 0.,
        [.0045*r*r, .05*r, .07*r, .001*r*r], .6*scale["velocity_p"], True, effort_limit=4.5*r)
    np.testing.assert_allclose(new, old/r, atol=1e-12, rtol=1e-12)


def test_leg_static_prior_checks_both_controls_and_rejects_gain_changes(tmp_path):
    from wheeled_algo.standing_load_initialization import verify_wheel_reduction_only
    old_path = ROOT / "contracts/flat_base_200hz_control_v1.json"
    old = dict(control_math_source=str(old_path), control_math_sha256=FREEZE.digest(old_path.read_bytes()))
    config, control, _ = inputs()
    proof = verify_wheel_reduction_only(old, config, ROOT)
    assert proof["wheel_gear_ratio"] == 15.62
    control["actuators"]["leg"]["kp"] = 161.
    path = tmp_path / "changed_control.json"
    path.write_text(json.dumps(control))
    changed = dict(control_math_source=str(path), control_math_sha256=FREEZE.digest(path.read_bytes()))
    with pytest.raises(ValueError, match="gains, caps"):
        verify_wheel_reduction_only(old, changed, ROOT)
    changed["control_math_sha256"] = config["control_math_sha256"]
    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_wheel_reduction_only(old, changed, ROOT)


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf"), True])
def test_invalid_recording_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="finite and positive"):
        IDENTIFICATION.recording_scale(ratio, 15.62)

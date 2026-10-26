"""Protect physical time, motor ordering and failure denominators in diagnosis."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("scratch_feedback_analysis",
    Path(__file__).resolve().parents[1] / "scripts/analyze_scratch_feedback_trace.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
sensor_spec = importlib.util.spec_from_file_location("scratch_sensor_timing",
    Path(__file__).resolve().parents[1] / "scripts/check_scratch_sensor_timing.py")
sensor = importlib.util.module_from_spec(sensor_spec)
sensor_spec.loader.exec_module(sensor)


def traces():
    valid = np.ones((3, 1), dtype=bool)
    physical = dict(valid=valid, force=np.zeros((3, 5, 1, 2, 3)),
        gravity=np.tile([0., 0., -1.], (3, 1, 1)), position=np.zeros((3, 1, 3)))
    physical["force"][..., 2] = 10.
    feedback = {name: np.zeros((3, 1, size)) for name, size in
        (("observation_policy", 35), ("action_raw_policy", 6), ("action_applied_control", 6),
         ("omega", 3), ("motor_velocity_control", 6), ("motor_effort_control", 6), ("velocity_world", 3))}
    feedback.update(valid=valid.copy(), time_s=np.arange(1, 4)[:, None]*.005,
                    initial_position_local=np.array([[.1, 0., .305]]))
    cases = [dict(height_m=.305, pitch_deg=0., physical_ms=15, standing_pass=False)]
    return physical, feedback, cases


def test_first_ms_loss_and_policy_control_motor_order():
    physical, feedback, cases = traces()
    physical["force"][0, 0, 0, 0, 2] = 2.  # Exactly2 N is not support.
    feedback["action_raw_policy"][:] = [.1, .2, .3, .4, .5, .6]
    feedback["action_applied_control"][:] = [.1, .2, .5, .3, .4, .6]
    row = analysis.explain(physical, feedback, cases)["cases"][0]
    assert row["first_missing_support_s"] == .001
    assert row["raw_to_applied_maximum_difference"] == 0.
    assert row["wheel_target_peak_rad_s"] == 6.
    assert row["body_xy_max_excursion_m"] == .1
    assert row["observed_policy_frames"] == 3


def test_failed_prefix_keeps_its_physical_denominator_and_tilt_order():
    physical, feedback, cases = traces()
    physical["valid"][-1] = False
    feedback["valid"][-1] = False
    cases[0]["physical_ms"] = 10
    angle = np.radians(22.)
    physical["gravity"][1, 0] = [np.sin(angle), 0., -np.cos(angle)]
    feedback["omega"][:, 0, :2] = [1., 1.]
    row = analysis.explain(physical, feedback, cases)["cases"][0]
    assert row["observed_policy_frames"] == 2
    assert row["roll_pitch_gyro_rms_rad_s"] == pytest.approx(np.sqrt(2.))
    assert all(r["time_s"] == .01 for r in row["first_tilt_crossings"].values())
    cases[0]["physical_ms"] = 15
    with pytest.raises(ValueError, match="denominator"):
        analysis.explain(physical, feedback, cases)


@pytest.mark.parametrize("fault", ["mask", "prefix", "clock", "nonfinite"])
def test_unreliable_telemetry_is_rejected(fault):
    physical, feedback, _ = traces()
    if fault == "mask":
        feedback["valid"][1, 0] = False
    elif fault == "prefix":
        physical["valid"][1, 0] = feedback["valid"][1, 0] = False
    elif fault == "clock":
        feedback["time_s"][0, 0] = 0.
    else:
        feedback["omega"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        analysis.validate_traces(physical, feedback)


def reset_boundary_trace():
    pitch = np.radians(np.array([0., -.5, 1.]))
    pose = np.zeros((3, 7))
    pose[:, 2], pose[:, 4], pose[:, 6] = .305, np.sin(pitch/2), np.cos(pitch/2)
    gravity = np.stack((np.sin(pitch), np.zeros(3), -np.cos(pitch)), -1)
    raw = np.zeros((1, 3, 35))
    raw[0, :, 7:10] = gravity
    return dict(observation_policy=raw, valid=np.ones((1, 3), dtype=bool),
        initial_root_pose_local=pose, initial_gravity=gravity, initial_omega=np.zeros((3, 3)),
        initial_motor_position_control=np.zeros((3, 6)), initial_motor_velocity_control=np.zeros((3, 6)),
        initial_action_control=np.zeros((3, 6)))


def test_first_actor_input_is_checked_independently_of_the_second_frame():
    feedback = reset_boundary_trace()
    assert all(v["maximum_absolute"] < 1e-12 for v in sensor.initial_channels(feedback, np.zeros(4)).values())
    feedback["observation_policy"][0, :, 7:10] = [0., 0., -1.]
    result = sensor.initial_channels(feedback, np.zeros(4))
    assert result["gravity_from_native"]["maximum_absolute"] > .017
    assert result["gravity_from_root_quaternion"]["maximum_absolute"] > .017
    assert result["native_gravity_from_root_quaternion"]["maximum_absolute"] < 1e-12


def test_missing_reset_evidence_cannot_validate_first_actor_input():
    feedback = reset_boundary_trace()
    del feedback["initial_root_pose_local"]
    assert sensor.initial_channels(feedback, np.zeros(4)) is None


def test_invalid_reset_quaternion_is_rejected():
    feedback = reset_boundary_trace()
    feedback["initial_root_pose_local"][1, 6] = 2.
    with pytest.raises(ValueError, match="normalized"):
        sensor.initial_channels(feedback, np.zeros(4))

"""The slow candidate must bind real stage speeds and both wheel motor signs."""
from copy import deepcopy
import math

import pytest
import torch

from test_recovery_training_integration import rig
from wheeled_tasks.chassis.recovery_training import RecoveryTraining
from wheeled_tasks.chassis.slow_recovery import (
    recovery_settings, runtime_expectation, native_runtime_binding, validate_settings,
)


def test_named_slow_recipe_binds_native_fsm_and_wheel_controller(rig, tmp_path):
    env, _ = rig
    settings = recovery_settings(tmp_path, "profiles.json", "lookup.json")
    recovery = RecoveryTraining(env, tmp_path, settings)
    assert native_runtime_binding(recovery) == runtime_expectation(settings)
    assert recovery.fsm.config["sagittal_plant_before_orbit"] is True
    assert recovery.fsm.config["stand_speed_rad_s"] == .8
    recovery.fsm.config["stand_speed_rad_s"] = 4.
    with pytest.raises(ValueError, match="Native slow recovery"):
        native_runtime_binding(recovery)


@pytest.mark.parametrize("field", ["fold_speed_rad_s", "plant_dwell_seconds", "orbit_thrust_min_angle_rad"])
def test_slow_recipe_cannot_silently_fall_back_to_a_fast_legacy_stage(rig, tmp_path, field):
    settings = recovery_settings(tmp_path, "profiles.json", "lookup.json")
    del settings[field]
    with pytest.raises(ValueError, match="complete named"):
        validate_settings(settings)


def test_legacy_control_remains_an_explicit_comparison_candidate(rig, tmp_path):
    legacy = recovery_settings(tmp_path, "profiles.json", "lookup.json", slow=False)
    assert "script_schema" not in legacy and "orbit_assist_enabled" not in legacy["wheel_balance"]
    env, _ = rig
    recovery = RecoveryTraining(env, tmp_path, legacy)
    assert recovery.fsm.config["orbit_speed_rad_s"] == 4.
    assert recovery.fsm.config["stand_speed_rad_s"] == 4.
    assert not recovery.fsm.config["sagittal_plant_before_orbit"]


def test_candidate_wheel_gains_and_assist_are_frozen_together(rig, tmp_path):
    settings = recovery_settings(tmp_path, "profiles.json", "lookup.json")
    expected = runtime_expectation(settings)
    settings["wheel_balance"]["orbit_assist_enabled"] = False
    with pytest.raises(ValueError, match="complete named"):
        validate_settings(settings)
    assert expected["wheel_balance"]["orbit_assist_enabled"]


def test_supported_fk_height_cannot_place_the_shell_below_the_floor(rig, tmp_path):
    env, _ = rig
    recovery = RecoveryTraining(env, tmp_path, recovery_settings(tmp_path, "profiles.json", "lookup.json"))
    angle = math.radians(105.)
    gravity = torch.tensor([[math.sin(angle), 0., -math.cos(angle)]]).repeat(2, 1)
    possible = recovery.shell_support_possible(gravity, torch.tensor([.12, .30]))
    assert possible.tolist() == [False, True]
    assert recovery.shell_minimum_height[0] > .25
    gravity[:, 0] *= -1
    assert recovery.shell_support_possible(gravity, torch.tensor([.12, .30])).tolist() == [False, True]


def test_low_grip_script_bounds_wheel_effort_before_the_native_motor_limit(rig, tmp_path):
    from wheeled_tasks.chassis.recovery import RecoveryPhase
    env, _ = rig
    settings = recovery_settings(tmp_path, "profiles.json", "lookup.json")
    settings["wheel_balance"]["effort_limit_nm"] = .5  # The explicitly tested v4 option, not the selected recipe.
    recovery = RecoveryTraining(env, tmp_path, settings)
    recovery.reset(torch.arange(2))
    recovery.release_finished[:] = True
    recovery.fsm.phase[:] = RecoveryPhase.PREPARE
    recovery.fsm.script_targets[:, [2, 5]] = torch.tensor([100., -100.])
    q, dq = env.nominal[:, env.ids], torch.zeros(2, 6)
    env.v5.motor_efforts(q, dq, q[:, [0, 1, 3, 4]], dq[:, [2, 5]])
    effort = recovery.torque(q, dq, torch.ones(2, 6))
    torch.testing.assert_close(effort[0, [2, 5]], torch.tensor([.5, -.5]))
    torch.testing.assert_close(effort[1], torch.ones(6))

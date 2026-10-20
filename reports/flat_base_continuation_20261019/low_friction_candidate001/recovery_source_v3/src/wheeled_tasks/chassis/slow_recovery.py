"""Opt-in compact/rotate/stand recovery recipe for the imported flat actor.

This candidate is separate from the SHA-bound historical campaign contracts.
Both sagittal route signs share the same speed and sensor gates; native first
episode replays must establish their behavior before it is used for training.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

SCHEMA = "compact-sagittal-recovery-v3"
FSM_SETTINGS = dict(fold_speed_rad_s=1.5, prepare_speed_rad_s=1., stand_speed_rad_s=.8,
    thrust_speed_rad_s=1., orbit_speed_rad_s=1.6, capture_speed_rad_s=.8, side_speed_rad_s=1.5,
    sagittal_plant_before_orbit=True, plant_dwell_seconds=.12, plant_requires_support=False,
    thrust_preserve_orbit_winding=True,
    orbit_thrust_min_angle_rad=.35, orbit_thrust_tilt_max_rad=math.radians(115.))
BASE_WHEEL_SETTINGS = dict(enabled=True, pitch_kp_nm_rad=8., pitch_kd_nm_s_rad=1.5,
    wheel_damping_nm_s_rad=.2, prepare_tilt_limit_deg=55., capture_tilt_limit_deg=85.)
ASSIST_SETTINGS = dict(orbit_assist_enabled=True, orbit_assist_speed_rad_s=1.5,
    thrust_assist_speed_rad_s=.8, assist_tilt_limit_deg=120., assist_ramp_seconds=.25,
    assist_requires_support=True)
SHELL_SUPPORT_GUARD = dict(enabled=True, penetration_tolerance_m=.003)


def recovery_settings(root, profiles_file, lookup_file, *, slow=True):
    root = Path(root)
    settings = dict(enabled=True, reference_dt=.005, blend_seconds=.2, stable_seconds=1.,
        max_script_seconds=8., reroute_tilt_rad=math.radians(70.), reroute_support_lost_seconds=.15,
        max_reroutes=1, learning_mask="pure_rl_intervals_only", reset_mode="initial_state_only",
        controller="encoder_imu_recovery_fsm", action_history="effective_reference",
        wheel_balance=deepcopy(BASE_WHEEL_SETTINGS), side_speed_rad_s=1.5)
    for prefix, file in (("profiles", profiles_file), ("height_lookup", lookup_file)):
        path = Path(file)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Recovery geometry must use a repository-relative file")
        settings[prefix + "_file"] = str(path)
        settings[prefix + "_sha256"] = hashlib.sha256((root / path).read_bytes()).hexdigest()
    if slow:
        settings.update(script_schema=SCHEMA, shell_support_guard=deepcopy(SHELL_SUPPORT_GUARD), **FSM_SETTINGS)
        settings["wheel_balance"].update(ASSIST_SETTINGS)
    return settings


def validate_settings(settings):
    if (settings.get("script_schema") != SCHEMA
            or any(settings.get(key) != value for key, value in FSM_SETTINGS.items())
            or settings.get("shell_support_guard") != SHELL_SUPPORT_GUARD
            or settings.get("wheel_balance") != BASE_WHEEL_SETTINGS | ASSIST_SETTINGS):
        raise ValueError("Slow recovery requires the complete named compact/rotate/stand recipe")
    return settings


def runtime_expectation(settings):
    validate_settings(settings)
    return dict(schema=SCHEMA, fsm=deepcopy(FSM_SETTINGS),
        wheel_balance=deepcopy(settings["wheel_balance"]),
        shell_support_guard=deepcopy(settings["shell_support_guard"]),
        initial_state_writes_only=True, policy_learning="pure_rl_intervals_only")


def native_runtime_binding(recovery):
    expected = runtime_expectation(recovery.settings)
    actual = dict(expected, fsm={key: recovery.fsm.config[key] for key in FSM_SETTINGS},
        wheel_balance=dict(recovery.wheel_balance), shell_support_guard=dict(recovery.shell_support_guard))
    if actual != expected:
        raise ValueError("Native slow recovery stage speeds or wheel assistance differ from the design")
    return actual

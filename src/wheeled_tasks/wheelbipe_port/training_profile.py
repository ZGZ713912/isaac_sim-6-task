"""Explicit, versioned training changes; the legacy profile is unchanged."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

PROFILE_PATH = Path(__file__).resolve().parents[3]/"contracts/wheelbipe_foundation_repair_v1.json"
PROFILE_PATHS = {"foundation-v1": PROFILE_PATH,
                 "foundation-v2": PROFILE_PATH.with_name("wheelbipe_foundation_repair_v2.json"),
                 "foundation-jump-v1": PROFILE_PATH.with_name("wheelbipe_foundation_jump_v1.json")}


def specification(name="foundation-v1"):
    return json.loads(PROFILE_PATHS[name].read_text())


def profile_sha256(name="foundation-v1"):
    return hashlib.sha256(PROFILE_PATHS[name].read_bytes()).hexdigest()


def phase_for_update(completed_updates, spec=None):
    if not isinstance(completed_updates, int) or completed_updates < 0:
        raise ValueError("Completed updates must be a nonnegative integer")
    phases = (spec or specification())["phases"]
    return max((row for row in phases if row["start_update"] <= completed_updates),
               key=lambda row: row["start_update"])


def apply_phase(cfg, completed_updates, spec=None):
    if spec is None:
        spec = specification(getattr(cfg,"port_training_profile","foundation-v1"))
    phase = phase_for_update(completed_updates, spec)
    cfg.commands.ranges.lin_vel_x = (-phase["vx_max_m_s"], phase["vx_max_m_s"])
    cfg.commands.ranges.ang_vel_z = (-phase["yaw_max_rad_s"], phase["yaw_max_rad_s"])
    cfg.commands.rel_standing_envs = phase["standing_fraction"]
    cfg.height_range = list(phase["height_range_m"])
    cfg.leg_length_range = list(phase["reset_leg_length_m"])
    cfg.leg_angle_range = list(phase["reset_leg_angle_rad"])
    cfg.use_predefined_leg_random_start = phase["predefined_ground_start"]
    return phase


def configure_profile(cfg, name):
    if name not in ("legacy", *PROFILE_PATHS):
        raise ValueError("Unknown Wheelbipe training profile: "+name)
    cfg.port_training_profile = name
    cfg.port_gas_feedforward = True
    cfg.port_reward_motor_only = False
    cfg.port_review_updates = ()
    cfg.port_behavior_auto_exit_enabled = False
    if name == "legacy":
        return
    spec = specification(name)
    cfg.port_profile_sha256 = profile_sha256(name)
    cfg.port_gas_feedforward = spec["gas_cancellation_feedforward"]
    cfg.port_reward_motor_only = True
    # User 2026-11-03: behavior diagnostics must never interrupt PPO. Historical
    # v1 review metadata remains immutable but no longer schedules learner exits.
    cfg.rewards = deepcopy(cfg.rewards)
    cfg.rewards.update(spec["reward_overrides"])
    for key, value in spec["parameter_overrides"].items():
        setattr(cfg, key, value)
    cfg.commands = deepcopy(cfg.commands)
    # Early random heading targets otherwise saturate yaw even when ordinary
    # yaw sampling is narrow. Train explicit yaw commands in every new phase.
    cfg.commands.heading_command = False
    cfg.commands.rel_heading_envs = 0.
    for mode_name, mode in cfg.commands.special_modes.items():
        mode.iteration_start = spec["special_mode_starts"].get(mode_name, spec["total_updates"])
    cfg.use_leg_random_start = True
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.reset_base.params["velocity_range"] = {}
    apply_phase(cfg, 0, spec)
    if name == "foundation-jump-v1":
        from .mixed_curriculum import REWARD_NAMES, validate
        validate(spec)
        cfg.port_mixed_spec = spec
        cfg.rewards.update({key: 1. for key in REWARD_NAMES})


def electrical_reward_terms(motor, velocity, previous, before_previous):
    """Motor order is (left rear, front, wheel, right rear, front, wheel)."""
    legs = [0, 1, 3, 4]
    wheels = [2, 5]
    delta = motor-previous
    second = motor-2.*previous+before_previous
    return dict(joint_torque=motor.square().sum(-1),
        wheel_power=(motor[:,wheels]*velocity[:,wheels]).clamp_min(0.).sum(-1),
        torque_rate=delta.square().sum(-1),
        torque_smoothness_leg=second[:,legs].square().sum(-1),
        torque_smoothness_wheel=second[:,wheels].square().sum(-1))


def behavior_gate(receipt, spec=None):
    """Offline historical-v1 score only; never a learner continuation control."""
    gate = (spec or specification())["review_gate"]
    errors = []
    if receipt.get("status") != "complete" or receipt.get("ppo_updates") != 0:
        return dict(passed=False, errors=["Native inference did not complete without PPO"])
    arguments = receipt.get("arguments",{})
    if (arguments.get("seed")!=gate["seed"] or arguments.get("vehicle")!="port"
            or arguments.get("control")!="actor" or arguments.get("training_profile")!="foundation-v1"):
        errors.append("Inference identity/seed differs from the gate")
    cases = {row["case"]: row for row in receipt.get("cases", [])}
    for name in gate["cases"]:
        if name not in cases:
            errors.append("Missing case: "+name)
            continue
        row = cases[name]
        w = row["windows"]["tail_half"]
        if row["envs"] != gate["envs"] or row["duration_s"] != gate["seconds_per_case"]:
            errors.append(name+": population/duration differs from the gate")
        if row["resets"] > gate["maximum_resets_per_case"]:
            errors.append(name+": reset occurred")
        speed = w["vx_m_s"]["mean"]
        expected_command = {"stand_mid":[0.,0.,.305],"forward":[.5,0.,.305],"reverse":[-.5,0.,.305]}[name]
        if row["command"]!=expected_command:
            errors.append(name+": fixed command differs from the gate")
        if not all(math.isfinite(float(value)) for value in (speed,w["vx_m_s"]["abs_mean"],
                w["tilt_deg"]["mean"],w["height_error_m"]["abs_mean"])):
            errors.append(name+": nonfinite summary")
        stand_limit = gate["maximum_mean_abs_stand_speed_m_s"]
        if name == "stand_mid" and stand_limit is not None and w["vx_m_s"]["abs_mean"] > stand_limit:
            errors.append(name+": excessive residual speed")
        if name == "forward" and not gate["minimum_forward_mean_speed_m_s"] <= speed <= gate["maximum_forward_mean_speed_m_s"]:
            errors.append(name+": forward tracking outside review range")
        if name == "reverse" and not gate["minimum_reverse_mean_speed_m_s"] <= speed <= gate["maximum_reverse_mean_speed_m_s"]:
            errors.append(name+": reverse tracking outside review range")
        if w["tilt_deg"]["mean"] > gate["maximum_mean_tilt_degrees"]:
            errors.append(name+": excessive mean tilt")
        # The audit records E|height-command| directly, avoiding cancellation
        # between high and low lanes in a mean-height-only check.
        if w["height_error_m"]["abs_mean"] > gate["maximum_mean_absolute_height_error_m"]:
            errors.append(name+": excessive height error")
    return dict(passed=not errors, errors=errors, qualification=False, scope=gate["scope"])

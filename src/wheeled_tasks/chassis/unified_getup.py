"""Deployable command math and reset/acceptance semantics for unified recovery.

No simulator imports: the same units and predicates are used by native training,
held-out evaluation and the exported policy contract.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch


SCHEMA = "wheel-leg-unified-getup-v1"
FAMILIES = ("upright", "front", "back", "left", "right", "inverted", "compound")
DEFAULTS = dict(
    schema=SCHEMA, policy_hz=200, physics_hz=1000, actor_dim=35, action_dim=6,
    passive_min_seconds=.2, passive_max_seconds=1.2, passive_hold_seconds=.08,
    episode_seconds=12., stable_seconds=.5, followup_seconds=2.,
    recovery_timeout_seconds=9., leg_action_clip=13., wheel_action_clip=9.,
    leg_position_scale=.25, closure_gap_m=.003, impact_limit_n=3500.,
    impact_window_seconds=.02, impact_impulse_limit_ns=20.,
    push_start_seconds=5., push_duration_seconds=.18, push_force_n=80., push_torque_nm=12.,
    motion_push_fraction=.25,
    motion_fraction=.4, arc_fraction_within_motion=.5,
    lateral_acceleration_m_s2=2.06, wheel_radius_m=.06, wheel_track_m=.4373,
    wheel_speed_limit_rad_s=90., friction_start=[.5, .5], friction_final=[.2, .5],
    curriculum_updates=128, origin_updates=0,
    motor_control="six_axis_RL_after_zero_effort_settling",
    input_source="IMU_encoders_commands_previous_action",
    fallback_script=False, state_writes="episode_reset_only",
)


def settings(**overrides):
    result = deepcopy(DEFAULTS)
    result.update(overrides)
    validate_settings(result)
    return result


def validate_settings(cfg):
    if set(cfg) != set(DEFAULTS) or cfg["schema"] != SCHEMA:
        raise ValueError("Unknown unified recovery contract")
    for key in ("policy_hz", "physics_hz", "actor_dim", "action_dim", "leg_position_scale",
                "closure_gap_m", "motor_control", "input_source", "fallback_script", "state_writes"):
        if cfg[key] != DEFAULTS[key]:
            raise ValueError(f"Changed unified policy/control interface: {key}")
    for key in ("passive_min_seconds", "passive_max_seconds", "passive_hold_seconds", "episode_seconds",
                "stable_seconds", "followup_seconds", "recovery_timeout_seconds", "leg_action_clip",
                "wheel_action_clip", "impact_limit_n", "lateral_acceleration_m_s2", "wheel_radius_m",
                "wheel_track_m", "wheel_speed_limit_rad_s", "impact_window_seconds", "impact_impulse_limit_ns",
                "push_start_seconds", "push_duration_seconds", "push_force_n", "push_torque_nm"):
        if type(cfg[key]) not in (float, int) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Invalid positive setting: {key}")
    if not cfg["passive_min_seconds"] <= cfg["passive_max_seconds"] < cfg["recovery_timeout_seconds"] < cfg["episode_seconds"]:
        raise ValueError("Invalid passive/recovery/episode clocks")
    if abs(round(cfg["impact_window_seconds"] * cfg["physics_hz"]) - cfg["impact_window_seconds"] * cfg["physics_hz"]) > 1e-6:
        raise ValueError("Impact impulse window must contain complete physical ticks")
    for key in ("motion_fraction", "arc_fraction_within_motion", "motion_push_fraction"):
        if not 0 < cfg[key] < 1:
            raise ValueError("Motion rehearsal and mixed commands must both remain populated")
    for key in ("curriculum_updates", "origin_updates"):
        if type(cfg[key]) is not int or cfg[key] < (1 if key == "curriculum_updates" else 0):
            raise ValueError("Invalid curriculum update clock")
    for key in ("friction_start", "friction_final"):
        values = cfg[key]
        if len(values) != 2 or not 0 < values[0] <= values[1] <= 2:
            raise ValueError("Invalid contact-material range")


def project_commands(commands, cfg):
    """Keep yaw, reduce translation at the declared lateral envelope (RMCS)."""
    result = commands.clone()
    demand = result[:, 0].abs() + cfg["wheel_track_m"] * result[:, 1].abs() / 2
    scale = (cfg["wheel_radius_m"] * cfg["wheel_speed_limit_rad_s"] / demand.clamp_min(1e-9)).clamp_max(1.)
    result[:, :2] *= scale[:, None]
    limit = cfg["lateral_acceleration_m_s2"] / result[:, 1].abs().clamp_min(1e-9)
    result[:, 0] = result[:, 0].sign() * torch.minimum(result[:, 0].abs(), limit)
    return result


def classify_gravity(gravity):
    """Classify actual post-passive attitude, not the requested spawn label."""
    dominant = gravity.abs().argmax(-1)
    result = torch.full_like(dominant, 6)
    result = torch.where((dominant == 2) & (gravity[:, 2] < 0), 0, result)
    result = torch.where((dominant == 2) & (gravity[:, 2] > 0), 5, result)
    result = torch.where((dominant == 0) & (gravity[:, 0] > 0), 1, result)
    result = torch.where((dominant == 0) & (gravity[:, 0] < 0), 2, result)
    result = torch.where((dominant == 1) & (gravity[:, 1] > 0), 3, result)
    result = torch.where((dominant == 1) & (gravity[:, 1] < 0), 4, result)
    return torch.where(gravity.abs().amax(-1) < .8, 6, result)


def stable_standing(gravity, omega, height, contacts, dq):
    return ((gravity[:, 2] < -math.cos(math.radians(15))) & (omega[:, :2].norm(dim=-1) < .7)
            & (height > .19) & contacts.all(-1) & (dq[:, [0, 1, 3, 4]].abs().amax(-1) < 3.))


def joint_tracking_score(vx_error, yaw_error):
    """Both axes must track: a perfect spin cannot hide missing translation."""
    return torch.exp(-(vx_error / .5).square() - (yaw_error / .75).square())


def push_window(elapsed, selected, passive, finished, cfg):
    """External perturbation; it never changes position, velocity or motor mode."""
    return (selected & ~passive & ~finished & (elapsed >= cfg["push_start_seconds"])
            & (elapsed < cfg["push_start_seconds"] + cfg["push_duration_seconds"]))

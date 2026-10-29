"""Independent upright-task semantics; no simulator or recovery controller."""
from copy import deepcopy
import math

import torch


SCHEMA = "wheel-leg-upright-locomotion-v1"


def resolve(spec, mode="balance", friction_profile="baseline"):
    if (spec.get("schema") != SCHEMA or spec.get("reset_height_m") != .305
            or spec.get("recovery_training") is not False
            or spec.get("automatic_task_or_height_curriculum") is not False
            or mode not in spec["modes"] or friction_profile not in spec["friction_profiles"]):
        raise ValueError("Need an explicit independent 305mm upright task")
    result = deepcopy(spec)
    result.update(mode=mode, friction_profile=friction_profile)
    material = spec["friction_profiles"][friction_profile]
    if any(not math.isfinite(v) or v <= 0. for v in material.values()):
        raise ValueError("Upright material coefficients must be positive and finite")
    result["effective_wheel_ground_friction"] = (material["wheel"] + material["ground"]) / 2.
    return result


def training_case(spec, lane, pose_index):
    height = spec["reset_height_m"]
    case = dict(pose_index=pose_index, recovery=False, command=[0., 0., height],
                pitch_deg=0., role="balance")
    if spec["mode"] == "basic_motion":
        # Forty percent stationary rehearsal remains explicit. No learning
        # clock can silently replace it with recovery, spin or other heights.
        vx = spec["basic_forward_speed_m_s"]
        yaw = spec["basic_yaw_rate_rad_s"]
        role, target = (("forward", [vx, 0., height]) if lane % 10 in (4, 5) else
                        ("reverse", [-vx, 0., height]) if lane % 10 in (6, 7) else
                        ("left", [0., yaw, height]) if lane % 10 == 8 else
                        ("right", [0., -yaw, height]) if lane % 10 == 9 else
                        ("balance", [0., 0., height]))
        case["role"] = role
        if role != "balance":
            case["sequence"] = [[spec["command_start_seconds"], target],
                                [spec["command_stop_seconds"], [0., 0., height]]]
    return case


def failures(gravity, height, target_height, nonwheel_force_peak, spec):
    limits = spec["termination"]
    return dict(
        tilt=gravity[:, 2] > -math.cos(math.radians(limits["tilt_deg"])),
        height=(height - target_height).abs() > limits["height_error_m"],
        nonwheel_contact=nonwheel_force_peak > limits["nonwheel_contact_n"])


def reward_terms(gravity, height, velocity, omega, command, torque, action_delta,
                 leg_speed_square):
    """Dense additive tracking with no potential, completion or recovery bonus.

    A velocity error still has a gradient when a narrow standing Gaussian
    would vanish. There is no penalty towards fixed hip/knee positions.
    Leg-speed cost uses all five actual 1ms samples supplied by the native task.
    """
    vx_error, yaw_error = velocity[:, 0] - command[:, 0], omega[:, 2] - command[:, 1]
    return dict(
        upright=2. * (-gravity[:, 2]).clamp(0., 1.),
        height=1.5 * torch.exp(-((height - command[:, 2]) / .02).square()),
        forward_tracking=2. * torch.exp(-(vx_error / .25).square()),
        yaw_tracking=torch.exp(-(yaw_error / .5).square()),
        tilt_cost=-2. * gravity[:, :2].square().sum(-1) / math.sin(math.radians(15.))**2,
        velocity_error_cost=-(vx_error / .5).square() - .5 * (velocity[:, 1] / .2).square(),
        angular_cost=-.05 * omega[:, :2].square().sum(-1),
        effort_cost=-.0001 * torque.square().mean(-1),
        action_rate_cost=-.002 * action_delta,
        leg_speed_cost=-.001 * leg_speed_square)

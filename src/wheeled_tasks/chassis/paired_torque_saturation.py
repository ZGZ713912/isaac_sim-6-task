"""Explicit task-level leg effort allocation, after PD plus spring holding.

The authenticated spring model and absolute position targets are unchanged.
Old plans retain independent motor clipping. At saturation this mode reserves
the requested differential effort before allocating the common rotation effort.
"""
from copy import deepcopy

import torch


CONTRACT = dict(schema="wheel-leg-shape-priority-torque-saturation-v1",
    mode="prioritize_leg_shape", total_motor_limit_nm=40.,
    sum_before_limit="unclipped_PD_plus_encoder_spring_holding",
    relative_limit_projection="outward_relative_torque_v1",
    target_semantics="unchanged_absolute_encoder_position",
    hardware_equivalence_verified=False)


def validate(contract):
    if contract != CONTRACT:
        raise ValueError("Use the explicit paired torque saturation contract")
    return deepcopy(contract)


def allocate(requested, encoder, limits):
    pair = requested.unflatten(-1, (2, 2))
    differential = ((pair[..., 0]-pair[..., 1])/2.).clamp(-40., 40.)
    common_capacity = 40.-differential.abs()
    common = pair.mean(-1).clamp(-common_capacity, common_capacity)
    limited = torch.stack((common+differential, common-differential), -1)
    limited = torch.where((pair.abs() <= 40.).all(-1, keepdim=True), pair, limited)
    limited = torch.where(torch.isfinite(pair).all(-1, keepdim=True), limited, float("nan"))
    coefficients = encoder.new_tensor([[-1., 1.], [1., -1.]])
    relative = (encoder.unflatten(-1, (2, 2))*coefficients).sum(-1)
    outward = (limited*coefficients).sum(-1)
    blocked = (((relative >= limits["maximum"]-limits["margin"]) & (outward > 0.))
        | ((relative <= limits["minimum"]+limits["margin"]) & (outward < 0.)))
    projected = limited-coefficients*torch.where(blocked, outward/2., 0.).unsqueeze(-1)
    return limited.flatten(-2), projected.flatten(-2)


def install(env, contract):
    validate(contract)
    if hasattr(env, "paired_torque_saturation"):
        raise RuntimeError("Paired torque saturation already installed")
    if env.plan.get("gas_spring_backend") != "dev_compensation_v2":
        raise ValueError("Paired torque saturation requires encoder holding and relative projection")
    feedforward = env.spring_feedforward
    original = feedforward.motor_efforts

    def motor_efforts(pd_effort, encoder_position):
        # Evaluate the original authenticated encoder-only force model once.
        # Its clamp is replaced; its force/geometry diagnostics remain intact.
        original(pd_effort, encoder_position)
        limited, projected = allocate(feedforward.last["requested_effort_nm"], encoder_position,
            feedforward.cfg["relative_soft_limits"])
        feedforward.last.update(limited_effort_nm=limited, projected_effort_nm=projected.clone())
        return projected

    feedforward.motor_efforts = motor_efforts
    env.paired_torque_saturation = deepcopy(contract)
    env.startup_report["paired_torque_saturation"] = deepcopy(contract)
    env.startup_report["encoder_spring_feedforward"]["formula"] = (
        "PD + dev fitted-load holding -> leg-shape priority within40Nm -> outward relative-torque projection")

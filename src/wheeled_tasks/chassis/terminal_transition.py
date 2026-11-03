"""Settle existing hard-fault terminals without learning invalid poststates.

No fault detector or termination is added; closure errors are not inputs.
Ordinary transitions retain their original dense reward. Mechanical/impact
terminals pay the original failure cost instead of costs of a failed poststate.
"""
from copy import deepcopy

import torch


CONTRACT = dict(schema="wheel-leg-hard-fault-terminal-v1",
    mode="existing_hard_fault_failure_cost_only",
    existing_faults=["mechanical", "body_impulse", "part_impact"],
    closure_error_is_not_a_fault=True,
    normal_dense_reward="unchanged",
    failure_cost="existing_role_failure_cost",
    physics_and_termination="unchanged")


def validate(config):
    if config != CONTRACT:
        raise ValueError("Use the complete versioned hard-fault terminal contract")
    return deepcopy(config)


def settle_reward(reward, failure_cost, mechanical, impact_bad, part_impact_bad=None):
    """Keep the pre-action sample and original failure cost for existing faults."""
    masks = (mechanical, impact_bad) if part_impact_bad is None else (
        mechanical, impact_bad, part_impact_bad)
    for mask in masks:
        if mask.shape != reward.shape or mask.device != reward.device or mask.dtype != torch.bool:
            raise ValueError("Hard-fault terminal masks must match the per-lane reward")
    hard = mechanical | impact_bad
    if part_impact_bad is not None:
        hard = hard | part_impact_bad
    cost = torch.as_tensor(failure_cost, dtype=reward.dtype, device=reward.device)
    return torch.where(hard, -cost, reward), hard

"""Explicit opt-in exploration envelope for a fresh recovery actor."""
from copy import deepcopy
import math


CONTRACT = dict(schema="wheel-leg-recovery-exploration-v1", std_range=[.003, 1.],
    initial_std=[.8, .8, .8, .8, .5, .5], coordinate="raw_policy_action",
    leg_target_std_rad=.2, wheel_target_std_rad_s=5.,
    physical_torque_and_joint_limits="unchanged", teacher_or_checkpoint=False)

INCREMENT_CONTRACT = dict(schema="wheel-leg-recovery-exploration-v2", std_range=[.003, 1.],
    initial_std=[.8, .8, .8, .8, .5, .5], coordinate="raw_common_relative_and_wheel_action",
    untruncated_common_rate_std_rad_s=1.6, untruncated_relative_rate_std_rad_s=1.6,
    leg_raw_clip=1., wheel_target_std_rad_s=5.,
    physical_torque_and_joint_limits="unchanged", teacher_or_checkpoint=False)


def std_bounds(plan):
    config = plan.get("exploration_distribution")
    if config is not None and config not in (CONTRACT, INCREMENT_CONTRACT):
        raise ValueError("Use the complete explicit recovery exploration contract")
    bounds = tuple(config["std_range"]) if config is not None else (.003, .3)
    values = plan["initial_std"]
    if (len(values) != 6 or any(not math.isfinite(v) or not bounds[0] <= v <= bounds[1] for v in values)
            or config is not None and values != config["initial_std"]):
        raise ValueError("Initial recovery exploration exceeds its declared distribution range")
    return bounds


def contract():
    return deepcopy(CONTRACT)

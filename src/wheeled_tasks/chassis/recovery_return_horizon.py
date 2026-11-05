"""Explicit physical-time return and TD-residual decay for recovery practice."""
from copy import deepcopy
import math


def contract():
    return dict(schema="wheel-leg-recovery-return-horizon-v1",
        discount_time_constant_s=10., gae_residual_time_constant_s=1.,
        scope="recovery_focus_training_with_original_auxiliary_roles",
        policy_dt_s=.01, rollout_steps=64,
        sample_budget="unchanged", potential_discount="same_as_learner_gamma",
        termination="unchanged", admission="unchanged")


def validate(config):
    expected = contract()
    if (type(config) is not dict or config != expected
            or any(type(config.get(k)) not in (int, float) for k in (
                "discount_time_constant_s", "gae_residual_time_constant_s", "policy_dt_s"))
            or type(config.get("rollout_steps")) is not int):
        raise ValueError("Use the complete recovery return horizon contract")
    return deepcopy(config)


def validate_plan(plan):
    from .control_contract import FINE_PD200_TIMING
    validate(plan["recovery_return_horizon"])
    if (plan.get("schema") != "wheel-leg-prone-motion-v4"
            or plan.get("global_curriculum", {}).get("recovery_focus") is None
            or not plan.get("multi_pose_recovery") or "jump_training" in plan
            or plan.get("rollout_steps") != 64
            or plan.get("timing") != FINE_PD200_TIMING):
        raise ValueError("Return horizon requires the explicit100Hz recovery-focus course with fine physics and200Hz PD")


def coefficients(config, policy_dt):
    """GAE's residual kernel decays as (gamma*lambda)**k, not lambda**k."""
    config = validate(config)
    if type(policy_dt) not in (float, int) or not math.isfinite(policy_dt) or policy_dt <= 0:
        raise ValueError("Return coefficients require a finite positive policy period")
    discount_rate = 1./config["discount_time_constant_s"]
    trace_rate = 1./config["gae_residual_time_constant_s"]
    return dict(gamma=math.exp(-policy_dt*discount_rate),
                lam=math.exp(-policy_dt*(trace_rate-discount_rate)))

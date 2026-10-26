"""Training reward from the native, unchanged continuous followup timer.

This adapter reads the environment's existing timer before autoreset. It never
changes the timer, success predicates, actor inputs, actions or transitions.
"""
import hashlib
import math
from pathlib import Path

import torch

MODULE = "src/wheeled_tasks/chassis/followup_continuity_reward.py"
SCHEMA = "native-followup-continuity-reward-v1"


def contract(weight, *, policy_dt=.005, followup_seconds=2.):
    if (isinstance(weight, bool) or not isinstance(weight, (float, int))
            or not math.isfinite(weight) or not 0. <= weight <= 5.
            or policy_dt != .005 or followup_seconds != 2.):
        raise ValueError("Require bounded0..5 density at native200Hz and unchanged2s followup")
    return dict(schema=SCHEMA, module=MODULE,
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        weight_density_per_second=float(weight), policy_dt=policy_dt,
        followup_seconds=followup_seconds,
        formula="weight*dt*clamp(native_followup_hold/2,0,1)^2*policy_learning_mask",
        scope="training_reward_only; native success timer, physics and policy function unchanged")


def validate_contract(value, sources):
    if not isinstance(value, dict):
        raise ValueError("Missing explicit training reward contract")
    expected = contract(value.get("weight_density_per_second"),
        policy_dt=value.get("policy_dt"), followup_seconds=value.get("followup_seconds"))
    if value != expected or sources.get(MODULE) != expected["module_sha256"]:
        raise ValueError("Training reward specification or source differs")


def bonus_from_hold(hold, active, *, weight, policy_dt=.005, followup_seconds=2.):
    if (hold.shape != active.shape or active.dtype != torch.bool
            or hold.device != active.device or not torch.is_floating_point(hold)
            or not torch.isfinite(hold).all() or (hold < 0.).any()):
        raise ValueError("Require finite nonnegative native timer and matching interval mask")
    # Weight validation also rejects NaN, Inf and altered units.
    if (not isinstance(weight, (int, float)) or isinstance(weight, bool)
            or not math.isfinite(weight) or not 0. <= weight <= 5.
            or policy_dt != .005 or followup_seconds != 2.):
        raise ValueError("Invalid continuity reward units or weight")
    return weight * policy_dt * (hold / followup_seconds).clamp(0., 1.).square() * active


def install(env, weight):
    """Snapshot only timers erased by native autoreset, then add the bonus."""
    if hasattr(env, "training_reward_contract") or env.fixed_cases is not None:
        raise ValueError("Attach once to a native training environment")
    spec = contract(weight, policy_dt=env.policy_dt,
                    followup_seconds=env.getup_cfg["followup_seconds"])
    native_step, native_reset = env.step, env.reset
    inside_step = False
    saved = {}

    def reset(ids, *args, **kwargs):
        if inside_step:
            saved["ids"] = ids.clone()
            saved["hold"] = env.followup_hold[ids].clone()
            saved["moving_fall"] = env.in_motion_fall[ids].clone()
        return native_reset(ids, *args, **kwargs)

    def step(actions):
        nonlocal inside_step
        if inside_step:
            raise RuntimeError("Recursive native step")
        saved.clear()
        inside_step = True
        try:
            obs, reward, done, extras = native_step(actions)
        finally:
            inside_step = False
        hold, moving_fall = env.followup_hold.clone(), env.in_motion_fall.clone()
        if saved:
            ids = saved["ids"]
            hold[ids], moving_fall[ids] = saved["hold"], saved["moving_fall"]
        active = extras["policy_learning_mask"]
        bonus = bonus_from_hold(hold, active, weight=weight, policy_dt=env.policy_dt,
                                followup_seconds=env.getup_cfg["followup_seconds"])
        env.training_reward_last = dict(native_reward=reward.detach().clone(),
            bonus=bonus.detach().clone(), native_followup_hold=hold,
            in_motion_fall=moving_fall)
        extras["log"].update({
            "FollowupReward/bonus_per_policy_step": bonus.mean(),
            "FollowupReward/paid_density_per_second": bonus.mean() / env.policy_dt,
            "FollowupReward/native_hold_mean_seconds": hold.mean(),
            "FollowupReward/two_second_hold_fraction": (hold >= 2.).float().mean(),
            "FollowupReward/moving_fall_bonus_per_policy_step": (bonus * moving_fall).mean(),
            "FollowupReward/weight_density_per_second": reward.new_tensor(weight)})
        return obs, reward if weight == 0. else reward + bonus, done, extras

    env.reset, env.step = reset, step
    env.training_reward_contract = spec
    return spec

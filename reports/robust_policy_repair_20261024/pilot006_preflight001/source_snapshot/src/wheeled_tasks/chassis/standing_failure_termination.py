"""End failed nominal standing training episodes, preserving fixed evaluation."""
import math
import torch


def install(env, *, tilt_deg=20., height_error_m=.025, cost=3.):
    if hasattr(env, "standing_failure_termination") or not env.cfg.get("record_diagnostics"):
        raise ValueError("Attach once after the precise standing objective")
    if env.getup_cfg["motion_fraction"] != 1. or env.getup_cfg["domains_enabled"]:
        raise ValueError("This admission profile is limited to nominal standing foundation training")
    native_step = env.step

    def step(actions):
        observation, reward, done, extras = native_step(actions)
        state = extras["diagnostics"]
        tilted = state["gravity"][:, 2] > -math.cos(math.radians(tilt_deg))
        wrong_height = (state["height"] - state["commands"][:, 2]).abs() > height_error_m
        requested = (state["requested_command"][:, :2].abs() < .01).all(-1)
        failure = (tilted | wrong_height) & requested & extras["policy_learning_mask"] & ~done
        # Held-out trajectories keep the entire native horizon and every raw
        # contact sample. Never improve an evaluation by truncating a failure.
        if env.fixed_cases is not None:
            return observation, reward, done, extras
        reward = reward - cost * failure
        done = done | failure
        extras["time_outs"][failure] = False
        extras["policy_bootstrap_mask"][failure] = False
        state["terminated"] = state["terminated"] | failure
        state["done"] = done.clone()
        for tag in ("UnifiedGetup/episode_events", "UnifiedGetup/terminated_events"):
            extras["log"][tag] = extras["log"].get(tag, reward.new_zeros(())) + failure.float().sum()
        extras["log"].update({
            "StandingFailure/episode_events": failure.float().sum(),
            "StandingFailure/tilt_events": (failure & tilted).float().sum(),
            "StandingFailure/height_events": (failure & wrong_height).float().sum()})
        if failure.any():
            env.reset(failure.nonzero().flatten())
            env.history.last_tick = None
            observation = env.get_observations()
        return observation, reward, done, extras

    env.step = step
    env.standing_failure_termination = dict(version="standing-foundation-failure-v1", max_tilt_deg=tilt_deg,
        max_height_error_m=height_error_m, additional_failure_cost=cost,
        state_writes="episode reset only", applied_to_fixed_evaluation=False,
        mechanical_guards_unchanged=True, contact_acceptance_unchanged=True)
    return env.standing_failure_termination

"""The verified009 standing recipe, with explicitly separate command exposure."""
from copy import deepcopy
import math

import torch

from .precise_standing_reward import scores


def validate(plan):
    if (plan.get("schema") != "wheel-leg-009-reproduction-curriculum-v1"
            or plan.get("reference_commit") != "643a0de0d61f049cd0ffe2007d52dd85fffbf6e6"
            or (plan["policy_hz"], plan["physics_hz"], plan["num_envs"], plan["rollout_steps"]) != (200, 1000, 256, 128)
            or plan["standing_recipe"] != dict(precise_objective=True, normalize_observations=True,
                desync_episodes=True, standing_failures=True, wheel_velocity_increment=True,
                separate_leg_velocity_feedback=True, static_load_initialization=True,
                contact_break_cost=0., speed_weight=2., contact_weight=8., learning_rate=.0003)):
        raise ValueError("Need the complete declared009 standing recipe")
    phases = plan["phases"]
    if ([p["id"] for p in phases] != ["standing", "tracking", "domains"]
            or phases[0]["min_updates"] < 512 or sum(p["max_updates"] for p in phases) != plan["max_updates"]
            or any(not 0 < p["min_updates"] <= p["max_updates"] for p in phases)):
        raise ValueError("Need bounded foundation, tracking and domain phases")
    if (plan["tracking"]["heights_m"] != [.23, .305, .42]
            or plan["tracking"]["roles_per20"].count("stand") != 8
            or plan["domain_sequence"][-1] != dict(until_phase_update=1024, profile="requested", strength=1.)):
        raise ValueError("Keep standing rehearsal and the final requested domains")
    return deepcopy(plan)


def domain_stage(plan, phase_updates):
    for stage in plan["domain_sequence"]:
        if phase_updates < stage["until_phase_update"]:
            return stage["profile"], stage["strength"]
    return "requested", 1.


def case(plan, role, birth_height=.305, pitch=0., progress=1.):
    cfg = plan["tracking"]
    progress = min(1., max(0., progress))
    amplitude = .2 + .8*progress
    vx = cfg["forward_speed_m_s"]*amplitude
    yaw = cfg["yaw_rate_rad_s"]*amplitude
    command = [0., 0., birth_height]
    target = list(command)
    target[0] = vx if role in ("forward", "forward_height") else -vx if role in ("reverse", "reverse_height") else 0.
    target[1] = yaw if role == "left" else -yaw if role == "right" else 0.
    if role in ("height_low", "reverse_height"):
        target[2] = .305 + (.23-.305)*progress
    elif role in ("height_high", "forward_height"):
        target[2] = .305 + (.42-.305)*progress
    return dict(role=role, recovery=False, command=command, target=target,
                pitch_deg=pitch, birth_height_m=birth_height)


def commands_at(cases, elapsed, plan):
    birth = elapsed.new_tensor([c["command"] for c in cases])
    target = elapsed.new_tensor([c["target"] for c in cases])
    def fraction(times):
        start, full, stop, end = times
        return torch.minimum(((elapsed-start)/(full-start)).clamp(0., 1.),
                             ((end-elapsed)/(end-stop)).clamp(0., 1.))
    movement = fraction(plan["tracking"]["command_ramps_seconds"])
    height = fraction(plan["tracking"]["height_ramps_seconds"])
    return birth + (target-birth)*torch.stack((movement, movement, height), -1)


def tracking_density(height, gravity, velocity, omega, command, plan):
    """Zero-motion requests remain bit-for-bit the original009 score function."""
    standing = scores(height, gravity, velocity, omega, command[:, 2])
    cfg = plan["tracking"]
    error = (velocity[:, 0]-command[:, 0])/cfg["motion_velocity_reward_width_m_s"]
    yaw = (omega[:, 2]-command[:, 1])/cfg["motion_yaw_reward_width_rad_s"]
    upright = ((1.-gravity[:, 2])*.5).clamp(0., 1.)
    attitude = upright*torch.exp(-gravity[:, :2].square().sum(-1)/math.sin(math.radians(5.))**2)
    height_score = torch.exp(-((height-command[:, 2])/.015).square())
    tracking = torch.exp(-error.square()-(velocity[:, 1]/.03).square()-yaw.square())
    angular = .1*torch.log1p((omega[:, :2]/.5).square()).sum(-1)
    moving = 12.*attitude*height_score*tracking-angular
    quiet = (command[:, :2].abs() < .01).all(-1)
    return torch.where(quiet, standing, moving)


def evaluation_cases(plan, task):
    rows = [case(plan, "stand", h, p) for h in ((.305,) if task == "standing" else (.23, .305, .42))
            for p in plan["evaluation"]["pitch_deg"]]
    if task != "standing":
        rows += [case(plan, role, h) for h in (.23, .305, .42)
                 for role in ("forward", "reverse", "left", "right")]
        rows += [case(plan, role) for role in ("height_low", "height_high", "forward_height", "reverse_height")]
    return rows


def install_tracking_reward(env, plan):
    if not hasattr(env, "physical_standing_reward"):
        raise ValueError("Install the original1ms physical observer first")
    native_step = env.step

    def step(actions):
        obs, native_reward, done, extra = native_step(actions)
        state = extra["diagnostics"]
        density = tracking_density(state["height"], state["gravity"], state["velocity"],
                                   state["omega"], state["commands"], plan)
        physical = env.physical_standing_last
        costs = physical["paid_speed"] + physical["paid_contact"] + physical["paid_slew"]
        exact = density*env.policy_dt-costs-3.*state["terminated"]
        reward = torch.where(extra["policy_learning_mask"], exact, native_reward)
        extra["log"].update({"Reproduce009/score_density_per_second": density.mean(),
                            "Reproduce009/paid_reward_per_policy_step": reward.mean()})
        return obs, reward, done, extra

    env.step = step
    env.precise_standing_reward = dict(version="009-exact-zero-command-with-explicit-tracking-v1",
        zero_motion="precise_standing_reward.scores unchanged", physical_costs=env.physical_standing_reward)


def install_tracking_failures(env):
    """The standing task owns falls/body contact, including during movement."""
    native_step = env.step

    def step(actions):
        obs, reward, done, extra = native_step(actions)
        state = extra["diagnostics"]
        tilted = state["gravity"][:, 2] > -math.cos(math.radians(20.))
        wrong_height = (state["height"]-state["commands"][:, 2]).abs() > .025
        body_contact = state["nonwheel_force_peak_n"] > 2.
        failure = (tilted | wrong_height | body_contact) & extra["policy_learning_mask"] & ~done
        reward = reward-3.*failure
        done = done | failure
        extra["time_outs"][failure] = False
        extra["policy_bootstrap_mask"][failure] = False
        state["terminated"] |= failure
        state["done"] = done.clone()
        extra["log"]["Reproduce009/tracking_failure_events"] = failure.float().sum()
        if failure.any():
            if env.fixed_cases is None:
                env.reset(failure.nonzero().flatten())
                env.history.last_tick = None
                obs = env.get_observations()
            else:
                env.finished |= failure
        return obs, reward, done, extra

    env.step = step
    return dict(version="009-tracking-upright-failure-v1", tilt_deg=20., height_error_m=.025,
                nonwheel_force_n=2., failure_cost=3., state_writes="reset only")


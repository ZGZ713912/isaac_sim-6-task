"""Pure task math. Commands change; all six motor actions always belong to RL."""
from __future__ import annotations

from copy import deepcopy
import math
import torch

ROLES = ("prone", "stand", "forward", "reverse", "spin_left", "spin_right",
         "arc_fl", "arc_fr", "arc_rl", "arc_rr", "height")
REFERENCE_DT = .005
REFERENCE_GAMMA = .9974905699336811
REFERENCE_LAMBDA = .9872585449014338


def timing(plan):
    """Historical v1 stays 200Hz/128; v2 explicitly selects 100Hz/64."""
    from .control_contract import FINE_POLICY_100_TIMING, FINE_POLICY_200_TIMING
    return dict(FINE_POLICY_200_TIMING if plan["schema"] == "wheel-leg-prone-motion-v1"
                else FINE_POLICY_100_TIMING)


def discounts(plan):
    """Retain discount and GAE horizons in seconds when the actor period changes."""
    ratio = timing(plan)["policy_dt"] / REFERENCE_DT
    return dict(gamma=REFERENCE_GAMMA**ratio, lam=REFERENCE_LAMBDA**ratio)


def normalized_action_delta(delta_square, plan):
    reference = plan.get("action_rate_reference_dt", timing(plan)["policy_dt"])
    return delta_square * (reference/timing(plan)["policy_dt"])**2


def validate(plan):
    if plan.get("schema") not in ("wheel-leg-prone-motion-v1", "wheel-leg-prone-motion-v2", "wheel-leg-prone-motion-v3"):
        raise ValueError("Need the explicit prone-motion contract")
    v2 = plan["schema"] != "wheel-leg-prone-motion-v1"
    if v2 and (plan.get("timing") != timing(plan) or plan.get("discounts") != discounts(plan)
               or plan.get("action_rate_reference_dt") != REFERENCE_DT):
        raise ValueError("100Hz requires explicit 1ms physics/PD, discount horizons and action-rate reference")
    if (plan["num_envs"] < 20 or plan["rollout_steps"] != (64 if v2 else 128)
            or plan["max_updates"] != 10000 or len(plan["roles_per20"]) != 20
            or set(plan["roles_per20"]) != set(ROLES)):
        raise ValueError("Training population/budget or persistent role coverage changed")
    if (plan["roles_per20"].count("prone") != 6 or plan["roles_per20"].count("stand") != 5
            or plan["forward_speed_m_s"] != 2. or plan["spin_target_rpm"] != 100.
            or plan["spin_minimum_rpm"] != 80.):
        raise ValueError("Keep30% prone,25% quiet, both signs2m/s and80..100rpm")
    if not (plan["post_completion_quiet_seconds"] == 30.
            and plan["standing_seconds"] == 30.
            and plan["training_prone_seconds"] >= plan["rise_deadline_seconds"] + 30. + plan["motion_seconds"]):
        raise ValueError("Rise -> full30s quiet -> motion must fit the episode")
    if (plan["domain_sequence"][0] != dict(start_update=0, profile="nominal", strength=0.)
            or [x["start_update"] for x in plan["domain_sequence"]] != sorted(set(x["start_update"] for x in plan["domain_sequence"]))
            or plan["domain_sequence"][-1] != dict(start_update=9000, profile="requested", strength=1.)):
        raise ValueError("Need ordered, explicit v2 domains ending in full requested ranges")
    if (plan.get("snapshot_interval") != 250 or plan.get("review_updates") != []
            or plan.get("execution") != "single_learn_call_manual_review"):
        raise ValueError("One continuous10000-update learn;250-update snapshots; manual reviews only")
    return deepcopy(plan)


def snapshot_name(completed, plan):
    """Names count COMPLETE PPO updates, independently of RSL's zero-based it."""
    if 0 < completed <= plan["max_updates"] and completed % plan["snapshot_interval"] == 0:
        return f"model_{completed:05d}.pt"
    return None


def domain_stage(plan, completed):
    return next((x["profile"], x["strength"]) for x in reversed(plan["domain_sequence"])
                if completed >= x["start_update"])


def progress(completed, start, end):
    return max(0., min(1., (completed-start)/(end-start)))


def target(role, completed, plan, *, final=False, height=.305):
    p = 1. if final else progress(completed, plan["foundation_until_update"], plan["speed_full_update"])
    spin = 1. if final else progress(completed, plan["foundation_until_update"], plan["spin_full_update"])
    h = 1. if final else progress(completed, plan["foundation_until_update"], plan["height_full_update"])
    v, w = plan["forward_speed_m_s"]*p, plan["spin_target_rpm"]*math.tau/60.*spin
    pairs = dict(prone=(v, 0.), stand=(0., 0.), forward=(v, 0.), reverse=(-v, 0.),
                 spin_left=(0., w), spin_right=(0., -w), arc_fl=(v, .6*p), arc_fr=(v, -.6*p),
                 arc_rl=(-v, .6*p), arc_rr=(-v, -.6*p), height=(0., 0.))
    vx, yaw = pairs[role]
    return [vx, yaw, .305+(height-.305)*h]


def envelope(seconds):
    """2s quiet,3s ramp,8s hold,3s braking,4s quiet; no discontinuous target."""
    return torch.minimum(((seconds-2.)/3.).clamp(0., 1.), ((16.-seconds)/3.).clamp(0., 1.))


def commands(targets, elapsed, recovered_at, prone, plan):
    age = elapsed.clone()
    # An actual completion event, never spawn time, starts the30s quiet clock.
    age[prone] = elapsed[prone]-recovered_at[prone]-plan["post_completion_quiet_seconds"]
    factor = envelope(torch.nan_to_num(age, nan=-1.))
    value = targets.clone()
    value[:, :2] *= factor[:, None]
    value[:, 2] = .305+(targets[:, 2]-.305)*factor
    value[prone & ~torch.isfinite(recovered_at), :] = value.new_tensor([0., 0., .305])
    return value


def stable(gravity, height, velocity, omega, dq, wheel_contact, nonwheel_force):
    return ((gravity[:, 2] < -math.cos(math.radians(8.))) & ((height-.305).abs() < .02)
            & (velocity[:, :2].norm(dim=-1) < .08) & (omega.norm(dim=-1) < .5)
            & (dq[:, [0, 1, 3, 4]].abs().amax(-1) < 3.) & wheel_contact.all(-1)
            & (nonwheel_force <= 2.))


def reward_terms(gravity, height, velocity, omega, command, torque, dq, action_delta,
                 excursion, anchor_error, rising, supported):
    """Independent dense terms; wrong height never zeroes speed/yaw credit."""
    quiet = (command[:, :2].abs() < command.new_tensor([.05, .1])).all(-1) & ~rising
    position_hold = (command[:, 0].abs() < .05) & ~rising
    ev, ew, eh = velocity[:, 0]-command[:, 0], omega[:, 2]-command[:, 1], height-command[:, 2]
    # Gaussian terms encourage precision; log1p terms retain a slope far away.
    pos = ((anchor_error-.01).clamp_min(0.)/.04).square().clamp_max(4.)
    level = (gravity-gravity.new_tensor([0., 0., -1.])).square().sum(-1)
    return dict(upright=2.*(-gravity[:, 2]).clamp(-1., 1.),
        height=2.*torch.exp(-(eh/.025).square()), height_error=-2.*torch.log1p((eh/.04).square()),
        forward=2.*torch.exp(-(ev/.25).square()), forward_error=-torch.log1p((ev/.5).square()),
        yaw=1.5*torch.exp(-(ew/.5).square()), yaw_error=-torch.log1p((ew/.8).square()),
        level=-2.*level/math.sin(math.radians(15.))**2, lateral=-.5*torch.log1p((velocity[:, 1]/.2).square()),
        body_rate=-.05*omega[:, :2].square().sum(-1), vertical=-.1*velocity[:, 2].square(),
        quiet=4.*quiet*supported*torch.exp(-(eh/.02).square()-level/math.sin(math.radians(5.))**2-(velocity[:, :2]/.08).square().sum(-1)
            -(omega/.5).square().sum(-1)),
        quiet_position=-2.*position_hold*pos, rise_position=-rising.float()*((excursion-.1).clamp_min(0.)/.1).square(),
        rise_time=-rising.float(), effort=-.0001*torque.square().mean(-1),
        leg_speed=-.002*dq[:, [0, 1, 3, 4]].square().mean(-1), action_rate=-.01*action_delta)


def failures(gravity, height, nonwheel, excursion, elapsed, rising, initialized, plan):
    supported_phase = initialized | ~rising
    return dict(refall=supported_phase & ((gravity[:, 2] > -math.cos(math.radians(25.)))
                  | (height < .15) | (height > .48) | (nonwheel > 2.)),
                rise_timeout=rising & (elapsed >= plan["rise_deadline_seconds"]),
                rise_excursion=rising & (excursion > plan["rise_training_abort_excursion_m"]))

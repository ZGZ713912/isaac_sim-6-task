"""Reward and outcome-driven height goals for an in-place, normally-up rise."""
from collections import deque
from copy import deepcopy
import math
import random

import torch

LEG_CLASSES = ("front_front", "rear_rear", "front_rear", "rear_front")


def validate(config):
    c = deepcopy(config)
    if c.get("schema") != "wheel-leg-prone-support-v1":
        raise ValueError("Need the explicit prone-support contract")
    if c.get("goal_heights_m") != [.23, .265, .305] or c.get("leg_classes") != list(LEG_CLASSES):
        raise ValueError("Keep low capture, intermediate lift and default standing goals explicit")
    if (c.get("capture_hold_s") != 3. or c.get("ignored_support_loss_s") != .002
            or c.get("actor_inputs_added") is not False or c.get("scripted_actions") is not False
            or c.get("state_writes") != "episode_reset_only"):
        raise ValueError("Capture clock, sensor boundary or control ownership changed")
    if c.get("goal_changes") != "future_resets_only_after_completed_outcome_window":
        raise ValueError("Ground truth must not change an active episode's command")
    for name in ("episode_active_s", "alignment_width_m", "capture_alignment_m", "tilt_deg",
                 "height_error_m", "linear_speed_m_s", "angular_speed_rad_s", "leg_speed_rad_s",
                 "wheel_clearance_m", "fold_scale_deg", "excursion_abort_m"):
        if not math.isfinite(c[name]) or c[name] <= 0.:
            raise ValueError("Invalid prone-support parameter: " + name)
    if c["stage_window"] != 64 or c["stage_success_rate"] != .75:
        raise ValueError("Height progression needs a complete, balanced outcome window")
    if c.get("discount_time_constant_s") != 10. or c.get("gae_residual_time_constant_s") != 1.:
        raise ValueError("Keep the physical-time discount and GAE residual kernel explicit")
    return c


def coefficients(config, dt):
    c = validate(config)
    return dict(gamma=math.exp(-dt/c["discount_time_constant_s"]),
                lam=math.exp(-dt*(1./c["gae_residual_time_constant_s"]-1./c["discount_time_constant_s"])))


class HeightCurriculum:
    """Goals are fixed at birth; simulator contacts are never actor commands."""
    def __init__(self, seed, config):
        self.config = validate(config)
        self.rng = random.Random(seed)
        self.phase = 0
        self.window = {name: deque(maxlen=config["stage_window"]//4) for name in LEG_CLASSES}
        self.completed_by_phase = [0, 0, 0]

    def observe(self, birth_phase, leg_class, success):
        if leg_class not in LEG_CLASSES:
            raise ValueError("Unknown leg class")
        if success:
            self.completed_by_phase[birth_phase] += 1
        if birth_phase != self.phase:
            return
        self.window[leg_class].append(bool(success))
        if (self.phase < 2 and all(len(cell) == cell.maxlen
                and sum(cell)/len(cell) >= self.config["stage_success_rate"] for cell in self.window.values())):
            self.phase += 1
            for cell in self.window.values():
                cell.clear()

    def state_dict(self):
        return dict(phase=self.phase, height_m=self.config["goal_heights_m"][self.phase],
            window={name:list(cell) for name,cell in self.window.items()}, completed_by_phase=list(self.completed_by_phase),
            rng=self.rng.getstate(), progression="per-class completed physical success; never PPO count or reward")


def validate_plan(plan):
    c = validate(plan["prone_support"])
    if plan.get("rollover_training") or plan.get("jump_training"):
        raise ValueError("Normally-up prone support is a separate skill")
    if (plan["timing"] != dict(physics_dt=.001, pc_control_dt=.005, policy_dt=.01, decimation=10)
            or plan["completion_hold_seconds"] != .01 or not plan.get("chassis_collision")
            or not plan.get("encoder_increment_actions")):
        raise ValueError("Use the admitted explicit fine-step, feedback and corrected-model path")
    if c["reset_parameters"]["file"] != "model/wheel_leg_v2.0.2/training/prone_support_resets_v1.json":
        raise ValueError("Unknown normally-up reset parameter bank")
    return c


def scores(gravity, height, velocity, omega, knee_deg, wheel_x_error, wheel_below_com, goal, spec):
    level = torch.exp(-((1. + gravity[:, 2])/ .08).clamp_min(0.))
    # Score each wheel. Opposite fore/aft errors cannot cancel at the midpoint.
    alignment = torch.exp(-(wheel_x_error/spec["alignment_width_m"]).square().mean(-1))
    below = torch.sigmoid((wheel_below_com.amin(-1)-.04)/.02)
    alignment *= below
    lift = ((height-spec["belly_height_m"])/(goal-spec["belly_height_m"])).clamp(0., 1.)
    folded = torch.exp(-((knee_deg-spec["fold_target_deg"]).clamp_min(0.)/spec["fold_scale_deg"]).square()).mean(-1)
    folding_weight = (1.-lift)*(1.-alignment)
    height_score = torch.exp(-((height-goal)/.025).square())
    quiet = torch.exp(-(velocity.norm(dim=-1)/.15).square()-(omega.norm(dim=-1)/.7).square())
    return dict(level=level, alignment=alignment, lift=lift, folded=folded,
                folding_weight=folding_weight, height=height_score, quiet=quiet)


def potential(s):
    # Folding is progress-only guidance and disappears before final extension.
    return s["level"]*(s["folding_weight"]*s["folded"] + 4.*s["alignment"]
        + 8.*s["lift"] + 2.*s["height"]*s["alignment"])


def densities(s, velocity, omega, torque, motor_velocity, action_delta, excursion,
              wheel_load_fraction, nonwheel_load_fraction, impulse_peak, weight, recovered):
    support = (wheel_load_fraction.amin(-1)/.1).clamp(0., 1.)
    wheels_only = (1.-nonwheel_load_fraction/.1).clamp(0., 1.)
    impact_excess = (impulse_peak/(weight[:, None]*.02)-1.).clamp_min(0.)
    return dict(time=-(~recovered).float(), orientation=s["level"],
        wheel_under_body=2.*s["level"]*s["alignment"]*(1.-s["lift"]),
        supported_lift=2.*s["level"]*s["lift"]*support,
        quiet_goal=8.*s["level"]*s["height"]*s["quiet"]*support*wheels_only,
        in_place=-2.*((excursion-.01).clamp_min(0.)/.1).square(),
        planar_speed=-.2*(velocity[:, :2]/.3).square().sum(-1),
        angular_speed=-.03*omega.square().sum(-1), torque=-2e-5*torque.square().sum(-1),
        # The parent supplies the mean squared six-axis change for each lane.
        # Preserve that lane axis; never reduce it across parallel robots.
        action_rate=-.01*action_delta,
        leg_speed=-.001*motor_velocity[:, :4].square().sum(-1),
        wheel_speed=-.0001*motor_velocity[:, 4:].square().sum(-1),
        impulse_excess=-.15*impact_excess.square().sum(-1))


def capture_conditions(gravity, height, velocity, omega, motor_velocity, wheel_x_error,
                       clearance, wheel_load_fraction, nonwheel_load_fraction, excursion, goal, spec):
    posture = ((gravity[:, 2] < -math.cos(math.radians(spec["tilt_deg"])))
        & ((height-goal).abs() <= spec["height_error_m"])
        & (velocity.norm(dim=-1) < spec["linear_speed_m_s"])
        & (omega.norm(dim=-1) < spec["angular_speed_rad_s"])
        & (motor_velocity[:, :4].abs().amax(-1) < spec["leg_speed_rad_s"])
        & (wheel_x_error.abs().amax(-1) < spec["capture_alignment_m"])
        & (nonwheel_load_fraction < .05) & (excursion <= spec["getup_xy_max_m"]))
    # Force is a load test; geometric clearance is independently checked.
    support = (clearance <= spec["wheel_clearance_m"]).all(-1) & (wheel_load_fraction > .01).all(-1)
    return posture, support


def advance_capture(streak, support_gap, posture, support, active, dt, ignored_loss_s):
    gap = torch.where(active & posture & ~support, support_gap+1, 0)
    acceptable = active & posture & (support | (gap <= round(ignored_loss_s/dt)))
    # A missing-contact sample cannot start a new capture streak.
    acceptable &= support | (streak > 0)
    return torch.where(acceptable, streak+1, 0), gap

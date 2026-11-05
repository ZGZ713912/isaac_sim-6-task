"""An inverted-to-prone skill, separate from standing and locomotion.

Reward geometry is privileged training information. The actor still receives
only its existing encoder/IMU policy35 frame and owns all six motor outputs.
"""
from collections import deque
from copy import deepcopy
import math
import random

import torch


CONTRACT = dict(schema="wheel-leg-rollover-learning-v1",
    birth_family="roof_down", leg_classes=["front_front", "rear_rear", "front_rear", "rear_front"],
    births="authenticated_bank_independent_left_right_angles_train_split_only",
    goal="normally_up_ground_supported_prone_at_rest_not_standing",
    tilt_deg=15., height_min_m=.12, height_max_m=.24, linear_speed_m_s=.08,
    angular_speed_rad_s=.35, leg_speed_rad_s=1., support_weight_fraction=.2,
    capture_hold_s=1., episode_active_s=8., evaluation_after_capture_s=2.,
    shaping_knee_deg=110., shaping_knee_scale_deg=15., shaping_common_rate_rad_s=3.,
    knee_is_soft_guidance_not_completion=True, wheel_contact_required=False,
    prefix=dict(durations_s=[1.7, 1.2, .6, 0.], zero_prefix_fraction=.25,
        stage_window=64, stage_success_rate=.75, same_direction_rate_rad_s=3.,
        alignment_rate_gain=2., relative_rate_gain=3., no_physical_reset_at_release=True,
        excluded_from_PPO=True, previous_actor_action_zero_at_release=True),
    reward=dict(time=-1., orientation=2., knee=.6, alignment=.4, coordinated_rotation=.3,
        capture=4., angular_braking=.15, torque=.00002, action_rate=.01,
        leg_speed=.001, wheel_speed=.0005, impact_excess=.15,
        potential_gain=1., completion_bonus=10.),
    safety="original_mechanical_and_impact_guards_unchanged",
    cold_scope="all_roof_holdouts_four_leg_classes_two_headings_zero_script",
    actor_inputs_added=False, standing_or_16001_claim=False)


def validate(config):
    if config != CONTRACT:
        raise ValueError("Rollover learning requires its complete versioned goal and curriculum")
    return deepcopy(config)


def validate_plan(plan):
    validate(plan["rollover_training"])
    from .recovery_actions import RANGE_CONTRACT
    if (plan.get("encoder_increment_actions") != RANGE_CONTRACT or
            plan.get("timing", {}).get("physics_dt") != .001 or
            plan["timing"]["policy_dt"] != .01 or
            plan["timing"]["pc_control_dt"] != .005 or
            plan.get("completion_hold_seconds") != .01):
        raise ValueError("Rollover uses range-v2 encoders and a native 1s hold, with 1000/200/100Hz clocks")
    if "recovery_policy" in plan or "recovery_passive_window" in plan:
        raise ValueError("This rollover candidate retains the original IID policy and passive enable")


def wrap(value):
    return torch.atan2(value.sin(), value.cos())


def leg_coordinates(q, dq):
    """Paired shaft common angle in one physical direction, not knee angle."""
    sign = q.new_tensor([1., -1.])
    q, dq = q.unflatten(-1, (2, 2)), dq.unflatten(-1, (2, 2))
    relative = sign*wrap(q[..., 1]-q[..., 0])
    common = wrap(sign*q[..., 0]+.5*relative)
    common_rate = sign*dq.mean(-1)
    return common, relative, common_rate


def prone_condition(gravity, height, linear_velocity, omega, motor_velocity, support_n, weight_n,
                    config=CONTRACT):
    """No wheel-load gate: legs and chassis may carry the normally-up body."""
    return ((-gravity[..., 2] >= math.cos(math.radians(config["tilt_deg"])))
        & (height >= config["height_min_m"]) & (height <= config["height_max_m"])
        & (linear_velocity.norm(dim=-1) <= config["linear_speed_m_s"])
        & (omega.norm(dim=-1) <= config["angular_speed_rad_s"])
        & (motor_velocity[..., :4].abs().amax(-1) <= config["leg_speed_rad_s"])
        & (support_n >= weight_n*config["support_weight_fraction"]))


def scores(gravity, height, linear_velocity, omega, knee_deg, common, common_rate, valid):
    up = ((1.-gravity[:, 2])*.5).clamp(0., 1.)
    # Preparation fades continuously. Orientation progress is never gated by it.
    inverted = ((gravity[:, 2]+.2)/1.2).clamp(0., 1.)
    shape = torch.exp(-((knee_deg-110.)/15.).square()).mean(-1)*valid
    alignment = .5*(1.+(common[:, 0]-common[:, 1]).cos())*valid
    coordinated = torch.exp(-((common_rate-3.)/3.).square()).mean(-1)*valid
    low = ((height >= .12) & (height <= .24)).float()
    capture = (up**8)*low*torch.exp(-(linear_velocity.norm(dim=-1)/.12).square()
        -(omega.norm(dim=-1)/.6).square())*valid
    return dict(up=up, inverted=inverted, shape=shape, alignment=alignment,
                coordinated=coordinated, capture=capture)


def potential(s):
    # Bounded state potential, not positive per-step angle travel: repeated
    # rotations cannot accumulate an uncapped progress bonus.
    return 8.*s["up"]+s["inverted"]*(s["shape"]+s["alignment"])+3.*s["capture"]


def densities(s, omega, torque, dq_policy, action_delta, impulse, weight_n, config=CONTRACT):
    w = config["reward"]
    nominal_window = (weight_n*.02).clamp_min(1e-6)
    excess = (impulse/nominal_window[:, None]-1.5).clamp_min(0.).square().sum(-1)
    return dict(time=torch.full_like(s["up"], w["time"]), orientation=w["orientation"]*s["up"],
        knee=w["knee"]*s["inverted"]*s["shape"],
        alignment=w["alignment"]*s["inverted"]*s["alignment"],
        coordinated_rotation=w["coordinated_rotation"]*s["inverted"]*s["coordinated"],
        capture=w["capture"]*s["capture"],
        angular_braking=-w["angular_braking"]*(s["up"]**4)*omega.square().sum(-1),
        torque=-w["torque"]*torque.square().mean(-1), action_rate=-w["action_rate"]*action_delta,
        leg_speed=-w["leg_speed"]*dq_policy[:, :4].square().mean(-1),
        wheel_speed=-w["wheel_speed"]*dq_policy[:, 4:].square().mean(-1),
        impact_excess=-w["impact_excess"]*excess)


def prefix_action(measured, gravity, relative_110, config, *, policy_dt, wheel_orbits=None):
    """Sensor-feedback preparation and sweep, used only in excluded prefixes.

    Each leg moves from its own angle. Alignment and common rotation occur
    together; body pose is never held fixed and no alignment milestone gates
    actual flip progress. Mirrored shafts have opposite numeric signs.
    """
    from .recovery_actions import RANGE_CONTRACT
    common, relative, _ = leg_coordinates(measured, torch.zeros_like(measured))
    # Encoder FK supplies actual wheel-to-hip orbit when available; motor
    # common angle alone is not exact leg alignment for unequal knees.
    angle = common if wheel_orbits is None else wheel_orbits
    mean = torch.atan2(angle.sin().sum(-1), angle.cos().sum(-1))
    direction = 1. if wheel_orbits is None else -1.
    correction = direction*config["prefix"]["alignment_rate_gain"]*wrap(mean[:, None]-angle)
    tilt = torch.acos((-gravity[:, 2]).clamp(-1., 1.))
    sweeping = tilt > math.radians(20.)
    physical_rate = (config["prefix"]["same_direction_rate_rad_s"]+correction)*sweeping[:, None]
    relative_rate = (relative_110-relative)*config["prefix"]["relative_rate_gain"]*sweeping[:, None]
    sign = measured.new_tensor([1., -1.])
    raw = torch.stack((sign*physical_rate/RANGE_CONTRACT["common_rate_rad_s"],
                       relative_rate/RANGE_CONTRACT["relative_rate_rad_s"]), -1)
    return raw.flatten(-2).clamp(-1., 1.)


class PrefixCurriculum:
    def __init__(self, seed, config=CONTRACT):
        self.config = validate(config)
        self.rng = random.Random(seed)
        self.phase = 0
        self.window = deque(maxlen=config["prefix"]["stage_window"])
        self.episodes = self.successes = 0

    def sample(self, lane):
        # Deterministic lane stratum guarantees pure-RL samples even in the
        # first64-step rollout; the remaining lanes use independent draws.
        if lane % 4 == 0 or self.phase == 3:
            return 0., self.phase
        return self.config["prefix"]["durations_s"][self.phase], self.phase

    def observe(self, birth_phase, success):
        self.episodes += 1
        self.successes += bool(success)
        if birth_phase != self.phase:
            return
        self.window.append(bool(success))
        c = self.config["prefix"]
        if (self.phase < 3 and len(self.window) == c["stage_window"]
                and sum(self.window)/len(self.window) >= c["stage_success_rate"]):
            self.phase += 1
            self.window.clear()

    def state_dict(self):
        return dict(phase=self.phase, window=list(self.window), episodes=self.episodes,
                    successes=self.successes, rng=self.rng.getstate())

    def load_state_dict(self, state):
        self.phase = state["phase"]
        if not 0 <= self.phase <= 3:
            raise ValueError("Invalid rollover prefix phase")
        self.window = deque(state["window"], maxlen=self.config["prefix"]["stage_window"])
        self.episodes, self.successes = state["episodes"], state["successes"]
        self.rng.setstate(state["rng"])

"""Mixed airborne/active-jump task math; this module never writes motor or root state.

Adapted conceptually from SCUTRobotLab wheeled-legged_RL's MIT-licensed
wheelbipe/state_machines/{airborne,jump_takeoff}. Robot geometry, actuators and
reward weights belong to this repository, not to the reference robot.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch

BASE, AIRBORNE, JUMP = 0, 1, 2
WAIT, CROUCH, PUSH, FLIGHT, LAND, COMPLETE, FAILED = range(7)
PHASE_NAMES = ("wait", "crouch", "push", "flight", "land", "complete", "failed")
COUNTERS = ("airborne_births", "jump_requests", "liftoffs", "landings", "airborne_successes", "jump_successes")


def validate(cfg):
    if cfg.get("schema") != "wheel-leg-jump-motion-v1":
        raise ValueError("Need the versioned mixed jump contract")
    roles = cfg.get("roles_per10", [])
    if len(roles) != 10 or sorted(roles) != sorted(["base"]*6+["airborne"]*2+["jump"]*2):
        raise ValueError("Jump course retains 60% base, 20% airborne and 20% active-jump lanes")
    for name in ("episode_seconds", "crouch_seconds", "push_timeout_seconds", "flight_timeout_seconds",
                 "landing_timeout_seconds", "landing_hold_seconds", "liftoff_hold_seconds",
                 "minimum_flight_seconds", "ground_hold_seconds", "minimum_release_vz_m_s",
                 "clearance_m", "landing_clearance_m", "maximum_height_m", "maximum_tilt_deg",
                 "landing_force_soft_n", "landing_force_scale_n"):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid jump parameter: "+name)
    for name in ("request_delay_s", "air_clearance_start_m", "air_clearance_final_m",
                 "apex_above_release_start_m", "apex_above_release_final_m"):
        pair = cfg[name]
        if (not isinstance(pair, list) or len(pair) != 2 or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in pair)
                or not 0 <= pair[0] <= pair[1]):
            raise ValueError("Invalid jump sampling range: "+name)
    if not (0 < cfg["crouch_height_m"] < cfg["standing_height_m"] < cfg["release_height_m"] < cfg["maximum_height_m"]):
        raise ValueError("Jump heights must order crouch < stand < release < ceiling")
    if (type(cfg["difficulty_full_update"]) is not int or cfg["difficulty_full_update"] <= 0
            or cfg["external_takeoff_assistance"] is not False
            or cfg["motor_owner"] != "six_axis_rl"
            or cfg["landing_clearance_m"] >= cfg["clearance_m"]):
        raise ValueError("Jump must retain RL ownership and separate airborne/contact hysteresis")
    if (cfg["request_delay_s"][1]+cfg["crouch_seconds"]+cfg["push_timeout_seconds"]
            +cfg["flight_timeout_seconds"]+cfg["landing_timeout_seconds"] >= cfg["episode_seconds"]):
        raise ValueError("Jump deadlines must fit the declared episode")
    if cfg["release_height_m"]+cfg["apex_above_release_final_m"][1] >= cfg["maximum_height_m"]:
        raise ValueError("Requested apex must stay below the physical termination ceiling")
    return deepcopy(cfg)


def sample_range(cfg, name, progress, random):
    low = cfg[name+"_start_m"]
    high = cfg[name+"_final_m"]
    lo, hi = (a+(b-a)*progress for a, b in zip(low, high))
    return lo+(hi-lo)*random


def ballistic_push(crouch_height, release_height, apex_height):
    velocity = (2.*9.81*(apex_height-release_height).clamp_min(.001)).sqrt()
    duration = 2.*(release_height-crouch_height)/velocity
    return duration, velocity, velocity/duration


class JumpMotion:
    """Event detector at physical rate; references are commands, never forced trajectories."""
    def __init__(self, count, device, dt, cfg, mass):
        self.cfg = validate(cfg)
        if not math.isfinite(dt) or dt <= 0 or dt > self.cfg["liftoff_hold_seconds"]:
            raise ValueError("Jump event sampling must resolve the liftoff hold")
        self.dt, self.count = dt, count
        self.weight = mass.to(device)*9.81
        self.mode = torch.zeros(count, device=device, dtype=torch.long)
        self.phase = torch.full_like(self.mode, WAIT)
        self.age = torch.zeros(count, device=device, dtype=torch.float64)
        for name in ("phase_time", "ground_hold", "air_hold", "flight_time", "landing_hold", "stable_hold",
                     "request_delay", "request_at"):
            setattr(self, name, torch.zeros_like(self.age))
        for name in ("target_apex", "peak_height", "release_vz", "wheel_force_peak", "interval_force_peak", "nonwheel_force_peak"):
            setattr(self, name, torch.zeros(count, device=device))
        for name in ("saw_flight", "landed", "succeeded", "liftoff_event", "landing_event", "success_event", "request_event"):
            setattr(self, name, torch.zeros_like(self.mode, dtype=torch.bool))
        self.total = torch.zeros(len(COUNTERS), dtype=torch.long, device=device)
        self.has_active = False
        self.last_reference = self.reference()
        self.interval_phase = self.phase.clone()

    @property
    def active(self):
        return self.mode != BASE

    def reset(self, ids, modes, delay, apex):
        self.mode[ids] = modes
        self.phase[ids] = torch.where(modes == AIRBORNE, FLIGHT, WAIT)
        for name in ("age", "phase_time", "ground_hold", "air_hold", "flight_time", "landing_hold", "stable_hold",
                     "request_at", "peak_height", "release_vz", "wheel_force_peak", "interval_force_peak", "nonwheel_force_peak"):
            getattr(self, name)[ids] = 0.
        for name in ("saw_flight", "landed", "succeeded", "liftoff_event", "landing_event", "success_event", "request_event"):
            getattr(self, name)[ids] = False
        self.request_delay[ids] = delay.to(self.request_delay.dtype)
        self.target_apex[ids] = apex.to(self.target_apex.dtype)
        self.has_active = bool(self.active.any())

    def reference(self):
        cfg = self.cfg
        h = torch.full_like(self.target_apex, cfg["standing_height_m"])
        v = torch.zeros_like(h)
        crouch = self.phase == CROUCH
        p = (self.phase_time/cfg["crouch_seconds"]).clamp(0., 1.).float()
        h = torch.where(crouch, h+(cfg["crouch_height_m"]-h)*(3.*p.square()-2.*p.pow(3)), h)
        duration, release, accel = ballistic_push(cfg["crouch_height_m"], cfg["release_height_m"], self.target_apex)
        t = torch.minimum(self.phase_time.float(), duration)
        h = torch.where(self.phase == PUSH, cfg["crouch_height_m"]+.5*accel*t.square(), h)
        v = torch.where(self.phase == PUSH, accel*t, v)
        return h, v

    def context(self):
        """Six reserved SCUT35 slots; slot 31 remains odometry validity.

        The mode, requested apex and command clock are task inputs. No root
        height/velocity, contact or wheel clearance enters the actor.
        """
        active = self.active
        requested = active & (self.phase != WAIT)
        return torch.stack(((~active).float(), (self.mode == AIRBORNE).float(),
            self.reference()[1]*active, (self.mode == JUMP).float(),
            5.*(self.target_apex-self.cfg["standing_height_m"])*active,
            torch.where(requested, self.age-self.request_at, 0.).clamp(0., 5.).float()), -1)

    def begin_interval(self):
        for name in ("liftoff_event", "landing_event", "success_event", "request_event"):
            getattr(self, name).zero_()
        self.last_reference = tuple(x.clone() for x in self.reference())
        self.interval_phase.copy_(self.phase)
        self.interval_force_peak.zero_()

    def _enter(self, mask, phase):
        self.phase[mask] = phase
        self.phase_time[mask] = 0.

    def observe(self, *, height, velocity, omega, gravity, wheel_clearance, wheel_force_z, nonwheel_force, enabled):
        cfg, dt = self.cfg, self.dt
        active = self.active & enabled
        old = self.phase.clone()
        births = active & (self.mode == AIRBORNE) & (self.age == 0.)
        self.total[0] += births.sum()
        self.age += dt*active
        self.phase_time += dt*active
        force = wheel_force_z.clamp_min(0.)
        supported = ((force > .05*self.weight[:, None]) & (wheel_clearance < cfg["landing_clearance_m"])).all(-1)
        air = (wheel_clearance > cfg["clearance_m"]).all(-1) & (force.sum(-1) < .1*self.weight)
        touching = ((force > .05*self.weight[:, None]) & (wheel_clearance < cfg["landing_clearance_m"])).any(-1)
        calm = (gravity[:, 2] < -math.cos(math.radians(12.))) & (omega.norm(dim=-1) < .8)
        self.ground_hold = torch.where(active & supported, self.ground_hold+dt, 0.)
        self.air_hold = torch.where(active & air, self.air_hold+dt, 0.)
        self.peak_height = torch.where(active, torch.maximum(self.peak_height, height), self.peak_height)
        self.wheel_force_peak = torch.where(active, torch.maximum(self.wheel_force_peak, force.amax(-1)), self.wheel_force_peak)
        self.interval_force_peak = torch.where(active, torch.maximum(self.interval_force_peak, force.amax(-1)), self.interval_force_peak)
        self.nonwheel_force_peak = torch.where(active, torch.maximum(self.nonwheel_force_peak, nonwheel_force), self.nonwheel_force_peak)
        request = (active & (old == WAIT) & (self.mode == JUMP) & calm
            & (self.age >= self.request_delay) & (self.ground_hold >= cfg["ground_hold_seconds"]-1e-7))
        self.request_event |= request
        self.total[1] += request.sum()
        self.request_at[request] = self.age[request]
        self._enter(request, CROUCH)
        self._enter(active & (old == CROUCH) & (self.phase_time >= cfg["crouch_seconds"]-1e-7), PUSH)
        # Ground force alone cannot establish flight, and airborne reset is
        # never counted as a self-propelled liftoff.
        liftoff = (active & (old == PUSH) & (self.air_hold >= cfg["liftoff_hold_seconds"]-1e-7)
            & (velocity[:, 2] >= cfg["minimum_release_vz_m_s"]))
        self.liftoff_event |= liftoff
        self.total[2] += liftoff.sum()
        self.saw_flight |= liftoff | (active & (self.mode == AIRBORNE) & air)
        self.release_vz[liftoff] = velocity[liftoff, 2]
        self._enter(liftoff, FLIGHT)
        flying = active & (old == FLIGHT)
        self.flight_time += dt*(flying & air)
        self.landing_hold = torch.where(flying & touching, self.landing_hold+dt, 0.)
        landing = flying & self.saw_flight & (self.flight_time >= cfg["minimum_flight_seconds"]-1e-7) & (self.landing_hold >= .01-1e-7)
        self.landing_event |= landing
        self.landed |= landing
        self.total[3] += landing.sum()
        self._enter(landing, LAND)
        stable = (active & (old == LAND) & supported & calm & ((height-cfg["standing_height_m"]).abs() < .035)
            & (velocity[:, :2].norm(dim=-1) < .15) & (velocity[:, 2].abs() < .15) & (nonwheel_force <= 2.))
        self.stable_hold = torch.where(stable, self.stable_hold+dt, 0.)
        apex_ok = (self.mode == AIRBORNE) | (self.peak_height >= self.target_apex-.025)
        success = active & ~self.succeeded & apex_ok & (self.stable_hold >= cfg["landing_hold_seconds"]-1e-7)
        self.success_event |= success
        self.succeeded |= success
        self.total[4] += (success & (self.mode == AIRBORNE)).sum()
        self.total[5] += (success & (self.mode == JUMP)).sum()
        self._enter(success, COMPLETE)
        push_timeout = active & (old == PUSH) & ~liftoff & (self.phase_time >= cfg["push_timeout_seconds"]-1e-7)
        flight_timeout = flying & ~landing & (self.phase_time >= cfg["flight_timeout_seconds"]-1e-7)
        land_timeout = active & (old == LAND) & ~success & (self.phase_time >= cfg["landing_timeout_seconds"]-1e-7)
        request_timeout = active & (old == WAIT) & ~request & (self.age >= self.request_delay+.8)
        self._enter(request_timeout | push_timeout | flight_timeout | land_timeout, FAILED)

    def reject_failed_successes(self, terminated):
        rejected = self.success_event & terminated
        self.total[4] -= (rejected & (self.mode == AIRBORNE)).sum()
        self.total[5] -= (rejected & (self.mode == JUMP)).sum()
        self.success_event &= ~terminated
        self.succeeded[rejected] = False
        self._enter(rejected, FAILED)

    def failures(self, height, gravity, nonwheel_force):
        return self.active & ((self.phase == FAILED) | (height < .14) | (height > self.cfg["maximum_height_m"])
            | (gravity[:, 2] > -math.cos(math.radians(self.cfg["maximum_tilt_deg"]))) | (nonwheel_force > 2.))

    def rewards(self, state, wheel_body, vz, action_delta):
        cfg = self.cfg
        phase, h = self.interval_phase, state["height"]
        flight = phase == FLIGHT
        push = phase == PUSH
        height_phase = ~flight
        legs = wheel_body[:, :, [0, 2]].norm(dim=-1)
        leg_target = torch.where(flight & (vz > 0.), .18, .25)
        # Both hip and knee affect wheel placement; all six RL actions retain
        # their existing PD, saturation, delay and gas-spring compensation.
        return dict(
            jump_upright=3.*torch.exp(-((1.+state["gravity"][:, 2])/.12).square()),
            jump_body_rate=-.08*state["omega"][:, :2].square().sum(-1),
            jump_horizontal=-.4*torch.log1p((state["velocity"][:, :2]/.3).square()).sum(-1),
            jump_height=4.*height_phase*torch.exp(-((h-self.last_reference[0])/.035).square()),
            jump_height_error=-1.*height_phase*torch.log1p(((h-self.last_reference[0])/.06).square()),
            jump_push_velocity=6.*push*torch.exp(-((vz-self.last_reference[1])/.5).square()),
            jump_push_shortfall=-2.*push*(self.last_reference[1]-vz).clamp_min(0.),
            jump_air_leg_length=2.*flight*torch.exp(-((legs-leg_target[:, None])/.06).square()).mean(-1),
            jump_wheel_centering=2.*flight*torch.exp(-(wheel_body[:, :, 0]/.08).square()).mean(-1),
            jump_air_wheel_effort=-.01*flight*state["torque"][:, [2, 5]].square().mean(-1),
            jump_landing_impact=-((self.interval_force_peak-cfg["landing_force_soft_n"]).clamp_min(0.)/cfg["landing_force_scale_n"]).clamp_max(5.),
            jump_effort=-.00005*state["torque"].square().mean(-1),
            jump_action_rate=-.002*action_delta,
            jump_leg_speed=-.0002*state["leg_speed_square"])

    def event_bonus(self):
        return 2.*self.liftoff_event+1.*self.landing_event+6.*self.success_event

    def summary(self):
        return dict(zip(COUNTERS, self.total.tolist()))

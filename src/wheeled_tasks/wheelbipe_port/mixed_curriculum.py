"""Disjoint task quotas and jump reward math, independent of the simulator.

The jump objectives follow the repository's JumpMotion experiment: crouch,
ballistic push, wheel placement in flight, and verified landing. Commands use
only request parameters and time. Contact-driven state belongs solely to the
reward observer and is never returned in the actor's seven control slots.
"""
from __future__ import annotations

import math

import torch

TASKS = ("stand", "forward", "reverse", "height", "turn", "dash", "spin", "jump", "airborne")
REWARD_NAMES = ("jump_height", "jump_push_velocity", "jump_air_placement", "jump_impact", "jump_events")


def validate(spec):
    fractions = spec["task_fractions"]
    if set(fractions) != set(TASKS) or any(not math.isfinite(v) or v <= 0 for v in fractions.values()):
        raise ValueError("Every mixed task requires a positive finite quota")
    if abs(sum(fractions.values())-1.) > 1e-9:
        raise ValueError("Mixed task quotas must sum to one")
    j = spec["jump"]
    if (j["external_takeoff_assistance"] is not False or j["motor_owner"] != "six_axis_rl"
            or not 0 < j["crouch_height_m"] < j["standing_height_m"] < j["release_height_m"]
            or j["landing_clearance_m"] >= j["clearance_m"]):
        raise ValueError("Invalid jump ownership, heights or clearance hysteresis")
    if not 0 < j["dense_scale"] <= 1 or j["event_bonuses"]["success"] > 2:
        raise ValueError("Jump payments exceed the declared foundation reward scale")
    return spec


def quota_counts(count, spec):
    validate(spec)
    if count < spec["sampling"]["minimum_envs"]:
        raise ValueError("Mixed task population is too small to cover every quota")
    exact = [count*spec["task_fractions"][name] for name in TASKS]
    result = [math.floor(x) for x in exact]
    order = sorted(range(len(TASKS)), key=lambda i: (-(exact[i]-result[i]), i))
    for i in order[:count-sum(result)]:
        result[i] += 1
    if min(result) < 1:
        raise ValueError("Every task requires at least one environment")
    return dict(zip(TASKS, result))


class TaskSchedule:
    """One fixed task per lane; resets cannot overwrite another task's quota."""
    def __init__(self, count, device, seed, spec):
        self.spec, self.device = validate(spec), device
        self.generator = torch.Generator(device=device).manual_seed(int(seed)+54173)
        counts = quota_counts(count, spec)
        roles = torch.cat([torch.full((counts[n],), i, device=device, dtype=torch.long)
                           for i, n in enumerate(TASKS)])
        self.role = roles[torch.randperm(count, device=device, generator=self.generator)]
        self.commands = torch.zeros(count, 3, device=device)
        self.height = torch.full((count,), spec["jump"]["standing_height_m"], device=device)
        self.next_sample = torch.zeros(count, device=device)
        self.delay = torch.zeros_like(self.height)
        self.apex = torch.zeros_like(self.height)
        self.drop = torch.zeros_like(self.height)
        self.wave_phase = torch.zeros_like(self.height)
        self.reset(torch.arange(count, device=device), 0)

    def mask(self, name):
        return self.role == TASKS.index(name)

    def random(self, count):
        return torch.rand(count, device=self.device, generator=self.generator)

    def reset(self, ids, update):
        j = self.spec["jump"]
        progress = min(max(update/j["difficulty_full_update"], 0.), 1.)
        def sample(key):
            low, high = [a+(b-a)*progress for a, b in zip(j[key+"_start_m"], j[key+"_final_m"])]
            return low+(high-low)*self.random(len(ids))
        self.apex[ids] = sample("apex")
        self.drop[ids] = sample("air_clearance")
        low, high = j["request_delay_s"]
        self.delay[ids] = low+(high-low)*self.random(len(ids))
        self.wave_phase[ids] = 2*math.pi*self.random(len(ids))
        self.next_sample[ids] = 0.

    def update_commands(self, age, phase):
        ids = (age >= self.next_sample).nonzero().flatten()
        if len(ids):
            r = self.role[ids]
            n, vmax, yawmax = len(ids), phase["vx_max_m_s"], phase["yaw_max_rad_s"]
            speed = .1+(max(vmax, .1)-.1)*self.random(n)
            low = self.random(n) < self.spec["sampling"]["low_speed_probability"]
            lo, hi = self.spec["sampling"]["low_speed_range_m_s"]
            speed = torch.where(low, lo+(min(hi, vmax)-lo)*self.random(n), speed)
            cmd = torch.zeros(n, 3, device=self.device)
            cmd[:, 0] = torch.where(r == TASKS.index("forward"), speed, 0.)
            cmd[:, 0] = torch.where(r == TASKS.index("reverse"), -speed, cmd[:, 0])
            # Alternate signs by lane and sample generation, preserving both
            # directions even when a task contains only one or two lanes.
            generation = torch.floor(age[ids]/self.spec["sampling"]["command_hold_seconds"]).long()
            sign = torch.where((ids+generation) % 2 == 0, 1., -1.)
            turn = r == TASKS.index("turn")
            travel_sign = torch.where((ids//2+generation//2) % 2 == 0, 1., -1.)
            cmd[:, 0] = torch.where(turn, travel_sign*.2, cmd[:, 0])
            cmd[:, 2] = torch.where(turn, sign*(.25+max(yawmax-.25, 0.)*self.random(n)), 0.)
            dash = r == TASKS.index("dash")
            dash_sign = torch.where((ids+generation//2) % 2 == 0, 1., -1.)
            sprint = generation % 2 == 0
            cmd[:, 0] = torch.where(dash & sprint, dash_sign*vmax*(.75+.25*self.random(n)), cmd[:, 0])
            spin = r == TASKS.index("spin")
            # Retain the original late-course spin range, including 100 rpm.
            spinmax = 4.5*math.pi if phase.get("completed_update", 0) >= 14000 else yawmax
            if phase["start_update"] >= 12000:
                spinmax = max(spinmax, 3.25*math.pi)
            cmd[:, 2] = torch.where(spin, sign*spinmax*(.5+.5*self.random(n)), cmd[:, 2])
            self.commands[ids] = cmd
            hlow, hhigh = phase["height_range_m"]
            self.height[ids] = hlow+(hhigh-hlow)*self.random(n)
            jump = (r == TASKS.index("jump")) | (r == TASKS.index("airborne"))
            self.height[ids[jump]] = self.spec["jump"]["standing_height_m"]
            hold = torch.where(r == TASKS.index("stand"),
                self.spec["sampling"]["standing_command_hold_seconds"], self.spec["sampling"]["command_hold_seconds"])
            self.next_sample[ids] = age[ids]+hold
        hlow, hhigh = phase["height_range_m"]
        wave = (hlow+hhigh)/2+(hhigh-hlow)/2*torch.sin(
            2*math.pi*self.spec["sampling"]["height_wave_hz"]*age+self.wave_phase)
        return self.commands, torch.where(self.mask("height"), wave, self.height)

    def reference(self, age):
        j = self.spec["jump"]
        t = age-self.delay
        release_v = (2*9.81*(self.apex-j["release_height_m"])).sqrt()
        duration = 2*(j["release_height_m"]-j["crouch_height_m"])/release_v
        push_time = (t-j["crouch_seconds"]).clamp_min(0.)
        crouch = self.mask("jump") & (t >= 0) & (t < j["crouch_seconds"])
        push = self.mask("jump") & (t >= j["crouch_seconds"]) & (push_time <= duration)
        flight = self.mask("jump") & (push_time > duration) & (push_time < duration+1.)
        active = ((self.mask("jump") & (t >= 0)) | self.mask("airborne")) & (age < j["outcome_seconds"])
        p = (t/j["crouch_seconds"]).clamp(0., 1.)
        h = torch.full_like(age, j["standing_height_m"])
        h = torch.where(crouch, h+(j["crouch_height_m"]-h)*(3*p.square()-2*p.pow(3)), h)
        acceleration = release_v/duration
        h = torch.where(push, j["crouch_height_m"]+.5*acceleration*push_time.square(), h)
        v = torch.where(push, acceleration*push_time, 0.)
        return dict(height=h, velocity=v, push=push, flight=flight, active=active,
                    requested=self.mask("jump") & (t >= 0))

    def context(self, base, age):
        """No sensor or observer parameter is accepted at this boundary."""
        result = base.clone()
        jump = self.mask("jump") & (age >= self.delay)
        air = self.mask("airborne")
        selected = jump | air
        result[selected] = 0.
        result[selected, 4] = 1.
        result[jump, 5] = self.apex[jump]
        result[selected, 6] = torch.where(jump, age-self.delay, age)[selected].clamp(0., 6.)
        return result


class JumpObserver:
    """Physical events affect payment only, never command references or timing."""
    def __init__(self, schedule, dt, mass):
        self.schedule, self.dt = schedule, dt
        self.j = schedule.spec["jump"]
        self.weight = mass*9.81
        z = schedule.height
        for name in ("air_time", "stable_time", "peak_height", "force_peak", "interval_force_peak"):
            setattr(self, name, torch.zeros_like(z))
        for name in ("flown", "landed", "succeeded", "failed", "liftoff_event", "landing_event", "success_event", "timeout_event"):
            setattr(self, name, torch.zeros_like(z, dtype=torch.bool))

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor) and value is not self.weight:
                value[ids] = 0

    def begin_interval(self):
        for name in ("liftoff_event", "landing_event", "success_event", "timeout_event", "interval_force_peak"):
            getattr(self, name).zero_()

    def observe(self, age, height, velocity, gravity, wheel_clearance, wheel_force, nonwheel_force):
        j, s = self.j, self.schedule
        jump, drop = s.mask("jump"), s.mask("airborne")
        eligible = drop | (jump & (age >= s.delay))
        air = (wheel_clearance > j["clearance_m"]).all(-1) & (wheel_force <= j["force_threshold_n"]).all(-1)
        support = (wheel_clearance < j["landing_clearance_m"]).all(-1) & (wheel_force > j["force_threshold_n"]).all(-1)
        self.air_time = torch.where(eligible & air, self.air_time+self.dt, 0.)
        liftoff = (jump & ~self.flown & (self.air_time >= j["minimum_air_seconds"]-1e-7)
                   & (velocity[:, 2] > j["minimum_release_vz_m_s"]))
        self.liftoff_event |= liftoff
        self.flown |= liftoff | (drop & (self.air_time >= j["minimum_air_seconds"]-1e-7))
        self.peak_height = torch.where(eligible, torch.maximum(self.peak_height, height), self.peak_height)
        touching = (wheel_clearance < j["landing_clearance_m"]).any(-1) & (wheel_force > j["force_threshold_n"]).any(-1)
        landing = eligible & self.flown & ~self.landed & touching
        self.landing_event |= landing
        self.landed |= landing
        self.force_peak = torch.maximum(self.force_peak, wheel_force.sum(-1)*eligible)
        self.interval_force_peak = torch.maximum(self.interval_force_peak, wheel_force.sum(-1)*eligible)
        calm = ((gravity[:, 2] < -math.cos(math.radians(12))) & (velocity[:, :2].norm(dim=-1) < .15)
                & (velocity[:, 2].abs() < .15) & ((height-j["standing_height_m"]).abs() < .035)
                & (nonwheel_force <= 2.))
        stable = eligible & self.landed & support & calm
        self.stable_time = torch.where(stable, self.stable_time+self.dt, 0.)
        apex_ok = drop | (self.peak_height >= s.apex-.025)
        success = eligible & ~self.succeeded & ~self.failed & apex_ok & (self.stable_time >= j["landing_hold_seconds"]-1e-7)
        self.success_event |= success
        self.succeeded |= success
        timeout = eligible & ~self.succeeded & ~self.failed & (age >= j["outcome_seconds"])
        self.timeout_event |= timeout
        self.failed |= timeout

    def reward_terms(self, age, height, vz, wheel_body, policy_dt):
        j, s = self.j, self.schedule
        ref = s.reference(age)
        active = ref["active"]
        flight = self.flown & ~self.landed & active
        height_phase = active & ~flight
        height_error = height-ref["height"]
        placement = torch.exp(-(wheel_body[:, :, 0]/.08).square()).mean(-1)
        legs = wheel_body[:, :, (0, 2)].norm(dim=-1)
        target_leg = torch.where(vz > 0, .18, .25)
        placement += torch.exp(-((legs-target_leg[:, None])/.06).square()).mean(-1)
        bonuses = j["event_bonuses"]
        events = (bonuses["liftoff"]*self.liftoff_event+bonuses["landing"]*self.landing_event
                  +bonuses["success"]*self.success_event+bonuses["timeout"]*self.timeout_event)
        return dict(jump_height=j["dense_scale"]*height_phase*(4*torch.exp(-(height_error/.035).square())
                       -torch.log1p((height_error/.06).square())),
                    jump_push_velocity=j["dense_scale"]*ref["push"]*(6*torch.exp(-((vz-ref["velocity"])/.5).square())
                       -2*(ref["velocity"]-vz).clamp_min(0.)),
                    jump_air_placement=j["dense_scale"]*2*flight*placement,
                    jump_impact=-j["dense_scale"]*((self.interval_force_peak/self.weight-j["impact_soft_bodyweights"])
                       .clamp_min(0.)/j["impact_scale_bodyweights"]).clamp_max(5.),
                    jump_events=events/policy_dt)


def rollout_metrics(roles, advantages, rewards, returns):
    """Inspect actual PPO tensors without changing normalization or gradients."""
    result = {}
    for name in TASKS:
        mask = roles == TASKS.index(name)
        for key, value in (("advantage", advantages), ("reward", rewards), ("return", returns)):
            selected = value[:, mask].detach().float()
            result["MixedRollout/"+name+"/"+key+"_rms"] = selected.square().mean().sqrt()
        result["MixedRollout/"+name+"/samples"] = int(mask.sum())*advantages.shape[0]
    return result

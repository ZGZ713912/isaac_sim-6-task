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
    if j.get("reward_version", 1) >= 3:
        positive = ("tuck_target_delta_m", "tuck_sigma_m", "tuck_hold_seconds",
                    "landing_extension_delta_m", "landing_extension_sigma_m",
                    "minimum_bilateral_tuck_delta_m", "landing_buffer_duration_s",
                    "landing_buffer_start_contact_s", "landing_buffer_min_down_vz_m_s",
                    "landing_buffer_height_sigma_m", "landing_buffer_velocity_sigma_m_s")
        if any(not math.isfinite(float(j.get(name, 0.))) or float(j.get(name, 0.)) <= 0.
               for name in positive):
            raise ValueError("v3 relative tuck and extension scales must be positive")
        if j["landing_extension_delta_m"] >= j["tuck_target_delta_m"]:
            raise ValueError("Landing extension must leave less tuck than the flight target")
        if j["landing_buffer_target_height_m"] <= 0. or j["landing_buffer_min_height_margin_m"] < 0.:
            raise ValueError("v3 landing buffer height and margin are invalid")
        for name in ("air_alignment_scale", "air_tuck_scale", "air_extension_scale"):
            if not math.isfinite(float(j.get(name, 0.))) or float(j.get(name, 0.)) < 0.:
                raise ValueError("v3 air reward scales must be finite and nonnegative")
    if j.get("reward_version",1) >= 4:
        if (not math.isfinite(j["apex_tuck_hold_seconds"]) or j["apex_tuck_hold_seconds"] < 0.
                or not 0. <= j["per_leg_progress_weight"] <= 1.):
            raise ValueError("Invalid v4 apex hold or per-leg progress weight")
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
        self.update = 0
        self.reset(torch.arange(count, device=device), 0)

    def mask(self, name):
        return self.role == TASKS.index(name)

    def random(self, count):
        return torch.rand(count, device=self.device, generator=self.generator)

    def reset(self, ids, update):
        j = self.spec["jump"]
        self.update = update
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
            spin_starts = self.spec["special_mode_starts"]
            spinmax = 4.5*math.pi if phase.get("completed_update", 0) >= spin_starts["spin_mid"] else yawmax
            if phase["start_update"] >= spin_starts["spin_low"]:
                spinmax = max(spinmax, 3.25*math.pi)
            cmd[:, 2] = torch.where(spin, sign*spinmax*(.5+.5*self.random(n)), cmd[:, 2])
            self.commands[ids] = cmd
            hlow, hhigh = phase["height_range_m"]
            self.height[ids] = hlow+(hhigh-hlow)*self.random(n)
            jump = (r == TASKS.index("jump")) | (r == TASKS.index("airborne"))
            self.height[ids[jump]] = self.spec["jump"]["standing_height_m"]
            moving = self.spec["jump"].get("moving_jump")
            if moving:
                ramp = min(max((self.update-moving["start_update"])/
                    (moving["full_update"]-moving["start_update"]), 0.), 1.)
                rank = (self.mask("jump").long().cumsum(0)-1)[ids]
                travel = (rank % 2 == 0) & (r == TASKS.index("jump"))
                direction = torch.where((rank//2+generation) % 2 == 0, 1., -1.)
                self.commands[ids, 0] = torch.where(travel,
                    direction*moving["maximum_vx_m_s"]*ramp, self.commands[ids, 0])
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
        v2 = j.get("reward_version", 1) >= 2
        flight_seconds = 2*release_v/9.81 if v2 else 1.
        flight = self.mask("jump") & (push_time > duration) & (push_time < duration+flight_seconds)
        active = ((self.mask("jump") & (t >= 0)) | self.mask("airborne")) & (age < j["outcome_seconds"])
        p = (t/j["crouch_seconds"]).clamp(0., 1.)
        h = torch.full_like(age, j["standing_height_m"])
        h = torch.where(crouch, h+(j["crouch_height_m"]-h)*(3*p.square()-2*p.pow(3)), h)
        acceleration = release_v/duration
        h = torch.where(push, j["crouch_height_m"]+.5*acceleration*push_time.square(), h)
        if v2:
            h = torch.where(flight, j["release_height_m"], h)
        v = torch.where(push, acceleration*push_time, 0.)
        return dict(height=h, velocity=v, push=push, flight=flight, active=active,
                    crouch=crouch, after_push=self.mask("jump") & (push_time > duration),
                    release_velocity=release_v, requested=self.mask("jump") & (t >= 0))

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
        # v3 air shaping is relative to the two wheel-to-chassis leg-length
        # proxies at physical flight onset. This makes the target reachable for
        # the current mechanism and prevents one wheel from earning a bilateral
        # tuck payment by itself.
        self.release_leg_length = torch.zeros_like(z)
        self.release_leg_lengths = torch.zeros(z.shape[0], 2, device=z.device, dtype=z.dtype)
        self.max_bilateral_tuck_delta = torch.zeros_like(z)
        self.apex_hold_time = torch.zeros_like(z)
        self.apex_seen = torch.zeros_like(z,dtype=torch.bool)
        self.extension_started = torch.zeros_like(z,dtype=torch.bool)
        self.last_air_alignment = torch.zeros_like(z)
        self.last_air_tuck = torch.zeros_like(z)
        self.last_air_extension = torch.zeros_like(z)
        self.landing_contact_time = torch.zeros_like(z)
        self.landing_buffer_time = torch.zeros_like(z)
        self.landing_buffer_h0 = torch.zeros_like(z)
        self.landing_buffer_v0 = torch.zeros_like(z)
        self.landing_buffer_acc = torch.zeros_like(z)
        self.landing_buffer_duration = torch.zeros_like(z)
        self.last_landing_buffer_height = torch.zeros_like(z)
        self.last_landing_buffer_velocity = torch.zeros_like(z)
        for name in ("tuck_seen", "landing_buffer_active", "landing_buffer_used", "flown", "landed", "succeeded", "failed", "liftoff_event", "landing_event", "success_event", "timeout_event"):
            setattr(self, name, torch.zeros_like(z, dtype=torch.bool))

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor) and value is not self.weight:
                value[ids] = 0

    def begin_interval(self):
        for name in ("liftoff_event", "landing_event", "success_event", "timeout_event", "interval_force_peak"):
            getattr(self, name).zero_()

    def observe(self, age, height, velocity, gravity, wheel_clearance, wheel_force, nonwheel_force,
                horizontal_speed_error=None, leg_lengths=None):
        j, s = self.j, self.schedule
        jump, drop = s.mask("jump"), s.mask("airborne")
        eligible = drop | (jump & (age >= s.delay))
        air = (wheel_clearance > j["clearance_m"]).all(-1) & (wheel_force <= j["force_threshold_n"]).all(-1)
        support = (wheel_clearance < j["landing_clearance_m"]).all(-1) & (wheel_force > j["force_threshold_n"]).all(-1)
        v4 = j.get("reward_version",1) >= 4
        first_air = eligible & air & (self.air_time <= 0.) & ~self.flown
        # A failed provisional flight must not donate a convenient release
        # baseline or a previously compressed leg to the next real takeoff.
        lost_candidate = eligible & ~air & ~self.flown
        if v4:
            for name in ("release_leg_length", "max_bilateral_tuck_delta"):
                value = getattr(self,name)
                value[lost_candidate] = 0.
            self.release_leg_lengths[lost_candidate] = 0.
        self.air_time = torch.where(eligible & air, self.air_time+self.dt, 0.)
        liftoff = (jump & ~self.flown & (self.air_time >= j["minimum_air_seconds"]-1e-7)
                   & (velocity[:, 2] > j["minimum_release_vz_m_s"]))
        self.liftoff_event |= liftoff
        new_flight = liftoff | (drop & (self.air_time >= j["minimum_air_seconds"]-1e-7))
        self.flown |= new_flight
        if v4:
            in_flight = eligible & self.flown & ~self.landed
            new_apex = in_flight & ~self.apex_seen & (velocity[:,2] <= j["flight_ascent_vz_threshold_m_s"])
            self.apex_hold_time = torch.where(in_flight & self.apex_seen,
                self.apex_hold_time+self.dt,self.apex_hold_time)
            self.apex_seen |= new_apex
            self.extension_started |= (in_flight & self.apex_seen
                & (self.apex_hold_time >= j["apex_tuck_hold_seconds"]-1e-7))
        touching = (wheel_clearance < j["landing_clearance_m"]).any(-1) & (wheel_force > j["force_threshold_n"]).any(-1)
        # Mirror WheelBipe's landing pre-contact window: contact must persist
        # briefly before a reward-only quadratic reference is created. The
        # policy remains the sole controller; no state or force is written.
        contact_window = self.flown & ~self.landing_buffer_used
        self.landing_contact_time = torch.where(
            contact_window & touching,
            self.landing_contact_time + self.dt,
            torch.where(contact_window, torch.zeros_like(self.landing_contact_time),
                        self.landing_contact_time),
        )
        duration = max(float(j.get("landing_buffer_duration_s", .3)), self.dt)
        active_buffer = self.landing_buffer_active
        next_buffer_time = torch.where(
            active_buffer,
            torch.minimum(self.landing_buffer_time + self.dt,
                          torch.full_like(self.landing_buffer_time, duration)),
            self.landing_buffer_time,
        )
        buffer_done = active_buffer & (next_buffer_time >= duration-1e-7)
        self.landing_buffer_time = next_buffer_time
        self.landing_buffer_active &= ~buffer_done
        target_height = float(j.get("landing_buffer_target_height_m", .24))
        can_start_buffer = (
            contact_window & touching
            & ~self.landing_buffer_active
            & (self.landing_contact_time >= float(j.get("landing_buffer_start_contact_s", .02))-1e-7)
            & (velocity[:, 2] < -float(j.get("landing_buffer_min_down_vz_m_s", .2)))
            & (height > target_height+float(j.get("landing_buffer_min_height_margin_m", .02)))
        )
        end_vz = min(float(j.get("landing_buffer_end_vz_m_s", 0.)), 0.)
        buffer_duration = torch.full_like(self.landing_buffer_duration, duration)
        displacement = target_height-height
        ref_v0 = 2.*displacement/duration-end_vz
        acceleration = (end_vz-ref_v0)/duration
        self.landing_buffer_active |= can_start_buffer
        self.landing_buffer_used |= can_start_buffer
        self.landing_buffer_time = torch.where(can_start_buffer,
                                               torch.zeros_like(self.landing_buffer_time),
                                               self.landing_buffer_time)
        self.landing_buffer_h0 = torch.where(can_start_buffer, height, self.landing_buffer_h0)
        self.landing_buffer_v0 = torch.where(can_start_buffer, ref_v0, self.landing_buffer_v0)
        self.landing_buffer_acc = torch.where(can_start_buffer, acceleration, self.landing_buffer_acc)
        self.landing_buffer_duration = torch.where(can_start_buffer, buffer_duration,
                                                   self.landing_buffer_duration)
        if leg_lengths is not None:
            if leg_lengths.shape != (len(self.weight), 2):
                raise ValueError("leg_lengths must have shape (num_envs, 2)")
            # Capture the geometry at the first measured flight sample.  The
            # observer is the only owner of this state; the command reference
            # remains a function of the request clock.
            capture = (first_air if v4 else new_flight) & (self.release_leg_length <= 0.)
            self.release_leg_lengths = torch.where(capture[:, None], leg_lengths.detach(),
                                                   self.release_leg_lengths)
            self.release_leg_length = torch.where(capture, leg_lengths.detach().mean(-1),
                                                  self.release_leg_length)
            in_flight = self.flown & ~self.landed & eligible
            release = torch.where(self.release_leg_lengths > 0., self.release_leg_lengths,
                                  leg_lengths.detach())
            bilateral_delta = (release-leg_lengths.detach()).clamp_min(0.).amin(-1)
            measure_tuck = (eligible & air & ~self.landed) if v4 else in_flight
            self.max_bilateral_tuck_delta = torch.where(
                measure_tuck,
                torch.maximum(self.max_bilateral_tuck_delta, bilateral_delta),
                self.max_bilateral_tuck_delta,
            )
            minimum_tuck = float(j.get("minimum_bilateral_tuck_delta_m", 0.04))
            measured = self.max_bilateral_tuck_delta if v4 else bilateral_delta
            self.tuck_seen |= in_flight & (measured >= minimum_tuck)
        self.peak_height = torch.where(eligible, torch.maximum(self.peak_height, height), self.peak_height)
        landing = eligible & self.flown & ~self.landed & touching
        self.landing_event |= landing
        self.landed |= landing
        self.force_peak = torch.maximum(self.force_peak, wheel_force.sum(-1)*eligible)
        self.interval_force_peak = torch.maximum(self.interval_force_peak, wheel_force.sum(-1)*eligible)
        speed_error = velocity[:, :2].norm(dim=-1) if horizontal_speed_error is None else horizontal_speed_error
        calm = ((gravity[:, 2] < -math.cos(math.radians(12))) & (speed_error < .15)
                & (velocity[:, 2].abs() < .15) & ((height-j["standing_height_m"]).abs() < .035)
                & (nonwheel_force <= 2.))
        stable = eligible & self.landed & support & calm
        self.stable_time = torch.where(stable, self.stable_time+self.dt, 0.)
        apex_ok = drop | (self.peak_height >= s.apex-.025)
        tuck_required = bool(j.get("success_requires_tuck", False))
        # Airborne is a passive drop diagnostic and must not be forced through
        # the active-jump tuck gate.  If a caller omits leg geometry, retain
        # the historical observer behavior for v1-v3. V4 fails closed if jump
        # geometry is omitted; the native environment always supplies it.
        if leg_lengths is not None:
            tuck_required_mask = jump if tuck_required else torch.zeros_like(jump)
            tuck_ok = (~tuck_required_mask) | self.tuck_seen
        elif v4:
            tuck_ok = ~jump if tuck_required else torch.ones_like(jump)
        else:
            tuck_ok = torch.ones_like(self.succeeded)
        success = (eligible & ~self.succeeded & ~self.failed & apex_ok
                    & (self.stable_time >= j["landing_hold_seconds"]-1e-7) & tuck_ok)
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
        self.last_air_alignment.zero_()
        self.last_air_tuck.zero_()
        self.last_air_extension.zero_()
        self.last_landing_buffer_height.zero_()
        self.last_landing_buffer_velocity.zero_()
        landing_buffer_payment = torch.zeros_like(height)
        buffer_active = torch.zeros_like(active, dtype=torch.bool)
        buffer_height_ref = height
        if j.get("reward_version", 1) >= 3:
            # Record geometry once if this observer is being used standalone;
            # the native environment captures it during the physics observer.
            leg_mean = legs.mean(-1)
            first_flight = flight & (self.release_leg_length <= 0.)
            self.release_leg_lengths = torch.where(first_flight[:, None], legs.detach(),
                                                   self.release_leg_lengths)
            self.release_leg_length = torch.where(first_flight, leg_mean.detach(),
                                                  self.release_leg_length)
            release = torch.where(self.release_leg_lengths > 0., self.release_leg_lengths,
                                  leg_mean.detach()[:, None])
            bilateral_delta = (release-legs).clamp_min(0.).amin(-1)
            tuck_target = float(j.get("tuck_target_delta_m", .08))
            tuck_sigma = float(j.get("tuck_sigma_m", .04))
            progress = (bilateral_delta/tuck_target).clamp(0., 1.)
            target_shape = torch.exp(-((bilateral_delta-tuck_target)/tuck_sigma).square())
            tuck = .5*progress+.5*target_shape
            if j.get("reward_version",1) >= 4:
                # Early asymmetric progress must receive a gradient, while
                # the event gate continues to require both legs together.
                per_leg_delta = (release-legs).clamp_min(0.)
                per_leg = (.5*(per_leg_delta/tuck_target).clamp(0.,1.)
                    +.5*torch.exp(-((per_leg_delta-tuck_target)/tuck_sigma).square())).mean(-1)
                weight = j["per_leg_progress_weight"]
                tuck = (1.-weight)*tuck+weight*per_leg
            extension_target = float(j.get("landing_extension_delta_m", .03))
            extension_sigma = float(j.get("landing_extension_sigma_m", .03))
            extension = torch.exp(-((bilateral_delta-extension_target)/extension_sigma).square())
            ascent = vz >= float(j.get("flight_ascent_vz_threshold_m_s", 0.))
            # Keep the tucked geometry through the apex and for a short
            # measured hold.  Only after that phase does the descent payment
            # prefer the smaller residual tuck used for landing.
            tuck_phase = ascent | (self.air_time < float(j.get("tuck_hold_seconds", .1)))
            if j.get("reward_version",1) >= 4:
                # Latched in the physical observer; continuous-air resets or
                # a noisy upward rebound cannot restart the tuck stage.
                tuck_phase = ~self.extension_started
            air_alignment_scale = float(j.get("air_alignment_scale", .25))
            air_tuck_scale = float(j.get("air_tuck_scale", 1.))
            air_extension_scale = float(j.get("air_extension_scale", .5))
            self.last_air_alignment = flight.float()*placement
            self.last_air_tuck = flight.float()*tuck_phase.float()*tuck
            self.last_air_extension = flight.float()*(~tuck_phase).float()*extension
            placement = (air_alignment_scale*placement
                         +air_tuck_scale*tuck_phase.float()*tuck
                         +air_extension_scale*(~tuck_phase).float()*extension)
            buffer_active = self.landing_buffer_active & active
            buffer_duration = self.landing_buffer_duration.clamp_min(policy_dt)
            buffer_t = torch.minimum(self.landing_buffer_time.clamp_min(0.), buffer_duration)
            buffer_height_ref = (self.landing_buffer_h0
                                 +self.landing_buffer_v0*buffer_t
                                 +.5*self.landing_buffer_acc*buffer_t.square())
            buffer_velocity_ref = self.landing_buffer_v0+self.landing_buffer_acc*buffer_t
            height_sigma = float(j.get("landing_buffer_height_sigma_m", .01))
            velocity_sigma = float(j.get("landing_buffer_velocity_sigma_m_s", .25))
            self.last_landing_buffer_height = buffer_active.float()*torch.exp(
                -((height-buffer_height_ref)/height_sigma).square())
            self.last_landing_buffer_velocity = buffer_active.float()*torch.exp(
                -((vz-buffer_velocity_ref)/velocity_sigma).square())
            landing_buffer_payment = (
                float(j.get("landing_buffer_height_scale", 1.)) * self.last_landing_buffer_height
                +float(j.get("landing_buffer_velocity_scale", .5)) * self.last_landing_buffer_velocity
            )
        else:
            target_leg = torch.where(vz > 0, .18, .25)
            placement += torch.exp(-((legs-target_leg[:, None])/.06).square()).mean(-1)
        bonuses = j["event_bonuses"]
        events = (bonuses["liftoff"]*self.liftoff_event+bonuses["landing"]*self.landing_event
                  +bonuses["success"]*self.success_event+bonuses["timeout"]*self.timeout_event)
        if j.get("reward_version", 1) >= 2:
            # Center preparation tracking on an idle robot. Waiting at standing
            # height must never earn the jump reward after a missed push.
            idle_error = j["standing_height_m"]-ref["height"]
            tracking = (torch.exp(-(height_error/.035).square())
                        -torch.exp(-(idle_error/.035).square()))
            preparing = active & (ref["crouch"] | ref["push"])
            recovery = active & self.landed
            recovery_height_error = torch.where(buffer_active,
                                                height-buffer_height_ref,
                                                height_error)
            missed = active & ref["after_push"] & ~self.flown
            velocity_gain = (torch.exp(-((vz-ref["velocity"])/.5).square())
                             -torch.exp(-(ref["velocity"]/.5).square()))
            progress = (vz/ref["release_velocity"]).clamp(-1., 1.)
            return dict(
                jump_height=j["dense_scale"]*(4*preparing*tracking
                    +recovery*(2*torch.exp(-(recovery_height_error/.035).square())-1.)
                    +landing_buffer_payment
                    -j["missed_takeoff_cost_per_second"]*missed),
                jump_push_velocity=j["dense_scale"]*ref["push"]*(6*velocity_gain+4*progress),
                jump_air_placement=j["dense_scale"]*2*flight*placement,
                jump_impact=-j["dense_scale"]*((self.interval_force_peak/self.weight-j["impact_soft_bodyweights"])
                    .clamp_min(0.)/j["impact_scale_bodyweights"]).clamp_max(5.),
                jump_events=events/policy_dt)
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

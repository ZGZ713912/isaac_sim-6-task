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
PLATFORM_TASKS = TASKS + ("platform",)
TERRAIN_TASKS = PLATFORM_TASKS + ("slope", "step_down", "bump", "balls")
REWARD_NAMES = ("jump_height", "jump_push_velocity", "jump_air_placement", "jump_impact", "jump_events")
PLATFORM_REWARD_NAMES = ("platform_clearance", "platform_events", "platform_impact")
# Platform episode modes: locked lanes run a flat moving jump far from their box.
FLAT, OVER, DRIVE_OFF = 0, 1, 2


def tasks_for(spec):
    """Contracts without an explicit list keep the historical nine-task order."""
    names = tuple(spec.get("tasks", TASKS))
    if names not in (TASKS, PLATFORM_TASKS, TERRAIN_TASKS):
        raise ValueError("Unsupported mixed task list")
    return names


def reward_names(spec):
    sequence = ("jump_release_energy", "jump_clearance") if spec["jump"].get("reward_version", 1) >= 5 else ()
    return (REWARD_NAMES+sequence+(PLATFORM_REWARD_NAMES if "platform" in tasks_for(spec) else ())
        +tuple(spec.get("terrain_reward_weights", {})))


def _validate_extensions(spec):
    arc = spec.get("arc_commands")
    if arc is not None:
        if (not 0. < arc["probability"] <= 1. or not 0. < arc["min_yaw_rad_s"]
                or not 0. < arc["max_centripetal_m_s2"] or not 0. <= arc["min_speed_m_s"]):
            raise ValueError("Invalid arc command sampling")
    moving = spec["jump"].get("moving_jump", {})
    if "speed_range_m_s" in moving:
        low, high = moving["speed_range_m_s"]
        if not 0. < low <= high or not 0. < moving["fraction"] <= 1.:
            raise ValueError("Invalid moving-jump speed range or fraction")
    if "platform" in tasks_for(spec):
        p = spec["platform"]
        heights = p["lane_heights_m"]
        unlock = p["unlock"]
        if (not heights or min(heights) <= 0. or max(heights) > .3 or p["length_m"] <= .5
                or p["width_m"] <= .6 or not 0. <= p["over_fraction"] <= 1.
                or not 0. < p["speed_range_m_s"][0] <= p["speed_range_m_s"][1]
                or unlock["start_update"] > unlock["full_update"]
                or unlock["start_height_m"] > unlock["final_height_m"]
                or p["episode_seconds"] <= spec["jump"]["outcome_seconds"]
                or not 0. < p["request_deadline_s"] < spec["jump"]["outcome_seconds"]):
            raise ValueError("Invalid platform curriculum")
        r = p["step_reference"]
        j = spec["jump"]
        if (not j["release_height_m"] <= r["max_release_m"] <= .438 or r["max_stance_m"] > r["max_release_m"]
                or min(r["armor_below_root_m"], r["stance_margin_m"], r["crouch_margin_m"], r["push_stroke_m"],
                       r["raise_seconds"], r["min_crouch_seconds"]) <= 0. or r["hold_seconds"] < 0.
                or r["stance_margin_m"] < r["crouch_margin_m"] or r["apex_min_m"] <= j["release_height_m"]):
            raise ValueError("Invalid step-jump reference")


def validate(spec):
    fractions = spec["task_fractions"]
    tasks = tasks_for(spec)
    if set(fractions) != set(tasks) or any(not math.isfinite(v) or v <= 0 for v in fractions.values()):
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
    if j.get("reward_version", 1) >= 5:
        from .jump_objectives import validate_objective
        validate_objective(j)
    _validate_extensions(spec)
    if "slope" in tasks:
        from .terrain_curriculum import validate_terrain
        validate_terrain(spec)
    return spec


def quota_counts(count, spec):
    validate(spec)
    tasks = tasks_for(spec)
    if count < spec["sampling"]["minimum_envs"]:
        raise ValueError("Mixed task population is too small to cover every quota")
    exact = [count*spec["task_fractions"][name] for name in tasks]
    result = [math.floor(x) for x in exact]
    order = sorted(range(len(tasks)), key=lambda i: (-(exact[i]-result[i]), i))
    for i in order[:count-sum(result)]:
        result[i] += 1
    if min(result) < 1:
        raise ValueError("Every task requires at least one environment")
    return dict(zip(tasks, result))


def jump_timing(apex, j):
    """Release velocity, push duration and request-to-apex time of the time-only reference."""
    release_v = (2*9.81*(apex-j["release_height_m"])).sqrt()
    push = 2*(j["release_height_m"]-j["crouch_height_m"])/release_v
    return release_v, push, j["crouch_seconds"]+push+release_v/9.81


def step_profile(rise, spec):
    """Time-only step-jump reference for a declared rise (tensor, m).

    The front/rear armor sits 0.143 m below the root, 0.29 m ahead of the
    axle, so it reaches the face first. Raise the stance until it clears the
    rise, crouch only as far as it stays clear, push from near-full extension
    and require the apex that lets tucked wheels pass over the edge.
    """
    j, s = spec["jump"], spec["platform"]["step_reference"]
    standing = j["standing_height_m"]
    armor = rise+s["armor_below_root_m"]
    stance = (armor+s["stance_margin_m"]).clamp(min=standing, max=s["max_stance_m"])
    crouch = (armor+s["crouch_margin_m"]).clamp(min=j["crouch_height_m"])
    crouch = torch.minimum(crouch, stance)
    release = torch.maximum(torch.full_like(rise, j["release_height_m"]),
        (crouch+s["push_stroke_m"]).clamp(max=s["max_release_m"]))
    apex = torch.maximum(torch.maximum(torch.full_like(rise, s["apex_min_m"]), release+.02),
        rise+s["apex_above_rise_m"])
    raised = stance > standing+1e-6
    raise_s = torch.where(raised, s["raise_seconds"], 0.)
    hold_s = torch.where(raised, s["hold_seconds"], 0.)
    crouch_s = (j["crouch_seconds"]*(stance-crouch)/(standing-j["crouch_height_m"])).clamp(
        min=s["min_crouch_seconds"], max=j["crouch_seconds"])
    release_v = (2*9.81*(apex-release)).sqrt()
    push = 2*(release-crouch)/release_v
    return dict(stance=stance, crouch=crouch, release=release, apex=apex, raise_s=raise_s,
                hold_s=hold_s, crouch_s=crouch_s, release_v=release_v,
                to_apex=raise_s+hold_s+crouch_s+push+release_v/9.81)


class TaskSchedule:
    """One fixed task per lane; resets cannot overwrite another task's quota."""
    def __init__(self, count, device, seed, spec):
        self.spec, self.device = validate(spec), device
        self.tasks = tasks_for(spec)
        self.generator = torch.Generator(device=device).manual_seed(int(seed)+54173)
        counts = quota_counts(count, spec)
        roles = torch.cat([torch.full((counts[n],), i, device=device, dtype=torch.long)
                           for i, n in enumerate(self.tasks)])
        self.role = roles[torch.randperm(count, device=device, generator=self.generator)]
        self.commands = torch.zeros(count, 3, device=device)
        self.height = torch.full((count,), spec["jump"]["standing_height_m"], device=device)
        self.next_sample = torch.zeros(count, device=device)
        self.delay = torch.zeros_like(self.height)
        self.apex = torch.zeros_like(self.height)
        self.drop = torch.zeros_like(self.height)
        self.wave_phase = torch.zeros_like(self.height)
        self.update = 0
        self.has_platform = "platform" in self.tasks
        if self.has_platform:
            # Lane geometry is fixed at scene construction: one static box per
            # platform lane, its height cycled from the contract list by rank.
            p = spec["platform"]
            lane = self.mask("platform")
            self.platform_rank = torch.where(lane, lane.long().cumsum(0)-1, -1)
            heights = torch.tensor(p["lane_heights_m"], device=device)
            self.rise = torch.where(lane, heights[self.platform_rank.clamp_min(0) % len(heights)], 0.)
            self.mode = torch.full((count,), FLAT, device=device, dtype=torch.long)
            self.platform_speed = torch.zeros_like(self.height)
            self.trigger_distance = torch.full_like(self.height, 1e9)
            self.odometry = torch.zeros_like(self.height)
            self.start_x = torch.zeros_like(self.height)
            self.height_limit = 0.
            # Per-lane time-only reference; flat-jump constants except on
            # step-up episodes (see step_profile).
            j = spec["jump"]
            self.ref_stance = torch.full_like(self.height, j["standing_height_m"])
            self.ref_crouch = torch.full_like(self.height, j["crouch_height_m"])
            self.ref_release = torch.full_like(self.height, j["release_height_m"])
            self.ref_raise_s = torch.zeros_like(self.height)
            self.ref_hold_s = torch.zeros_like(self.height)
            self.ref_crouch_s = torch.full_like(self.height, j["crouch_seconds"])
        self.terrain = None
        if "slope" in self.tasks:
            from .terrain_curriculum import TerrainSchedule
            self.terrain = TerrainSchedule(self)
        self.reset(torch.arange(count, device=device), 0)

    def mask(self, name):
        if name not in self.tasks:
            return torch.zeros_like(self.role, dtype=torch.bool)
        return self.role == self.tasks.index(name)

    def requesters(self):
        """Lanes that receive the explicit jump request (identical to jump lanes without platforms)."""
        selected = self.mask("jump") | self.mask("platform")
        if self.terrain is not None:
            from .terrain_curriculum import JUMP
            selected |= self.mask("slope") & (self.terrain.mode == JUMP)
        return selected

    def random(self, count):
        return torch.rand(count, device=self.device, generator=self.generator)

    def moving_ramp(self):
        moving = self.spec["jump"].get("moving_jump")
        if not moving:
            return 1.
        return min(max((self.update-moving["start_update"])/
            (moving["full_update"]-moving["start_update"]), 0.), 1.)

    def platform_height_limit(self, update):
        u = self.spec["platform"]["unlock"]
        if update < u["start_update"]:
            return 0.
        progress = min(max((update-u["start_update"])/max(u["full_update"]-u["start_update"], 1), 0.), 1.)
        return u["start_height_m"]+progress*(u["final_height_m"]-u["start_height_m"])

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
        if self.has_platform:
            self._reset_platform(ids[self.mask("platform")[ids]], update)
        if self.terrain is not None:
            self.terrain.reset(ids, update)
        evaluation = self.spec.get("evaluation")
        if evaluation:
            selected = ids[self.mask(evaluation["target"])[ids]]
            cell = evaluation["cell"]
            self.height[selected] = cell["height"]
            if self.terrain is not None:
                if "yaw_deg" in cell and cell["task"] != "bump":
                    self.terrain.yaw[selected] = math.radians(cell["yaw_deg"])
                if "rpm" in cell:
                    self.terrain.sign[selected] = 1. if cell["rpm"] >= 0 else -1.
                if cell.get("mode") == "jump":
                    from .terrain_curriculum import JUMP
                    self.terrain.mode[selected] = JUMP
                    a,b = self.spec["slope"]["jump"]["request_delay_s"]
                    self.delay[selected] = a+(b-a)*self.random(len(selected))
                    self.terrain.speed[selected] = cell["speed"]

    def _reset_platform(self, ids, update):
        """Episode mode, speed, reference, run-up and start pose; terrain never drives the request."""
        if not len(ids):
            return
        p, j = self.spec["platform"], self.spec["jump"]
        n = len(ids)
        self.height_limit = self.platform_height_limit(update)
        unlocked = self.rise[ids] <= self.height_limit+1e-6
        over = self.random(n) < p["over_fraction"]
        mode = torch.where(unlocked, torch.where(over, OVER, DRIVE_OFF), FLAT)
        low, high = p["speed_range_m_s"]
        speed = low+(high-low)*self.random(n)
        a, b = p["runup_seconds"]
        runup = speed*(a+(b-a)*self.random(n))
        jitter = speed*p["timing_jitter_s"]*(2*self.random(n)-1)
        lo, hi = p["drive_off_start_m"]
        before_far = lo+(hi-lo)*self.random(n)
        step = step_profile(self.rise[ids], self.spec)
        is_over = mode == OVER
        near, wheel = p["near_edge_x_m"], p["wheel_center_offset_x_m"]
        # At the reference apex the wheel centre is half a radius before the
        # near face; the raised stance has already carried the armor over it.
        over_x = near-.5*j["wheel_radius_m"]-wheel-speed*step["to_apex"]-runup+jitter
        self.start_x[ids] = torch.where(is_over, over_x,
            torch.where(mode == DRIVE_OFF, near+p["length_m"]-wheel-before_far, near-p["flat_offset_m"]))
        # Step-up: odometry run-up. Locked lanes keep the time-based flat-jump
        # request already sampled in reset(); drive-off never requests.
        self.trigger_distance[ids] = torch.where(is_over, runup, 1e9)
        self.delay[ids] = torch.where(mode == FLAT, self.delay[ids], 1e3)
        self.mode[ids] = mode
        self.platform_speed[ids] = speed
        self.apex[ids] = torch.where(is_over, step["apex"], self.apex[ids])
        for name in ("stance", "crouch", "release", "raise_s", "hold_s", "crouch_s"):
            flat = {"stance": j["standing_height_m"], "crouch": j["crouch_height_m"],
                    "release": j["release_height_m"], "raise_s": 0., "hold_s": 0.,
                    "crouch_s": j["crouch_seconds"]}[name]
            getattr(self, "ref_"+name)[ids] = torch.where(is_over, step[name], flat)
        self.odometry[ids] = 0.

    def advance_platform_requests(self, age, distance):
        """Explicit step-up request from wheel odometry along the commanded run-up.

        A lane that has not covered its run-up by the deadline is requested
        anyway, so standing still never avoids the step-up outcome.
        """
        if not self.has_platform:
            return
        over = self.mask("platform") & (self.mode == OVER)
        self.odometry += torch.where(over, distance, 0.)
        fire = over & (self.delay > 999.) & ((self.odometry >= self.trigger_distance)
            | (age >= self.spec["platform"]["request_deadline_s"]))
        self.delay = torch.where(fire, age, self.delay)

    def support_height(self, xy):
        """Ground height under env-local points (..., 2); reward/acceptance only."""
        if not self.has_platform:
            return torch.zeros(xy.shape[:-1], device=xy.device)
        p = self.spec["platform"]
        near = p["near_edge_x_m"]
        inside = ((xy[..., 0] >= near) & (xy[..., 0] <= near+p["length_m"])
                  & (xy[..., 1].abs() <= .5*p["width_m"]))
        rise = self.rise.reshape(-1, *([1]*(xy.dim()-2)))
        lane = self.mask("platform").reshape_as(rise)
        value = torch.where(inside & lane, rise, 0.)
        if self.terrain is not None:
            value = value+self.terrain.support_height(xy)
        return value

    def update_commands(self, age, phase):
        due = age >= self.next_sample
        if self.terrain is not None:
            due &= ~self.terrain.terrain_lane
        ids = due.nonzero().flatten()
        if len(ids):
            index = self.tasks.index
            r = self.role[ids]
            n, vmax, yawmax = len(ids), phase["vx_max_m_s"], phase["yaw_max_rad_s"]
            speed = .1+(max(vmax, .1)-.1)*self.random(n)
            low = self.random(n) < self.spec["sampling"]["low_speed_probability"]
            lo, hi = self.spec["sampling"]["low_speed_range_m_s"]
            speed = torch.where(low, lo+(min(hi, vmax)-lo)*self.random(n), speed)
            cmd = torch.zeros(n, 3, device=self.device)
            cmd[:, 0] = torch.where(r == index("forward"), speed, 0.)
            cmd[:, 0] = torch.where(r == index("reverse"), -speed, cmd[:, 0])
            # Alternate signs by lane and sample generation, preserving both
            # directions even when a task contains only one or two lanes.
            generation = torch.floor(age[ids]/self.spec["sampling"]["command_hold_seconds"]).long()
            sign = torch.where((ids+generation) % 2 == 0, 1., -1.)
            turn = r == index("turn")
            travel_sign = torch.where((ids//2+generation//2) % 2 == 0, 1., -1.)
            cmd[:, 0] = torch.where(turn, travel_sign*.2, cmd[:, 0])
            cmd[:, 2] = torch.where(turn, sign*(.25+max(yawmax-.25, 0.)*self.random(n)), 0.)
            dash = r == index("dash")
            dash_sign = torch.where((ids+generation//2) % 2 == 0, 1., -1.)
            sprint = generation % 2 == 0
            cmd[:, 0] = torch.where(dash & sprint, dash_sign*vmax*(.75+.25*self.random(n)), cmd[:, 0])
            spin = r == index("spin")
            # Retain the original late-course spin range, including 100 rpm.
            spin_starts = self.spec["special_mode_starts"]
            spinmax = 4.5*math.pi if phase.get("completed_update", 0) >= spin_starts["spin_mid"] else yawmax
            if phase["start_update"] >= spin_starts["spin_low"]:
                spinmax = max(spinmax, 3.25*math.pi)
            cmd[:, 2] = torch.where(spin, sign*spinmax*(.5+.5*self.random(n)), cmd[:, 2])
            arc = self.spec.get("arc_commands")
            if arc:
                # Arcs: the same speed samples plus a yaw rate whose centripetal
                # acceleration stays below the configured wheel-slip margin.
                lanes = (r == index("forward")) | (r == index("reverse")) | dash
                vx = cmd[:, 0].abs()
                limit = torch.minimum(torch.full_like(vx, yawmax), arc["max_centripetal_m_s2"]/vx.clamp_min(1e-3))
                pick = (lanes & (self.random(n) < arc["probability"]) & (vx >= arc["min_speed_m_s"])
                        & (limit > arc["min_yaw_rad_s"]))
                yaw_sign = torch.where(self.random(n) < .5, 1., -1.)
                yaw = yaw_sign*(arc["min_yaw_rad_s"]+(limit-arc["min_yaw_rad_s"]).clamp_min(0.)*self.random(n))
                cmd[:, 2] = torch.where(pick, yaw, cmd[:, 2])
            self.commands[ids] = cmd
            hlow, hhigh = phase["height_range_m"]
            self.height[ids] = hlow+(hhigh-hlow)*self.random(n)
            jump = (r == index("jump")) | (r == index("airborne"))
            self.height[ids[jump]] = self.spec["jump"]["standing_height_m"]
            moving = self.spec["jump"].get("moving_jump")
            if moving:
                ramp = self.moving_ramp()
                rank = (self.mask("jump").long().cumsum(0)-1)[ids]
                direction = torch.where((rank//2+generation) % 2 == 0, 1., -1.)
                if "speed_range_m_s" in moving:
                    fraction = moving["fraction"]
                    travel = (torch.floor((rank+1)*fraction) > torch.floor(rank*fraction)) & (r == index("jump"))
                    a, b = moving["speed_range_m_s"]
                    jump_speed = (a+(b-a)*self.random(n))*ramp
                else:
                    travel = (rank % 2 == 0) & (r == index("jump"))
                    jump_speed = moving["maximum_vx_m_s"]*ramp
                self.commands[ids, 0] = torch.where(travel, direction*jump_speed, self.commands[ids, 0])
            hold = torch.where(r == index("stand"),
                self.spec["sampling"]["standing_command_hold_seconds"], self.spec["sampling"]["command_hold_seconds"])
            if self.has_platform:
                plat = r == index("platform")
                platform_speed = self.platform_speed[ids]
                platform_speed = torch.where(self.mode[ids] == FLAT, platform_speed*self.moving_ramp(), platform_speed)
                self.commands[ids] = torch.where(plat[:, None], 0., self.commands[ids])
                self.commands[ids, 0] = torch.where(plat, platform_speed, self.commands[ids, 0])
                self.height[ids[plat]] = self.spec["jump"]["standing_height_m"]
                hold = torch.where(plat, 1e3, hold)
            self.next_sample[ids] = age[ids]+hold
        if self.terrain is not None:
            self.terrain.update_commands(age)
        evaluation = self.spec.get("evaluation")
        if evaluation:
            selected = self.mask(evaluation["target"])
            cell = evaluation["cell"]
            if cell.get("command") is not None:
                self.commands[selected] = self.commands.new_tensor(cell["command"])
            elif cell["task"] == "jump":
                self.commands[selected] = 0.
            self.height[selected] = cell["height"]
            if cell.get("mode") == "height":
                self.height[selected] = .32+.06*torch.sin(age[selected]*math.pi)
            if cell.get("mode") == "brake":
                self.commands[selected] = 0.
                self.commands[selected,0] = torch.where(age[selected] < 2.5,.6,0.)
        hlow, hhigh = phase["height_range_m"]
        wave = (hlow+hhigh)/2+(hhigh-hlow)/2*torch.sin(
            2*math.pi*self.spec["sampling"]["height_wave_hz"]*age+self.wave_phase)
        return self.commands, torch.where(self.mask("height"), wave, self.height)

    def reference(self, age):
        if self.has_platform:
            return self._staged_reference(age)
        j = self.spec["jump"]
        jumper = self.requesters()
        t = age-self.delay
        release_v = (2*9.81*(self.apex-j["release_height_m"])).sqrt()
        duration = 2*(j["release_height_m"]-j["crouch_height_m"])/release_v
        push_time = (t-j["crouch_seconds"]).clamp_min(0.)
        crouch = jumper & (t >= 0) & (t < j["crouch_seconds"])
        push = jumper & (t >= j["crouch_seconds"]) & (push_time <= duration)
        v2 = j.get("reward_version", 1) >= 2
        flight_seconds = 2*release_v/9.81 if v2 else 1.
        flight = jumper & (push_time > duration) & (push_time < duration+flight_seconds)
        active = ((jumper & (t >= 0)) | self.mask("airborne")) & (age < j["outcome_seconds"])
        p = (t/j["crouch_seconds"]).clamp(0., 1.)
        h = torch.full_like(age, j["standing_height_m"])
        h = torch.where(crouch, h+(j["crouch_height_m"]-h)*(3*p.square()-2*p.pow(3)), h)
        acceleration = release_v/duration
        h = torch.where(push, j["crouch_height_m"]+.5*acceleration*push_time.square(), h)
        if v2:
            h = torch.where(flight, j["release_height_m"], h)
        v = torch.where(push, acceleration*push_time, 0.)
        return dict(height=h, velocity=v, push=push, flight=flight, active=active,
                    crouch=crouch, after_push=jumper & (push_time > duration),
                    release_velocity=release_v, requested=jumper & (t >= 0))

    def _staged_reference(self, age):
        """Per-lane raise -> hold -> crouch -> push -> flight; flat lanes have no raise/hold.

        With flat constants this equals the historical reference. ``crouch``
        marks every pre-push preparation stage so tracking pays the raise too.
        """
        j = self.spec["jump"]
        jumper = self.requesters()
        t = age-self.delay
        stance, low, release = self.ref_stance, self.ref_crouch, self.ref_release
        release_v = (2*9.81*(self.apex-release)).sqrt()
        duration = 2*(release-low)/release_v
        hold_end = self.ref_raise_s+self.ref_hold_s
        push_start = hold_end+self.ref_crouch_s
        push_time = (t-push_start).clamp_min(0.)
        prepare = jumper & (t >= 0) & (t < push_start)
        push = jumper & (t >= push_start) & (push_time <= duration)
        flight = jumper & (push_time > duration) & (push_time < duration+2*release_v/9.81)
        active = ((jumper & (t >= 0)) | self.mask("airborne")) & (age < j["outcome_seconds"])
        smooth = lambda x: 3*x.square()-2*x.pow(3)
        rise = smooth((t/self.ref_raise_s.clamp_min(1e-6)).clamp(0., 1.))
        down = smooth(((t-hold_end)/self.ref_crouch_s).clamp(0., 1.))
        standing = j["standing_height_m"]
        h = torch.where(t < hold_end, standing+(stance-standing)*rise, stance+(low-stance)*down)
        h = torch.where(prepare, h, torch.full_like(age, standing))
        acceleration = release_v/duration
        h = torch.where(push, low+.5*acceleration*push_time.square(), h)
        h = torch.where(flight, release, h)
        v = torch.where(push, acceleration*push_time, 0.)
        return dict(height=h, velocity=v, push=push, flight=flight, active=active,
                    crouch=prepare, after_push=jumper & (push_time > duration),
                    release_velocity=release_v, requested=jumper & (t >= 0))

    def context(self, base, age):
        """No sensor or observer parameter is accepted at this boundary."""
        result = base.clone()
        jump = self.requesters() & (age >= self.delay)
        air = self.mask("airborne")
        selected = jump | air
        result[selected] = 0.
        result[selected, 4] = 1.
        result[jump, 5] = self.apex[jump]
        result[selected, 6] = torch.where(jump, age-self.delay, age)[selected].clamp(0., 6.)
        if self.has_platform:
            # Step-jump request: the operator-declared rise rides in the
            # reserved "stair" slot; flat requests keep it at zero.
            step = jump & self.mask("platform") & (self.mode == OVER)
            result[step, 1] = self.rise[step]
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
        self.sequence = None
        if self.j.get("reward_version", 1) >= 5:
            from .jump_objectives import JumpSequence
            self.sequence = JumpSequence(schedule, dt, mass)

    def reset(self, ids):
        if self.sequence is not None:
            self.sequence.reset(ids)
        for value in vars(self).values():
            if isinstance(value, torch.Tensor) and value is not self.weight:
                value[ids] = 0

    def begin_interval(self):
        if self.sequence is not None:
            self.sequence.begin_interval()
        for name in ("liftoff_event", "landing_event", "success_event", "timeout_event", "interval_force_peak"):
            getattr(self, name).zero_()

    def observe(self, age, height, velocity, gravity, wheel_clearance, wheel_force, nonwheel_force,
                horizontal_speed_error=None, leg_lengths=None, on_top=None, physics=None):
        j, s = self.j, self.schedule
        # Platform requests share the jump physics; tuck/apex gates stay jump-lane only.
        jump, drop, jump_lane = s.requesters(), s.mask("airborne"), s.mask("jump")
        eligible = drop | (jump & (age >= s.delay))
        air = (wheel_clearance > j["clearance_m"]).all(-1) & (wheel_force <= j["force_threshold_n"]).all(-1)
        support = (wheel_clearance < j["landing_clearance_m"]).all(-1) & (wheel_force > j["force_threshold_n"]).all(-1)
        v4 = j.get("reward_version",1) >= 4
        first_air = eligible & air & (self.air_time <= 0.) & ~self.flown
        # A failed provisional flight must not donate a convenient release
        # baseline or a previously compressed leg to the next real takeoff.
        lost_candidate = eligible & ~air & ~self.flown
        touching = (wheel_clearance < j["landing_clearance_m"]).any(-1) & (wheel_force > j["force_threshold_n"]).any(-1)
        phase_vz = velocity[:, 2]
        if self.sequence is not None:
            phase_vz = self.sequence.measure(self, age, air, touching, first_air, lost_candidate,
                wheel_clearance, nonwheel_force, gravity, physics)
        if v4:
            for name in ("release_leg_length", "max_bilateral_tuck_delta"):
                value = getattr(self,name)
                value[lost_candidate] = 0.
            self.release_leg_lengths[lost_candidate] = 0.
        self.air_time = torch.where(eligible & air, self.air_time+self.dt, 0.)
        liftoff = (jump & ~self.flown & (self.air_time >= j["minimum_air_seconds"]-1e-7)
                   & (phase_vz > j["minimum_release_vz_m_s"]))
        if self.sequence is not None:
            # A reset/preparation rebound is not an actuated jump. Require a
            # measured supported push before the first accepted flight.
            liftoff &= self.sequence.values["push_duration"] > 0.
        self.liftoff_event |= liftoff
        new_flight = liftoff | (drop & (self.air_time >= j["minimum_air_seconds"]-1e-7))
        self.flown |= new_flight
        if v4:
            in_flight = eligible & self.flown & ~self.landed
            new_apex = in_flight & ~self.apex_seen & (phase_vz <= j["flight_ascent_vz_threshold_m_s"])
            self.apex_hold_time = torch.where(in_flight & self.apex_seen,
                self.apex_hold_time+self.dt,self.apex_hold_time)
            self.apex_seen |= new_apex
            ready = (self.apex_hold_time >= j["apex_tuck_hold_seconds"]-1e-7)
            if self.sequence is not None:
                ready = self.sequence.extension_ready(self, wheel_clearance, phase_vz)
            self.extension_started |= in_flight & self.apex_seen & ready
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
            if self.sequence is not None:
                self.tuck_seen |= self.sequence.measure_legs(self, age, in_flight, air, leg_lengths, bilateral_delta)
            else:
                self.tuck_seen |= in_flight & (measured >= minimum_tuck)
        self.peak_height = torch.where(eligible, torch.maximum(self.peak_height, height), self.peak_height)
        landing = eligible & self.flown & ~self.landed & touching
        self.landing_event |= landing
        self.landed |= landing
        if self.sequence is not None:
            self.sequence.after_observe(self, new_flight, landing, leg_lengths, wheel_clearance, physics)
        self.force_peak = torch.maximum(self.force_peak, wheel_force.sum(-1)*eligible)
        self.interval_force_peak = torch.maximum(self.interval_force_peak, wheel_force.sum(-1)*eligible)
        speed_error = velocity[:, :2].norm(dim=-1) if horizontal_speed_error is None else horizontal_speed_error
        calm = ((gravity[:, 2] < -math.cos(math.radians(12))) & (speed_error < .15)
                & (velocity[:, 2].abs() < .15) & ((height-j["standing_height_m"]).abs() < .035)
                & (nonwheel_force <= 2.))
        stable = eligible & self.landed & support & calm
        self.stable_time = torch.where(stable, self.stable_time+self.dt, 0.)
        apex_ok = drop | (self.peak_height >= s.apex-.025)
        if s.has_platform:
            # A step jump counts only once both wheels are supported on top;
            # the edge itself enforces the clearance instead of an apex gate.
            over = s.mask("platform") & (s.mode == OVER)
            on_top = torch.zeros_like(over) if on_top is None else on_top
            apex_ok = torch.where(over, on_top, apex_ok)
        tuck_required = bool(j.get("success_requires_tuck", False))
        # Airborne is a passive drop diagnostic and must not be forced through
        # the active-jump tuck gate.  If a caller omits leg geometry, retain
        # the historical observer behavior for v1-v3. V4 fails closed if jump
        # geometry is omitted; the native environment always supplies it.
        if leg_lengths is not None:
            tuck_required_mask = jump_lane if tuck_required else torch.zeros_like(jump_lane)
            tuck_ok = (~tuck_required_mask) | self.tuck_seen
        elif v4:
            tuck_ok = ~jump_lane if tuck_required else torch.ones_like(jump_lane)
        else:
            tuck_ok = torch.ones_like(self.succeeded)
        if self.sequence is not None:
            tuck_ok &= ~jump | self.sequence.success_ready()
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
            if self.sequence is not None:
                # More retraction must not lose the v4 Gaussian target payment.
                tuck = progress
            if j.get("reward_version",1) >= 4:
                # Early asymmetric progress must receive a gradient, while
                # the event gate continues to require both legs together.
                per_leg_delta = (release-legs).clamp_min(0.)
                per_leg = (.5*(per_leg_delta/tuck_target).clamp(0.,1.)
                    +.5*torch.exp(-((per_leg_delta-tuck_target)/tuck_sigma).square())).mean(-1)
                if self.sequence is not None:
                    per_leg = (per_leg_delta/tuck_target).clamp(0., 1.).mean(-1)
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
            if self.sequence is not None:
                preparing = active & ref["crouch"]
            recovery = active & self.landed
            recovery_height_error = torch.where(buffer_active,
                                                height-buffer_height_ref,
                                                height_error)
            missed = active & ref["after_push"] & ~self.flown
            velocity_gain = (torch.exp(-((vz-ref["velocity"])/.5).square())
                             -torch.exp(-(ref["velocity"]/.5).square()))
            progress = (vz/ref["release_velocity"]).clamp(-1., 1.)
            terms = dict(
                jump_height=j["dense_scale"]*(4*preparing*tracking
                    +recovery*(2*torch.exp(-(recovery_height_error/.035).square())-1.)
                    +landing_buffer_payment
                    -j["missed_takeoff_cost_per_second"]*missed),
                jump_push_velocity=j["dense_scale"]*ref["push"]*(6*velocity_gain+4*progress),
                jump_air_placement=j["dense_scale"]*2*flight*placement,
                jump_impact=-j["dense_scale"]*((self.interval_force_peak/self.weight-j["impact_soft_bodyweights"])
                    .clamp_min(0.)/j["impact_scale_bodyweights"]).clamp_max(5.),
                jump_events=events/policy_dt)
            if self.sequence is not None:
                sequence = self.sequence
                terms["jump_push_velocity"] = sequence.values["interval_push_payment"]/policy_dt
                terms["jump_release_energy"] = sequence.values["interval_release_bonus"]/policy_dt
                terms["jump_clearance"] = sequence.values["interval_clearance_payment"]/policy_dt
                terms["jump_impact"] = -j["dense_scale"]*((sequence.values["interval_landing_force"]/
                    self.weight-j["impact_soft_bodyweights"]).clamp_min(0.)/j["impact_scale_bodyweights"]).clamp_max(5.)
            return terms
        return dict(jump_height=j["dense_scale"]*height_phase*(4*torch.exp(-(height_error/.035).square())
                       -torch.log1p((height_error/.06).square())),
                    jump_push_velocity=j["dense_scale"]*ref["push"]*(6*torch.exp(-((vz-ref["velocity"])/.5).square())
                       -2*(ref["velocity"]-vz).clamp_min(0.)),
                    jump_air_placement=j["dense_scale"]*2*flight*placement,
                    jump_impact=-j["dense_scale"]*((self.interval_force_peak/self.weight-j["impact_soft_bodyweights"])
                       .clamp_min(0.)/j["impact_scale_bodyweights"]).clamp_max(5.),
                    jump_events=events/policy_dt)


class PlatformObserver:
    """Step-up / drive-off payments from true box geometry and contact.

    Reward and diagnostics only: nothing here reaches the actor, the command
    clock or the request trigger. Wheel arrays are (N, 2); positions env-local.
    """
    EVENTS = ("strike_event", "top_land_event", "drop_land_event", "drop_success_event", "drop_timeout_event")

    def __init__(self, schedule, dt, mass):
        self.schedule, self.dt = schedule, dt
        self.p, self.j = schedule.spec["platform"], schedule.spec["jump"]
        self.weight = mass*9.81
        z = schedule.height
        for name in ("drop_stable_time", "drop_clock", "interval_force_peak", "min_clearance"):
            setattr(self, name, torch.zeros_like(z))
        for name in ("seen_top", "top_landed", "struck", "dropped", "drop_landed", "drop_succeeded",
                     "drop_failed", *self.EVENTS):
            setattr(self, name, torch.zeros_like(z, dtype=torch.bool))
        self.min_clearance.fill_(1.)

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor) and value is not self.weight:
                value[ids] = 0
        self.min_clearance[ids] = 1.

    def begin_interval(self):
        for name in (*self.EVENTS, "interval_force_peak"):
            getattr(self, name).zero_()

    def lanes(self):
        s = self.schedule
        return s.mask("platform") & (s.mode != FLAT)

    def observe(self, age, wheel_xy, wheel_bottom, support, wheel_force, wheel_force_x,
                height, velocity, gravity, speed_error, nonwheel_force):
        """wheel_bottom is height above the floor; support the box height under each wheel."""
        s, p, j = self.schedule, self.p, self.j
        lanes = self.lanes()
        over = lanes & (s.mode == OVER)
        rise = s.rise[:, None]
        clear = wheel_bottom-support
        threshold = j["force_threshold_n"]
        supported = (clear < j["landing_clearance_m"]) & (wheel_force > threshold)
        top = lanes & (support > 0.).all(-1) & supported.all(-1)
        floor = (support <= 0.).all(-1) & supported.all(-1)
        near, radius = p["near_edge_x_m"], j["wheel_radius_m"]
        # Wheel actually against the vertical face: within 5 mm of touching it,
        # below the top edge, and pushed back. Braking friction on the floor
        # further away is not a strike.
        face = ((wheel_xy[..., 0] > near-radius-.005) & (wheel_xy[..., 0] < near)
                & (wheel_bottom < rise-.01) & (wheel_force_x < -p["strike_force_n"])).any(-1)
        strike = over & ~self.struck & ~self.top_landed & face
        self.strike_event |= strike
        self.struck |= strike
        landed_top = over & ~self.top_landed & top
        self.top_land_event |= landed_top
        self.top_landed |= landed_top
        self.seen_top |= top
        near_face = (wheel_xy[..., 0] > near-.25) & (wheel_xy[..., 0] < near+.05)
        gap = torch.where(near_face, wheel_bottom-rise, 1.).amin(-1)
        self.min_clearance = torch.where(over & ~self.top_landed, torch.minimum(self.min_clearance, gap), self.min_clearance)
        # Leaving the box (both wheel centres off it) counts as the drop even if
        # a wheel slides down the far corner instead of flying clear.
        drop = lanes & self.seen_top & ~self.dropped & (support <= 0.).all(-1)
        self.dropped |= drop
        land = self.dropped & ~self.drop_landed & floor
        self.drop_land_event |= land
        self.drop_landed |= land
        self.drop_clock = torch.where(self.dropped, self.drop_clock+self.dt, self.drop_clock)
        calm = ((gravity[:, 2] < -math.cos(math.radians(12))) & (speed_error < .2)
                & (velocity[:, 2].abs() < .15) & ((height-j["standing_height_m"]).abs() < .035)
                & (nonwheel_force <= 2.))
        stable = self.drop_landed & floor & calm
        self.drop_stable_time = torch.where(stable, self.drop_stable_time+self.dt, 0.)
        success = (lanes & self.drop_landed & ~self.drop_succeeded & ~self.drop_failed
                   & (self.drop_stable_time >= j["landing_hold_seconds"]-1e-7))
        self.drop_success_event |= success
        self.drop_succeeded |= success
        timeout = (lanes & self.seen_top & ~self.drop_succeeded & ~self.drop_failed
                   & (age >= p["outcome_seconds"]))
        self.drop_timeout_event |= timeout
        self.drop_failed |= timeout
        window = self.dropped & (self.drop_clock < .5)
        self.interval_force_peak = torch.maximum(self.interval_force_peak, wheel_force.sum(-1)*window)

    def reward_terms(self, age, wheel_xy, wheel_bottom, policy_dt):
        s, p, j = self.schedule, self.p, self.j
        over = self.lanes() & (s.mode == OVER)
        ref = s.reference(age)
        near = p["near_edge_x_m"]
        before, after = p["clearance_zone_m"]
        zone = (wheel_xy[..., 0] > near-before) & (wheel_xy[..., 0] < near+after)
        score = ((wheel_bottom-s.rise[:, None])/p["clearance_scale_m"]).clamp(-1., 1.)
        weight = zone.float()
        clearance = (score*weight).sum(-1)/weight.sum(-1).clamp_min(1.)
        # Flight only: wheels on the floor during the push are expected.
        active = over & ref["after_push"] & ~self.top_landed & ~self.struck & zone.any(-1)
        b = p["event_bonuses"]
        events = (b["strike"]*self.strike_event+b["top_land"]*self.top_land_event
                  +b["drop_land"]*self.drop_land_event+b["drop_success"]*self.drop_success_event
                  +b["drop_timeout"]*self.drop_timeout_event)
        return dict(
            platform_clearance=j["dense_scale"]*p["clearance_weight"]*active*clearance,
            platform_events=events/policy_dt,
            platform_impact=-j["dense_scale"]*((self.interval_force_peak/self.weight-j["impact_soft_bodyweights"])
                .clamp_min(0.)/j["impact_scale_bodyweights"]).clamp_max(5.))


def rollout_metrics(roles, advantages, rewards, returns, tasks=TASKS):
    """Inspect actual PPO tensors without changing normalization or gradients."""
    result = {}
    for name in tasks:
        mask = roles == tasks.index(name)
        for key, value in (("advantage", advantages), ("reward", rewards), ("return", returns)):
            selected = value[:, mask].detach().float()
            result["MixedRollout/"+name+"/"+key+"_rms"] = selected.square().mean().sqrt()
        result["MixedRollout/"+name+"/samples"] = int(mask.sum())*advantages.shape[0]
    return result

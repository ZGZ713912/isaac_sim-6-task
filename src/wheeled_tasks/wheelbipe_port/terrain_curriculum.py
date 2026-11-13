"""V9 time-only terrain commands. Physical observations never enter this module."""
from __future__ import annotations

import math

import torch

TERRAIN_TASKS = ("slope", "step_down", "bump", "balls")
STAND, SPIN, DRIVE, JUMP = 1, 2, 3, 4


def ramp(update, span, start, final):
    progress = min(max((update-span[0])/max(span[1]-span[0], 1), 0.), 1.)
    return start+(final-start)*progress


def unlocked_limit(update, cfg, unit):
    u = cfg["unlock"]
    if update < u["start_update"]:
        return 0.
    return ramp(update, (u["start_update"], u["full_update"]), u["start_"+unit], u["final_"+unit])


def validate_terrain(spec):
    """Fail closed on nonfinite geometry, time, physical and reward parameters."""
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return not isinstance(value, (int, float)) or math.isfinite(value)
    for key in (*TERRAIN_TASKS, "level", "position_hold", "terrain_reward_weights"):
        if key not in spec or not finite(spec[key]):
            raise ValueError("Invalid or missing V9 terrain block: "+key)
    for name, unit, ceiling in (("slope", "angle_deg", 12.), ("step_down", "height_m", .25),
                                ("bump", "height_m", .07)):
        cfg = spec[name]
        values = cfg["lane_angles_deg" if name == "slope" else "lane_heights_m"]
        u = cfg["unlock"]
        if (not values or min(values) < 0. or max(values) > ceiling+1e-8
                or not 0 <= u["start_update"] <= u["full_update"]
                or not 0 <= u["start_"+unit] <= u["final_"+unit] <= ceiling
                or cfg["episode_seconds"] <= 0):
            raise ValueError("Invalid V9 "+name+" curriculum")
    for name in ("slope", "balls"):
        f = spec[name]["mode_fractions"]
        if any(v < 0. for v in f.values()) or abs(sum(f.values())-1.) > 1e-9:
            raise ValueError("Invalid V9 mode fractions")
    b = spec["balls"]
    if (not 0 < b["diameter_m"] <= .02 or not 0 < b["mass_kg"] <= .01
            or not 0 < b["coverage_start"] <= b["coverage_final"] <= .30
            or b["patch_side_m"] <= 0 or b["layouts"] < 1
            or not 0 <= b["coverage_review_probability"] <= 1):
        raise ValueError("Invalid V9 ball field")
    if not .001 <= b.get("reset_clearance_m",.001) <= .02:
        raise ValueError("V9 ball reset collision clearance must be 1..20 mm")
    from .training_profile import solver_settings
    solver_settings(spec)
    jump = spec["slope"]["jump"]
    if (not 0 <= jump["probability_final"] <= .30 or jump["angle_limit_final_deg"] > 12
            or not 0 < spec["position_hold"]["hold_sigma_m"]
            or not 0 < spec["level"]["roll_scale_deg"]
            or not 0 < spec["level"]["pitch_scale_deg"]):
        raise ValueError("Invalid V9 level/hold/jump settings")


class TerrainSchedule:
    """Fixed geometry and reset-sampled modes; a seeded child of TaskSchedule."""
    def __init__(self, schedule):
        self.s, self.spec = schedule, schedule.spec
        z = schedule.height
        self.rank = {}
        self.ids = {task:schedule.mask(task).nonzero().flatten() for task in TERRAIN_TASKS}
        for task in TERRAIN_TASKS:
            mask = schedule.mask(task)
            self.rank[task] = torch.where(mask, mask.long().cumsum(0)-1, -1)
        for name in ("yaw", "start_x", "start_y", "speed", "yaw_rate", "spin_rpm", "coverage",
                     "episode_count", "terrain_height", "slope_angle", "slope_yaw", "bump_yaw"):
            setattr(self, name, torch.zeros_like(z))
        self.mode = torch.zeros_like(z, dtype=torch.long)
        self.active = torch.zeros_like(z, dtype=torch.bool)
        self.sign = torch.ones_like(z)
        self.terrain_lane = torch.zeros_like(self.active)
        self.segment = torch.full_like(self.mode, -1)
        self.force_full_ball_coverage = False
        for task in TERRAIN_TASKS:
            self.terrain_lane |= schedule.mask(task)
        self.slope_angle = self.fixed("slope", "lane_angles_deg")*math.pi/180.
        self.slope_yaw = 2*math.pi*torch.frac(self.rank["slope"].clamp_min(0)*.6180339887)
        for task in ("step_down", "bump"):
            self.terrain_height += self.fixed(task, "lane_heights_m")
        self.bump_yaw = self.fixed("bump", "lane_yaws_deg")*math.pi/180.
        widths = torch.tensor(self.spec["bump"]["strip_widths_m"], device=z.device)
        rank = self.rank["bump"].clamp_min(0)
        self.strip_widths = widths[(rank[:, None]+torch.arange(4, device=z.device)) % len(widths)]
        self.strip_starts = torch.full_like(self.strip_widths, self.spec["bump"]["first_strip_x_m"])
        self.strip_starts[:, 1:] += (self.strip_widths[:, :-1]+self.spec["bump"]["gap_m"]).cumsum(-1)

    def fixed(self, task, key):
        values = torch.tensor(self.spec[task][key], device=self.s.device, dtype=self.s.height.dtype)
        return torch.where(self.s.mask(task), values[self.rank[task].clamp_min(0) % len(values)], 0.)

    def limits(self, update=None):
        u = self.s.update if update is None else update
        p, b = self.spec["slope"], self.spec["balls"]
        j = p["jump"]
        return dict(slope=unlocked_limit(u, p, "angle_deg"),
            step_down=unlocked_limit(u, self.spec["step_down"], "height_m"),
            bump=unlocked_limit(u, self.spec["bump"], "height_m"),
            spin_rpm=ramp(u, p["spin"]["rpm_ramp"], p["spin"]["rpm_start"], p["spin"]["rpm_final"]),
            jump_probability=ramp(u, j["probability_ramp"], 0., j["probability_final"]),
            jump_angle=ramp(u, j["angle_limit_ramp"], j["angle_limit_start_deg"], j["angle_limit_final_deg"]),
            coverage=0. if u < b["coverage_ramp"][0] else ramp(u, b["coverage_ramp"], b["coverage_start"], b["coverage_final"]))

    def uniform(self, ids, bounds):
        return bounds[0]+(bounds[1]-bounds[0])*self.s.random(len(ids))

    def reset(self, ids, update):
        limits = self.limits(update)
        s = self.s
        for task in TERRAIN_TASKS:
            lane = ids[s.mask(task)[ids]]
            if not len(lane):
                continue
            c, n = self.spec[task], len(lane)
            self.sign[lane] = torch.where((self.rank[task][lane]+self.episode_count[lane].long()) % 2 == 0, 1., -1.)
            self.episode_count[lane] += 1
            self.yaw[lane] = self.uniform(lane, [-math.pi, math.pi]) if task in ("slope", "balls") else 0.
            self.start_x[lane] = self.start_y[lane] = self.yaw_rate[lane] = 0.
            self.coverage[lane] = 0.
            self.segment[lane] = -1
            s.height[lane] = self.uniform(lane, c["height_range_m"])
            s.next_sample[lane] = 1e3
            s.delay[lane] = 1e3
            if task == "slope":
                self.active[lane] = self.slope_angle[lane] <= math.radians(limits[task])+1e-7
                pick = s.random(n)
                f = c["mode_fractions"]
                mode = torch.where(pick < f["spin"], SPIN, torch.where(pick < f["spin"]+f["stand"], STAND, DRIVE))
                j = c["jump"]
                jump = ((s.random(n) < limits["jump_probability"])
                    & (self.slope_angle[lane] <= math.radians(limits["jump_angle"])+1e-7) & self.active[lane])
                # Keep at least 70% non-jump exposure even for small reset
                # batches; independent Bernoulli draws alone exceed the quota.
                outside = s.mask("slope").clone()
                outside[lane] = False
                available = max(0,math.floor(int(s.mask("slope").sum())*j["probability_final"])
                    -int((outside & (self.mode == JUMP)).sum()))
                chosen = jump.nonzero().flatten()
                if len(chosen) > available:
                    jump[chosen[torch.argsort(s.random(len(chosen)))[available:]]] = False
                self.mode[lane] = torch.where(jump, JUMP, mode)
                review = self.uniform(lane, c["spin"]["review_fraction_of_max"])*limits["spin_rpm"]
                late = (update >= c["spin"]["late_update"]) & (s.random(n) < c["spin"]["late_band_probability"])
                self.spin_rpm[lane] = torch.where(late, self.uniform(lane, c["spin"]["late_band_rpm"]), review)
                self.speed[lane] = self.uniform(lane, c["drive"]["speed_range_m_s"])*self.sign[lane]
                self.yaw_rate[lane] = (self.uniform(lane, c["drive"]["yaw_range_rad_s"])*self.sign[lane]
                    *(s.random(n) < c["drive"]["arc_probability"]))
                self.start_x[lane] = self.uniform(lane, [-c["spawn_offset_m"], c["spawn_offset_m"]])
                self.start_y[lane] = self.uniform(lane, [-c["spawn_offset_m"], c["spawn_offset_m"]])
                jump_ids = lane[jump]
                s.delay[jump_ids] = self.uniform(jump_ids, j["request_delay_s"])
                self.speed[jump_ids] = self.uniform(jump_ids, j["moving_speed_m_s"])*self.sign[jump_ids]*(s.random(len(jump_ids)) < j["moving_fraction"])
            elif task == "step_down":
                self.active[lane] = self.terrain_height[lane] <= limits[task]+1e-7
                self.mode[lane] = DRIVE
                vmax = ramp(update, c["speed_ramp"], c["speed_max_start_m_s"], c["speed_max_final_m_s"])
                self.speed[lane] = self.uniform(lane, [c["speed_min_m_s"], vmax])
                self.yaw[lane] = self.uniform(lane, [-c["heading_offset_deg"], c["heading_offset_deg"]])*math.pi/180
                distance = self.uniform(lane, c["start_distance_m"])
                self.start_x[lane] = c["box_length_m"]-(distance-.011)*self.yaw[lane].cos()
                self.start_y[lane] = self.uniform(lane, [-c["lateral_offset_m"], c["lateral_offset_m"]])
            elif task == "bump":
                self.active[lane] = self.terrain_height[lane] <= limits[task]+1e-7
                self.mode[lane] = DRIVE
                reverse = s.random(n) < c["reverse_fraction"]
                self.yaw[lane] = torch.where(reverse, math.pi, 0.)
                self.speed[lane] = self.uniform(lane, c["speed_range_m_s"])*torch.where(reverse, -1., 1.)
                self.yaw_rate[lane] = c["arc_yaw_rad_s"]*self.sign[lane]*(s.random(n) < c["arc_fraction"])
            else:
                self.active[lane] = limits["coverage"] > 0.
                review = self.uniform(lane, [c["coverage_review_min"], max(c["coverage_review_min"], limits["coverage"])])
                self.coverage[lane] = torch.where(s.random(n) < c["coverage_review_probability"], review, limits["coverage"])*self.active[lane]
                if self.force_full_ball_coverage:
                    self.coverage[lane] = limits["coverage"]
                f, pick = c["mode_fractions"], s.random(n)
                self.mode[lane] = torch.where(pick < f["stand"], STAND, torch.where(pick < f["stand"]+f["spin"], SPIN, DRIVE))
                self.spin_rpm[lane] = self.uniform(lane, c["spin_rpm"])
                self.speed[lane] = self.uniform(lane, c["shuttle"]["speed_m_s"])
                self.yaw_rate[lane] = c["shuttle"]["arc_yaw_rad_s"]*self.sign[lane]*(s.random(n) < .3)
            self.start_x[lane] = torch.where(self.active[lane], self.start_x[lane], -c["flat_offset_m"])

    def update_commands(self, age):
        s = self.s
        for task in TERRAIN_TASKS:
            ids = self.ids[task]
            c, t = self.spec[task], age[ids]
            mode = self.mode[ids]
            cmd = torch.zeros(len(ids), 3, device=s.device)
            if task == "slope":
                spin = (mode == SPIN) & (t >= c["spin"]["pre_hold_s"]) & (t < c["spin"]["pre_hold_s"]+c["spin"]["spin_s"])
                drive = (mode == DRIVE) & (t >= 1.) & (t < 1.+c["drive"]["drive_s"])
                jumping = (mode == JUMP) & (t >= s.delay[ids])
                cmd[:, 0] = self.speed[ids]*(drive | jumping)
                cmd[:, 2] = torch.where(spin, self.sign[ids]*self.spin_rpm[ids]*math.pi/30., self.yaw_rate[ids]*drive)
                segment = torch.floor(t/c["stand"]["segment_s"]).long()
            elif task == "balls":
                spin = (mode == SPIN) & (t >= 1.) & (t < 5.)
                direction = torch.where(torch.floor(t/c["shuttle"]["period_s"]).long() % 2 == 0, 1., -1.)
                cmd[:, 0] = self.speed[ids]*direction*(mode == DRIVE)
                cmd[:, 2] = torch.where(spin, self.sign[ids]*self.spin_rpm[ids]*math.pi/30., self.yaw_rate[ids]*(mode == DRIVE)*direction)
                segment = torch.floor(t/2.).long()
            else:
                cmd[:, 0] = self.speed[ids]
                cmd[:, 2] = self.yaw_rate[ids]
                segment = torch.zeros_like(mode)
            changed = (mode == STAND) & (segment != self.segment[ids])
            sample = ids[changed]
            if len(sample):
                s.height[sample] = self.uniform(sample, c["height_range_m"])
            self.segment[ids] = segment
            s.commands[ids] = cmd

    def support_height(self, xy):
        from .terrain_objectives import slope_height
        shape = (-1,)+ (1,)*(xy.dim()-2)
        slope = self.s.mask("slope") & self.active
        height = slope_height(xy, self.slope_angle, self.slope_yaw, self.spec["slope"])
        height = torch.where(slope.reshape(shape), height, 0.)
        c = self.spec["step_down"]
        inside = (xy[..., 0] >= 0) & (xy[..., 0] <= c["box_length_m"]) & (xy[..., 1].abs() <= c["box_width_m"]/2)
        lane = (self.s.mask("step_down") & self.active).reshape(shape)
        return torch.where(lane & inside, self.terrain_height.reshape(shape), height)

    def bump_windows(self, xy, margin=0.):
        normal = torch.stack((self.bump_yaw.cos(), self.bump_yaw.sin()), -1)
        along = (xy*normal[:, None]).sum(-1)
        cross = (xy*torch.stack((-normal[:, 1], normal[:, 0]), -1)[:, None]).sum(-1)
        inside = ((along[..., None] >= self.strip_starts[:, None]-margin)
            & (along[..., None] <= (self.strip_starts+self.strip_widths)[:, None]+margin)
            & (cross[..., None].abs() <= self.spec["bump"]["strip_length_m"]/2))
        return inside & (self.s.mask("bump") & self.active)[:, None, None]

    def bump_strip_height(self, xy):
        return self.bump_windows(xy).any(-1)*self.terrain_height[:, None]

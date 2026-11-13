"""Reward-only terrain geometry and physical observers, independent of Isaac Sim.

All rewards returned here are densities. Event impulses are divided by policy_dt;
the inherited reward entry applies weights and policy_dt exactly once.
"""
from __future__ import annotations

import math

import torch


def roll_pitch_deg(gravity):
    return (torch.rad2deg(torch.atan2(gravity[:, 1], -gravity[:, 2])),
            torch.rad2deg(torch.atan2(-gravity[:, 0], -gravity[:, 2])))


def deadband_square(angle, deadband, scale, clamp):
    return ((angle.abs()-deadband).clamp_min(0.)/scale).square().clamp_max(clamp)


def slope_geometry(angle, yaw, cfg):
    """Top normal, scalar-first quaternion and centre, in the terrain frame."""
    st, ct, sy, cy = angle.sin(), angle.cos(), yaw.sin(), yaw.cos()
    normal = torch.stack((st*cy, st*sy, ct), -1)
    quat = torch.stack(((yaw/2).cos()*(angle/2).cos(), -(yaw/2).sin()*(angle/2).sin(),
                       (yaw/2).cos()*(angle/2).sin(), (yaw/2).sin()*(angle/2).cos()), -1)
    z0 = cfg["box_length_m"]/2*st+cfg["edge_clearance_m"]
    center = -normal*cfg["box_thickness_m"]/2
    center[..., 2] += z0
    return normal, quat, center


def slope_height(xy, angle, yaw, cfg):
    shape = (-1,)+(1,)*(xy.dim()-2)
    a, y = angle.reshape(shape), yaw.reshape(shape)
    u = xy[..., 0]*y.cos()+xy[..., 1]*y.sin()
    v = -xy[..., 0]*y.sin()+xy[..., 1]*y.cos()
    z0 = cfg["box_length_m"]/2*a.sin()+cfg["edge_clearance_m"]
    inside = (u.abs() <= cfg["box_length_m"]/2*a.cos()) & (v.abs() <= cfg["box_width_m"]/2)
    return torch.where(inside, z0-a.tan()*u, 0.)


def slope_leg_lengths(height, angle, slope_yaw, robot_yaw):
    side_y = torch.tensor([.218, -.219], device=height.device, dtype=height.dtype)
    side_h = -side_y*angle.tan()[:, None]*(slope_yaw-robot_yaw).sin()[:, None]
    return (height[:, None]-.06-side_h).clamp(.17, .37)


class PositionHold:
    def __init__(self, schedule):
        self.s, self.spec = schedule, schedule.spec["position_hold"]
        z = schedule.height
        self.anchor = torch.zeros(len(z), 2, device=z.device)
        self.jump_anchor = torch.zeros_like(self.anchor)
        for name in ("clock", "radius", "radius_max"):
            setattr(self, name, torch.zeros_like(z))
        for name in ("valid", "prev_zero", "jump_captured"):
            setattr(self, name, torch.zeros_like(z, dtype=torch.bool))
        self.scale = torch.zeros_like(z)
        for name in self.spec["lanes"]:
            self.scale[schedule.mask(name)] = self.spec["lane_scale"][name]

    def reset(self, ids):
        for name in ("anchor", "jump_anchor", "clock", "radius", "radius_max", "valid", "prev_zero", "jump_captured"):
            getattr(self, name)[ids] = 0

    def update(self, xy, command, dt, *, blocked=None, request=None, jump_return=None):
        empty = torch.zeros_like(self.valid)
        blocked = empty if blocked is None else blocked
        request = empty if request is None else request
        jump_return = empty if jump_return is None else jump_return
        capture = request & ~self.jump_captured
        self.jump_anchor[capture] = xy[capture]
        self.jump_captured |= request
        zero = (command[:, :2].abs().amax(-1) < 1e-6) & (self.scale > 0.) & ~blocked
        new = zero & ~self.prev_zero
        self.valid &= zero & ~new
        self.clock = torch.where(zero, torch.where(new, 0., self.clock)+dt, 0.)
        ready = zero & ~self.valid & (self.clock >= self.spec["settle_s"]-1e-7)
        # A settled in-place jump is measured against its request location,
        # rather than silently accepting the new landing location as zero drift.
        ready |= zero & jump_return & self.jump_captured & ~self.valid
        self.anchor[ready] = torch.where((jump_return & self.jump_captured)[ready, None],
                                       self.jump_anchor[ready], xy[ready])
        self.valid |= ready
        self.prev_zero = zero
        self.radius = torch.where(self.valid, (xy-self.anchor).norm(dim=-1), 0.)
        self.radius_max = torch.maximum(self.radius_max, self.radius)

    def reward_terms(self):
        c, d = self.spec, self.radius
        return dict(position_hold=self.scale*self.valid*torch.exp(-(d/c["hold_sigma_m"]).square()),
            position_drift=self.scale*self.valid*deadband_square(d, c["drift_deadband_m"], c["drift_scale_m"], c["drift_clamp"]))


class StepDownObserver:
    EVENTS = ("success_event", "stumble_event", "timeout_event")

    def __init__(self, schedule, dt, mass):
        self.s, self.dt, self.cfg, self.weight = schedule, dt, schedule.spec["step_down"], mass*9.81
        z = schedule.height
        for name in ("land_time", "max_tilt_w", "stable_time", "force_peak_interval", "force_peak"):
            setattr(self, name, torch.zeros_like(z))
        self.min_ratio_w = torch.ones_like(z)
        for name in ("seen_top", "edge_left", "landed", "stumble", "success", "timeout", "window", "interval_window", *self.EVENTS):
            setattr(self, name, torch.zeros_like(z, dtype=torch.bool))

    def reset(self, ids):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor) and name != "weight":
                value[ids] = 1. if name == "min_ratio_w" else 0.

    def begin_interval(self):
        for name in (*self.EVENTS, "force_peak_interval", "interval_window"):
            getattr(self, name).zero_()

    def observe(self, age, xy, bottom, support, force_z, height, vz, roll, pitch, speed, cmd, nonwheel):
        c = self.cfg
        lane = self.s.mask("step_down") & self.s.terrain.active
        touching = (force_z > self.s.spec["jump"]["force_threshold_n"]) & ((bottom-support).abs() < .025)
        self.seen_top |= lane & (support > 0.).all(-1) & touching.all(-1)
        left = self.seen_top & ((xy[..., 0] > c["box_length_m"]).any(-1))
        self.edge_left |= left & lane
        landed = self.edge_left & ~self.landed & (support == 0.).all(-1) & touching.all(-1)
        self.land_time = torch.where(landed, age, self.land_time)
        self.landed |= landed
        self.window = lane & self.edge_left & (~self.landed | (age <= self.land_time+c["window_after_landing_s"]))
        self.interval_window |= self.window
        tilt = torch.maximum(roll.abs(), pitch.abs())
        ratio = speed/cmd.clamp_min(1e-6)
        self.max_tilt_w = torch.maximum(self.max_tilt_w, tilt*self.window)
        self.min_ratio_w = torch.where(self.window, torch.minimum(self.min_ratio_w, ratio), self.min_ratio_w)
        peak = force_z.clamp_min(0.).sum(-1)*self.window
        self.force_peak_interval = torch.maximum(self.force_peak_interval, peak)
        self.force_peak = torch.maximum(self.force_peak, peak)
        stumble = self.window & ((nonwheel > c["stumble"]["nonwheel_force_n"]) | (tilt > c["stumble"]["tilt_deg"]))
        self.stumble_event |= stumble & ~self.stumble
        self.stumble |= stumble
        calm = ((tilt < c["calm"]["tilt_deg"]) & (vz.abs() < c["calm"]["vz_m_s"])
            & ((height-self.s.height).abs() < c["calm"]["height_error_m"])
            & ((speed-cmd).abs() < c["calm"]["speed_error_m_s"])
            & (nonwheel <= c["calm"]["nonwheel_force_n"]) & touching.all(-1))
        self.stable_time = torch.where(lane & self.landed & calm, self.stable_time+self.dt, 0.)
        success = (lane & self.landed & (self.stable_time >= c["hold_seconds"]-1e-7)
            & (self.max_tilt_w <= c["success"]["max_window_tilt_deg"])
            & (self.min_ratio_w >= c["success"]["min_speed_ratio"]) & ~self.stumble & ~self.timeout)
        self.success_event |= success & ~self.success
        self.success |= success
        timeout = lane & (age >= c["outcome_seconds"]) & ~self.success
        self.timeout_event |= timeout & ~self.timeout
        self.timeout |= timeout

    def reward_terms(self, roll, pitch, speed, cmd, com_vz, policy_dt):
        win, c = self.interval_window, self.cfg
        approach = self.s.mask("step_down") & self.s.terrain.active & ~self.edge_left
        b = c["event_bonuses"]
        return dict(step_down_level=win*(deadband_square(roll, 3., 3., 25.)+deadband_square(pitch, 3., 3., 25.)),
            step_down_speed=win*torch.exp(-((speed-cmd)/.3).square()),
            step_down_impact=-self.s.spec["jump"]["dense_scale"]*((self.force_peak_interval/self.weight-2.5)/2).clamp(0., 5.),
            step_down_events=(b["success"]*self.success_event+b["stumble"]*self.stumble_event+b["timeout"]*self.timeout_event)/policy_dt,
            step_down_hop=approach*(com_vz-c["hop_vz_threshold_m_s"]).clamp_min(0.).square())


class BumpObserver:
    def __init__(self, schedule, dt):
        self.s, self.dt, self.cfg = schedule, dt, schedule.spec["bump"]
        shape = (len(schedule.role), 4)
        for name in ("entered_at", "max_tilt", "max_hdev"):
            setattr(self, name, torch.zeros(shape, device=schedule.device))
        self.min_ratio = torch.ones(shape, device=schedule.device)
        for name in ("in_window", "nonwheel_seen", "settled"):
            setattr(self, name, torch.zeros(shape, device=schedule.device, dtype=torch.bool))
        self.success_count = torch.zeros_like(schedule.height)
        self.failure_count = torch.zeros_like(schedule.height)
        self.interval_window = torch.zeros_like(schedule.role, dtype=torch.bool)

    def reset(self, ids):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                value[ids] = 1. if name == "min_ratio" else 0.

    def begin_interval(self):
        self.success_count.zero_(); self.failure_count.zero_(); self.interval_window.zero_()

    def observe(self, age, xy, roll, pitch, speed, cmd, height_error, nonwheel):
        c = self.cfg
        window = self.s.terrain.bump_windows(xy, c["window_margin_m"]).any(1)
        active = window & ~self.settled
        entered = active & ~self.in_window
        self.entered_at = torch.where(entered, age[:, None], self.entered_at)
        tilt = torch.maximum(roll.abs(), pitch.abs())[:, None]
        ratio = (speed/cmd.sign().mul(cmd.abs().clamp_min(1e-6)).where(cmd != 0., torch.ones_like(cmd)))[:, None]
        self.max_tilt = torch.where(active, torch.maximum(self.max_tilt, tilt), self.max_tilt)
        self.min_ratio = torch.where(active, torch.minimum(self.min_ratio, ratio), self.min_ratio)
        self.max_hdev = torch.where(active, torch.maximum(self.max_hdev, height_error.abs()[:, None]), self.max_hdev)
        self.nonwheel_seen |= active & (nonwheel[:, None] > 2.)
        elapsed = age[:, None]-self.entered_at
        deadline = (self.s.terrain.strip_widths+2*c["window_margin_m"])/cmd.abs().clamp_min(.01)[:, None]+c["stall_extra_s"]
        stalled = active & (elapsed > deadline)
        exited = self.in_window & ~window & ~self.settled
        settle = exited | stalled
        passed = (settle & ~stalled & (self.max_tilt <= c["success"]["max_tilt_deg"])
            & (self.min_ratio >= c["success"]["min_speed_ratio"])
            & (self.max_hdev <= c["success"]["max_height_dev_m"]) & ~self.nonwheel_seen)
        self.success_count += passed.sum(-1)
        self.failure_count += (settle & ~passed).sum(-1)
        self.settled |= settle
        self.interval_window |= window.any(-1) | exited.any(-1)
        self.in_window = window

    def reward_terms(self, roll, pitch, speed, cmd, vz, policy_dt):
        win, b = self.interval_window, self.cfg["event_bonuses"]
        return dict(bump_level=win*(deadband_square(roll, 1.5, 2., 25.)+deadband_square(pitch, 1.5, 2., 25.)),
            bump_speed=win*torch.exp(-((speed-cmd)/.2).square()), bump_body_vz=win*vz.square(),
            bump_events=(b["success"]*self.success_count+b["failure"]*self.failure_count)/policy_dt)


class BallsObserver:
    def __init__(self, schedule):
        self.s, self.cfg = schedule, schedule.spec["balls"]
        z = schedule.height
        for name in ("max_tilt", "max_radius", "slip_sum", "samples"):
            setattr(self, name, torch.zeros_like(z))
        self.anchor = torch.zeros(len(z), 2, device=z.device)
        self.anchor_valid = torch.zeros_like(z, dtype=torch.bool)
        self.success_event = torch.zeros_like(self.anchor_valid)

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0

    def begin_interval(self):
        self.success_event.zero_()

    def observe(self, xy, roll, pitch, slip):
        lane = self.s.mask("balls") & self.s.terrain.active
        first = lane & ~self.anchor_valid
        self.anchor[first] = xy[first]
        self.anchor_valid |= lane
        self.max_tilt = torch.maximum(self.max_tilt, torch.maximum(roll.abs(), pitch.abs())*lane)
        self.max_radius = torch.maximum(self.max_radius, (xy-self.anchor).norm(dim=-1)*lane)
        self.slip_sum += slip*lane
        self.samples += lane

    def finish(self, timeout, terminated):
        from .terrain_curriculum import STAND, SPIN
        lane, c = self.s.mask("balls") & self.s.terrain.active, self.cfg["success"]
        radius = torch.where(self.s.terrain.mode == STAND, c["stand_radius_m"],
                            torch.where(self.s.terrain.mode == SPIN, c["spin_radius_m"], float("inf")))
        self.success_event = lane & timeout & ~terminated & (self.max_tilt <= c["max_tilt_deg"]) & (self.max_radius <= radius)

    def reward_terms(self, roll, pitch, policy_dt):
        lane = self.s.mask("balls") & self.s.terrain.active
        return dict(balls_level=lane*(deadband_square(roll, 1.5, 2., 25.)+deadband_square(pitch, 1.5, 2., 25.)),
            balls_events=self.success_event*self.cfg["success"]["bonus"]/policy_dt)


class TerrainObjectives:
    """Physical measurements, never passed back to the command scheduler."""
    def __init__(self, schedule, dt, mass):
        self.s, self.dt = schedule, dt
        self.hold = PositionHold(schedule)
        self.step_down = StepDownObserver(schedule, dt, mass)
        self.bump = BumpObserver(schedule, dt)
        self.balls = BallsObserver(schedule)
        z = schedule.height
        for name in ("air_land_time", "slope_land_time", "slope_jump_tilt", "slope_jump_offset",
                     "slope_spin_radius", "slope_spin_tilt", "episode_max_tilt", "speed", "com_vz",
                     "roll", "pitch", "body_vz", "slope_event"):
            setattr(self,name,torch.zeros_like(z))
        self.spin_anchor = torch.zeros(len(z),2,device=z.device)
        for name in ("air_landed", "slope_landed", "spin_started", "spin_finished", "spin_success",
                     "slope_jump_window", "slope_interval_window", "slope_requested"):
            setattr(self,name,torch.zeros_like(z,dtype=torch.bool))
        self.totals = {task: {key: torch.zeros((),device=z.device,dtype=torch.float64)
            for key in ("completed", "success", "terminated", "tilt_sum", "radius_sum")}
            for task in ("slope","slope_jump","step_down","bump","balls")}

    def reset(self, ids):
        for observer in (self.hold,self.step_down,self.bump,self.balls):
            observer.reset(ids)
        for value in vars(self).values():
            if isinstance(value,torch.Tensor):
                value[ids] = 0

    def begin_interval(self, age, xy, command, jump_observer, policy_dt):
        from .terrain_curriculum import JUMP
        for observer in (self.step_down,self.bump,self.balls):
            observer.begin_interval()
        self.slope_event.zero_()
        self.slope_interval_window.zero_()
        slope_jump = self.s.mask("slope") & (self.s.terrain.mode == JUMP)
        request = slope_jump & (age >= self.s.delay)
        blocked = request & (~self.slope_landed | (age < self.slope_land_time+self.s.spec["slope"]["jump"]["hold_after_landing_s"]))
        jump_return = request & ~blocked & (self.s.terrain.speed.abs() < 1e-6)
        self.hold.update(xy,command,policy_dt,blocked=blocked,request=request,jump_return=jump_return)
        self.slope_requested = request

    def observe(self, age, root_xy, wheel_xy, bottom, support, force_z, height, velocity_z,
                gravity, speed, cmd, nonwheel, com_vz, slip, jump_observer):
        from .terrain_curriculum import SPIN,JUMP
        s,t = self.s,self.s.terrain
        self.roll,self.pitch = roll_pitch_deg(gravity)
        self.speed,self.com_vz,self.body_vz = speed,com_vz,velocity_z.clone()
        tilt = torch.maximum(self.roll.abs(),self.pitch.abs())
        self.episode_max_tilt = torch.maximum(self.episode_max_tilt,tilt)
        contact = ((force_z > s.spec["jump"]["force_threshold_n"]) & ((bottom-support) < s.spec["jump"]["landing_clearance_m"])).all(-1)
        first_air_land = s.mask("airborne") & ~self.air_landed & contact
        self.air_land_time = torch.where(first_air_land,age,self.air_land_time)
        self.air_landed |= first_air_land
        jump_lane = s.mask("slope") & (t.mode == JUMP)
        first_land = jump_lane & jump_observer.flown & ~self.slope_landed & contact
        self.slope_land_time = torch.where(first_land,age,self.slope_land_time)
        self.slope_landed |= first_land
        self.slope_jump_offset = torch.where(first_land,(root_xy-self.hold.jump_anchor).norm(dim=-1),self.slope_jump_offset)
        ref = s.reference(age)
        window = jump_lane & (ref["push"] | ref["after_push"]) & (age >= s.delay) & (~self.slope_landed
            | (age < self.slope_land_time+s.spec["slope"]["jump"]["level_window_after_landing_s"]))
        self.slope_jump_window = window
        self.slope_interval_window |= window
        self.slope_jump_tilt = torch.maximum(self.slope_jump_tilt,tilt*window)
        spin_cfg = s.spec["slope"]["spin"]
        spin = s.mask("slope") & t.active & (t.mode == SPIN)
        in_segment = spin & (age >= spin_cfg["pre_hold_s"]) & (age < spin_cfg["pre_hold_s"]+spin_cfg["spin_s"])
        start = in_segment & ~self.spin_started
        self.spin_anchor[start] = root_xy[start]
        self.spin_started |= in_segment
        radius = (root_xy-self.spin_anchor).norm(dim=-1)
        self.slope_spin_radius = torch.maximum(self.slope_spin_radius,radius*in_segment)
        self.slope_spin_tilt = torch.maximum(self.slope_spin_tilt,tilt*in_segment)
        end = spin & self.spin_started & ~self.spin_finished & (age >= spin_cfg["pre_hold_s"]+spin_cfg["spin_s"])
        c = s.spec["slope"]["spin_success"]
        passed = end & (self.slope_spin_radius <= c["max_radius_m"]) & (self.slope_spin_tilt <= c["max_tilt_deg"])
        self.slope_event += passed*c["bonus"]
        self.spin_success |= passed
        self.spin_finished |= end
        self.step_down.observe(age,wheel_xy,bottom,support,force_z,height,velocity_z,self.roll,self.pitch,speed,cmd,nonwheel)
        self.bump.observe(age,wheel_xy,self.roll,self.pitch,speed,cmd,
                          height-s.height,nonwheel)
        self.balls.observe(root_xy,self.roll,self.pitch,slip)

    def finish(self, timeout, terminated, jump_observer):
        from .terrain_curriculum import JUMP
        self.balls.finish(timeout,terminated)
        self.step_down.success_event &= ~terminated
        self.step_down.success &= ~terminated
        self.spin_success &= ~terminated
        self.slope_event *= ~terminated
        done = timeout | terminated
        slope_jump = self.s.mask("slope") & (self.s.terrain.mode == JUMP)
        jump_observer.succeeded[slope_jump & terminated] = False
        jump_observer.success_event[slope_jump & terminated] = False
        for task,row in self.totals.items():
            if task == "slope_jump":
                mask,passed,tilt,radius = slope_jump,jump_observer.succeeded,self.slope_jump_tilt,self.slope_jump_offset
            elif task == "slope":
                mask = self.s.mask(task) & self.s.terrain.active & ~slope_jump
                passed,tilt,radius = self.spin_success,self.episode_max_tilt,self.slope_spin_radius
            elif task == "step_down":
                mask,passed,tilt,radius = self.s.mask(task) & self.s.terrain.active,self.step_down.success,self.step_down.max_tilt_w,self.hold.radius_max
            elif task == "balls":
                mask,passed,tilt,radius = self.s.mask(task) & self.s.terrain.active,self.balls.success_event,self.balls.max_tilt,self.balls.max_radius
            else:
                mask = self.s.mask(task) & self.s.terrain.active
                passed = (self.bump.settled.any(-1) & ~self.bump.nonwheel_seen.any(-1)
                    & (self.bump.max_tilt.amax(-1) <= self.s.spec["bump"]["success"]["max_tilt_deg"])
                    & (self.bump.min_ratio.amin(-1) >= self.s.spec["bump"]["success"]["min_speed_ratio"]))
                tilt,radius = self.bump.max_tilt.amax(-1),self.hold.radius_max
            selected = mask & done
            row["completed"] += selected.sum()
            row["success"] += (selected & passed & ~terminated).sum()
            row["terminated"] += (selected & terminated).sum()
            row["tilt_sum"] += (tilt*selected).sum()
            row["radius_sum"] += (radius*selected).sum()

    def reward_terms(self, age, policy_dt):
        c,s,t = self.s.spec["level"],self.s,self.s.terrain
        valid = (age >= c["reset_grace_s"]) & ~(s.mask("airborne") & self.air_landed
            & (age < self.air_land_time+c["airborne_grace_after_landing_s"]))
        r,p = self.roll,self.pitch
        slope = s.mask("slope") & t.active & ~self.slope_interval_window
        terms = dict(level_roll=valid*deadband_square(r,c["roll_deadband_deg"],c["roll_scale_deg"],c["clamp"]),
            level_pitch=valid*deadband_square(p,c["pitch_deadband_deg"],c["pitch_scale_deg"],c["clamp"]),
            slope_level=slope*(deadband_square(r,1.5,2.,25.)+.5*deadband_square(p,1.5,2.,25.)),
            slope_jump_level=self.slope_interval_window*(deadband_square(r,3.,3.,25.)+deadband_square(p,3.,3.,25.)),
            slope_events=self.slope_event/policy_dt)
        terms.update(self.hold.reward_terms())
        terms.update(self.step_down.reward_terms(r,p,self.speed,s.commands[:,0],self.com_vz,policy_dt))
        terms.update(self.bump.reward_terms(r,p,self.speed,s.commands[:,0],self.body_vz,policy_dt))
        terms.update(self.balls.reward_terms(r,p,policy_dt))
        return terms

    def metrics(self):
        result = {}
        for task,row in self.totals.items():
            prefix = "Mixed/"+task+"/"
            result[prefix+"completed_episodes_total"] = row["completed"]
            result[prefix+"success_fraction"] = row["success"]/row["completed"].clamp_min(1.)
            result[prefix+"terminated_episodes_total"] = row["terminated"]
            if task == "slope_jump":
                result[prefix+"landing_tilt_max_deg_mean"] = row["tilt_sum"]/row["completed"].clamp_min(1.)
                result[prefix+"landing_offset_m_mean"] = row["radius_sum"]/row["completed"].clamp_min(1.)
        return result

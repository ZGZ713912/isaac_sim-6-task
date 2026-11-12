"""Measured jump stages, energy objectives and episode accounting (reward only).

No value in this module is returned to the actor or command clock. Forces are
world Z components; electrical effort excludes spring and passive friction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

COUNTERS = ("completed", "succeeded", "terminated", "flown", "landed", "apex_seen",
            "push_attempted", "tuck_held", "extended", "buffered")
AXES = ("left_hip", "left_knee", "right_hip", "right_knee")


@dataclass
class JumpPhysics:
    com_height: torch.Tensor
    com_vz: torch.Tensor
    wheel_force_z: torch.Tensor
    motor_effort: torch.Tensor
    joint_velocity: torch.Tensor
    chassis_clearance_lower_bound: torch.Tensor


def validate_objective(j):
    spec = j["sequence_objective"]
    positive = ("energy_height_scale_m", "push_velocity_scale_m_s", "release_energy_bonus",
        "push_velocity_scale", "clearance_scale_m", "clearance_reward_scale",
        "tuck_hold_s", "extension_lead_s", "urgent_extension_lead_s",
        "minimum_reextension_m", "minimum_buffer_compression_m", "leg_motor_cap_nm",
        "minimum_flight_com_rise_m")
    if any(not math.isfinite(float(spec[k])) or spec[k] <= 0. for k in positive):
        raise ValueError("Jump sequence scales must be positive and finite")
    if (spec["urgent_extension_lead_s"] >= spec["extension_lead_s"]
            or spec["leg_motor_cap_nm"] != 40.
            or not 0. <= spec["push_torque_cost_fraction"] <= 1.):
        raise ValueError("Invalid jump stage timing, torque cap or effort cost")
    return spec


def energy_shape(vz, scale_m):
    """Bounded, monotone upward kinetic energy per unit weight; no target peak."""
    return -torch.expm1(-vz.clamp_min(0.).square()/(2.*9.81*scale_m))


def descent_time(clearance, vz):
    """Ballistic time estimate to the support under the wheels, reward only."""
    return (vz+(vz.square()+2.*9.81*clearance.clamp_min(0.)).sqrt())/9.81


class JumpSequence:
    """Per-physics-tick measurements; per-episode totals survive logger windows."""
    def __init__(self, schedule, dt, mass):
        self.schedule, self.dt, self.mass = schedule, dt, mass
        self.spec = validate_objective(schedule.spec["jump"])
        z = schedule.height
        names = ("release_vz", "release_com_height", "flight_com_rise", "air_start_age",
            "tuck_time", "tuck_delay", "reextension",
            "buffer_compression", "max_tuck", "max_wheel_clearance", "chassis_clearance_at_apex",
            "push_work", "push_impulse", "push_duration", "push_utilization_integral",
            "push_saturation_time", "push_force_peak_bw", "landing_force_peak_bw",
            "landing_down_vz", "interval_landing_force", "interval_release_bonus",
            "interval_push_payment", "interval_clearance_payment", "best_push_shape")
        self.values = {key: torch.zeros_like(z) for key in names}
        for key in ("capture_valid", "tuck_held", "extended", "buffered", "apex_sampled", "recorded"):
            self.values[key] = torch.zeros_like(z, dtype=torch.bool)
        for key in ("push_axis_effort_integral", "push_axis_peak_effort", "push_axis_saturation_time"):
            self.values[key] = torch.zeros(len(z), 4, device=z.device)
        self.push = torch.zeros_like(z, dtype=torch.bool)
        self.last_push_density = torch.zeros_like(z)
        self.last_clearance_density = torch.zeros_like(z)
        self.totals = {task: {} for task in ("jump", "airborne", "platform")
                       if task in schedule.tasks}

    def reset(self, ids):
        for value in self.values.values():
            value[ids] = 0
        self.push[ids] = False
        self.last_push_density[ids] = self.last_clearance_density[ids] = 0.

    def begin_interval(self):
        self.values["interval_landing_force"].zero_()
        self.values["interval_release_bonus"].zero_()
        self.values["interval_push_payment"].zero_()
        self.values["interval_clearance_payment"].zero_()

    def measure(self, observer, age, air, touching, first_air, lost_candidate,
                wheel_clearance, nonwheel_force, gravity, physics):
        if physics is None:
            raise ValueError("v8 jump objectives require measured COM, force and electrical effort")
        n = len(self.mass)
        if (physics.com_height.shape != (n,) or physics.com_vz.shape != (n,)
                or physics.wheel_force_z.shape != (n, 2)
                or physics.motor_effort.shape != (n, 4) or physics.joint_velocity.shape != (n, 4)
                or physics.chassis_clearance_lower_bound.shape != (n,)):
            raise ValueError("Invalid jump physics shapes")
        s, v = self.schedule, self.values
        ref = s.reference(age)
        active = ref["active"]
        for key in ("release_vz", "release_com_height", "air_start_age"):
            v[key][lost_candidate] = 0.
        v["capture_valid"][lost_candidate] = False
        v["capture_valid"] |= first_air
        for key, measured in (("release_vz", physics.com_vz),
                              ("release_com_height", physics.com_height), ("air_start_age", age)):
            v[key] = torch.where(first_air, measured, v[key])
        # Support and true upward movement are needed: a fast spinning wheel
        # or held torque against a stop cannot earn a push velocity reward.
        support = (physics.wheel_force_z.sum(-1) > observer.weight*.1)
        upright = gravity[:, 2] < -math.cos(math.radians(20.))
        self.push = (ref["requested"] & active & (ref["push"] | ref["after_push"])
                     & ~observer.flown & ~air & support & upright & (nonwheel_force <= 2.))
        force_z = physics.wheel_force_z.sum(-1)
        effort = physics.motor_effort
        positive_power = (effort*physics.joint_velocity).clamp_min(0.).sum(-1)
        utilization = effort.abs()/self.spec["leg_motor_cap_nm"]
        v["push_work"] += self.push*positive_power*self.dt
        v["push_impulse"] += self.push*(force_z-observer.weight)*self.dt
        v["push_duration"] += self.push*self.dt
        v["push_utilization_integral"] += self.push*utilization.mean(-1)*self.dt
        v["push_saturation_time"] += self.push*(utilization >= .95).float().mean(-1)*self.dt
        v["push_axis_effort_integral"] += self.push[:, None]*effort.abs()*self.dt
        v["push_axis_saturation_time"] += self.push[:, None]*(utilization >= .95)*self.dt
        v["push_axis_peak_effort"] = torch.maximum(v["push_axis_peak_effort"], self.push[:, None]*effort.abs())
        v["push_force_peak_bw"] = torch.maximum(v["push_force_peak_bw"],
            self.push*force_z/observer.weight)
        progress = self.push*energy_shape(physics.com_vz,
            self.spec["push_velocity_scale_m_s"]**2/(2.*9.81))
        best = torch.maximum(v["best_push_shape"], progress)
        gain = self.spec["push_velocity_scale"]*(best-v["best_push_shape"])
        v["best_push_shape"] = best
        # Improvement pays once, independent of push duration or the 200 Hz
        # sampling rate. Repeated sub-threshold bounces cannot farm velocity.
        self.last_push_density = gain/self.dt
        v["interval_push_payment"] += gain
        # Save the first collision, even when flight and landing both occur
        # inside one policy interval. Push contact never enters this payment.
        landing = active & observer.flown & touching
        force = force_z.clamp_min(0.)*landing
        v["interval_landing_force"] = torch.maximum(v["interval_landing_force"], force)
        v["landing_force_peak_bw"] = torch.maximum(v["landing_force_peak_bw"], force/observer.weight)
        first_touch = landing & ~observer.landed
        v["landing_down_vz"] = torch.where(first_touch, (-physics.com_vz).clamp_min(0.), v["landing_down_vz"])
        # Clearance uses actual wheel cylinder bottoms, including raised support.
        self.last_clearance_density = (active & observer.flown & ~observer.landed & air).float()*self.spec[
            "clearance_reward_scale"]*(-torch.expm1(-wheel_clearance.amin(-1).clamp_min(0.)/
                                                  self.spec["clearance_scale_m"]))
        v["interval_clearance_payment"] += self.last_clearance_density*self.dt
        return physics.com_vz

    def extension_ready(self, observer, wheel_clearance, vz):
        time = descent_time(wheel_clearance.amin(-1), vz)
        hold = observer.apex_hold_time >= observer.j["apex_tuck_hold_seconds"]-1e-7
        ready = (hold & (time <= self.spec["extension_lead_s"])) | (time <= self.spec["urgent_extension_lead_s"])
        return (vz < 0.) & ready

    def measure_legs(self, observer, age, in_flight, air, leg_lengths, bilateral_delta):
        v, j = self.values, observer.j
        held = in_flight & air & ~observer.extension_started & (bilateral_delta >= j["minimum_bilateral_tuck_delta_m"])
        v["tuck_time"] = torch.where(held, v["tuck_time"]+self.dt, 0.)
        reached = held & (v["tuck_time"] >= self.spec["tuck_hold_s"]-1e-7)
        first = reached & ~v["tuck_held"]
        v["tuck_delay"] = torch.where(first, age-v["air_start_age"], v["tuck_delay"])
        v["tuck_held"] |= reached
        # The extension check is bilateral; the less extended leg decides.
        if "minimum_leg_lengths" not in v:
            v["minimum_leg_lengths"] = torch.zeros_like(leg_lengths)
        minimum = v["minimum_leg_lengths"]
        new = in_flight & (minimum.amin(-1) <= 0.)
        minimum = torch.where(new[:, None], leg_lengths, minimum)
        minimum = torch.where((in_flight & air)[:, None], torch.minimum(minimum, leg_lengths), minimum)
        v["minimum_leg_lengths"] = minimum
        delta = (leg_lengths-minimum).amin(-1).clamp_min(0.)
        v["reextension"] = torch.maximum(v["reextension"],
            delta*(in_flight & air & observer.extension_started & v["tuck_held"]))
        v["extended"] |= v["reextension"] >= self.spec["minimum_reextension_m"]
        v["max_tuck"] = torch.maximum(v["max_tuck"], bilateral_delta*(in_flight & air))
        return v["tuck_held"]

    def after_observe(self, observer, new_flight, landing, leg_lengths, wheel_clearance, physics):
        v = self.values
        v["interval_release_bonus"] += (new_flight & self.schedule.requesters())*self.spec[
            "release_energy_bonus"]*energy_shape(v["release_vz"], self.spec["energy_height_scale_m"])
        flying = observer.flown & ~observer.landed
        v["flight_com_rise"] = torch.maximum(v["flight_com_rise"],
            (physics.com_height-v["release_com_height"]).clamp_min(0.)*flying)
        v["max_wheel_clearance"] = torch.maximum(v["max_wheel_clearance"],
            wheel_clearance.amin(-1).clamp_min(0.)*flying)
        apex = observer.apex_seen & ~v["apex_sampled"]
        v["chassis_clearance_at_apex"] = torch.where(apex, physics.chassis_clearance_lower_bound,
            v["chassis_clearance_at_apex"])
        v["apex_sampled"] |= apex
        if leg_lengths is not None:
            if "touch_leg_lengths" not in v:
                v["touch_leg_lengths"] = torch.zeros_like(leg_lengths)
            v["touch_leg_lengths"] = torch.where(landing[:, None], leg_lengths, v["touch_leg_lengths"])
            compression = (v["touch_leg_lengths"]-leg_lengths).amin(-1).clamp_min(0.)
            v["buffer_compression"] = torch.maximum(v["buffer_compression"], compression*observer.landed)
            v["buffered"] |= v["buffer_compression"] >= self.spec["minimum_buffer_compression_m"]

    def success_ready(self):
        return (self.values["tuck_held"] & self.values["extended"] & self.values["buffered"]
                & (self.values["flight_com_rise"] >= self.spec["minimum_flight_com_rise_m"]))

    def record_outcomes(self, observer, done, terminated):
        v = self.values
        done = done & ~v["recorded"]
        for task, totals in self.totals.items():
            mask = done & self.schedule.mask(task)
            counts = dict(completed=mask, succeeded=mask & observer.succeeded,
                terminated=mask & terminated, flown=mask & observer.flown,
                landed=mask & observer.landed, apex_seen=mask & v["apex_sampled"],
                push_attempted=mask & (v["push_duration"] > 0.),
                tuck_held=mask & v["tuck_held"], extended=mask & v["extended"], buffered=mask & v["buffered"])
            for name, selected in counts.items():
                totals[name] = totals.get(name, torch.zeros((), device=mask.device, dtype=torch.long))+selected.sum()
            for name in ("release_vz", "flight_com_rise", "max_tuck", "max_wheel_clearance",
                         "push_work", "push_impulse", "push_duration", "push_utilization_integral",
                         "push_saturation_time", "push_force_peak_bw", "landing_force_peak_bw",
                         "landing_down_vz", "buffer_compression", "reextension", "tuck_delay", "chassis_clearance_at_apex"):
                sample = counts[self.denominator(name)]
                totals[name] = totals.get(name, torch.zeros((), device=mask.device))+(v[name]*sample).sum()
            for name in ("push_axis_effort_integral", "push_axis_peak_effort", "push_axis_saturation_time"):
                sample = counts["push_attempted"][:, None]
                totals[name] = totals.get(name, torch.zeros(4, device=mask.device))+(v[name]*sample).sum(0)
        v["recorded"] |= done

    @staticmethod
    def denominator(name):
        if name.startswith("push_"):
            return "push_attempted"
        if name == "tuck_delay":
            return "tuck_held"
        if name == "chassis_clearance_at_apex":
            return "apex_seen"
        if name in ("landing_force_peak_bw", "landing_down_vz", "buffer_compression"):
            return "landed"
        return "flown"

    def metrics(self):
        logs = {}
        units = dict(release_vz="m_s", flight_com_rise="m", max_tuck="m", max_wheel_clearance="m",
            push_work="J", push_impulse="N_s", push_duration="s", push_force_peak_bw="bodyweights",
            landing_force_peak_bw="bodyweights", landing_down_vz="m_s", buffer_compression="m",
            reextension="m", tuck_delay="s", chassis_clearance_at_apex="lower_bound_m")
        for task, totals in self.totals.items():
            prefix = "JumpSequence/"+task+"/"
            if not totals:
                continue
            for name in COUNTERS:
                logs[prefix+name+"_episodes_total"] = totals[name].clone().float()
            logs[prefix+"success_fraction"] = totals["succeeded"].float()/totals["completed"].clamp_min(1)
            for name, unit in units.items():
                count = totals[self.denominator(name)]
                logs[prefix+name+"_mean_"+unit] = (totals[name]/count.clamp_min(1)).clone()
            for name in ("push_utilization_integral", "push_saturation_time"):
                label = "push_mean_torque_fraction" if name.endswith("integral") else "push_saturation_fraction"
                logs[prefix+label] = (totals[name]/totals["push_duration"].clamp_min(1e-9)).clone()
            for i, axis in enumerate(AXES):
                logs[prefix+axis+"/push_abs_effort_mean_Nm"] = (totals["push_axis_effort_integral"][i]/
                    totals["push_duration"].clamp_min(1e-9)).clone()
                logs[prefix+axis+"/push_episode_peak_mean_Nm"] = (totals["push_axis_peak_effort"][i]/
                    totals["push_attempted"].clamp_min(1)).clone()
                logs[prefix+axis+"/push_saturation_fraction"] = (totals["push_axis_saturation_time"][i]/
                    totals["push_duration"].clamp_min(1e-9)).clone()
        return logs

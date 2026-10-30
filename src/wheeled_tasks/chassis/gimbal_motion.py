"""Independent, torque-limited gimbal motion used as a chassis disturbance.

The yaw velocity servo and pitch position servo follow wheelbipe V14's
structure. Pitch scans smoothly through the user's mechanical range; none of
these targets control a leg/wheel or write a physical state during stepping.
"""
from copy import deepcopy
import math

import torch


def motion_contract():
    return dict(schema="full-vehicle-gimbal-motion-v1", joint_names=["M_joint1", "M_joint2"],
        yaw_velocity_range_rad_s=[-math.pi, math.pi], yaw_kd=.5, yaw_effort_limit_nm=2.,
        yaw_acceleration_rad_s2=math.pi, pitch_limits_deg=[-30., 35.],
        pitch_kp=20., pitch_kd=.5, pitch_effort_limit_nm=10., pitch_period_seconds=[8., 16.],
        pitch_positive="up", pitch_trajectory="bounded_sinusoid_random_period_and_direction",
        yaw_sampling="uniform_per_episode", disabled="zero_motor_effort_and_pause_motion_clock",
        reference="wheelbipe_V14_2 yaw velocity PD and pitch position PD; engineering disturbance, not identified motors",
        reference_pitch="wheelbipe holds a fixed pitch; this task adds a smooth bounded scan",
        state_writes="episode_reset_only", policy_motor_outputs=6)


class GimbalMotion:
    def __init__(self, count, device, dt, contract, seed):
        if contract != motion_contract() or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Gimbal motion requires its versioned servo/limit contract")
        self.cfg, self.dt = deepcopy(contract), dt
        self.rng = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0x47494D42)
        self.time = torch.zeros(count, device=device)
        self.yaw_goal = torch.zeros_like(self.time)
        self.yaw_velocity = torch.zeros_like(self.time)
        self.phase = torch.zeros_like(self.time)
        self.period = torch.zeros_like(self.time)
        self.target = torch.zeros(count, 2, device=device)
        self.effort = torch.zeros_like(self.target)
        self.reset(torch.arange(count, device=device))

    def reset(self, ids):
        draws = torch.rand(len(ids), 3, generator=self.rng).to(self.time.device)
        self.time[ids] = self.yaw_velocity[ids] = 0.
        self.yaw_goal[ids] = (2.*draws[:, 0]-1.)*math.pi
        lo, hi = self.cfg["pitch_period_seconds"]
        self.period[ids] = lo+(hi-lo)*draws[:, 1]
        lower, upper = (math.radians(v) for v in self.cfg["pitch_limits_deg"])
        midpoint, amplitude = (lower+upper)/2., (upper-lower)/2.
        initial = math.asin(-midpoint/amplitude)
        self.phase[ids] = torch.where(draws[:, 2] < .5, initial, math.pi-initial)
        self.target[ids] = self.effort[ids] = 0.

    def step(self, q, dq, enabled):
        delta = self.yaw_goal-self.yaw_velocity
        limit = self.cfg["yaw_acceleration_rad_s2"]*self.dt
        self.yaw_velocity.add_(delta.clamp(-limit, limit)*enabled)
        self.time.add_(enabled*self.dt)
        lower, upper = (math.radians(v) for v in self.cfg["pitch_limits_deg"])
        pitch = (lower+upper)/2.+(upper-lower)/2.*torch.sin(math.tau*self.time/self.period+self.phase)
        self.target[:, 0] = self.yaw_velocity
        self.target[:, 1] = pitch.clamp(lower, upper)
        self.effort[:, 0] = (self.cfg["yaw_kd"]*(self.yaw_velocity-dq[:, 0])).clamp(
            -self.cfg["yaw_effort_limit_nm"], self.cfg["yaw_effort_limit_nm"])
        self.effort[:, 1] = (self.cfg["pitch_kp"]*(self.target[:, 1]-q[:, 1])
            -self.cfg["pitch_kd"]*dq[:, 1]).clamp(
            -self.cfg["pitch_effort_limit_nm"], self.cfg["pitch_effort_limit_nm"])
        self.effort.mul_(enabled[:, None])
        return self.effort

"""Explicit sim-to-real domains; no privileged domain value enters the actor."""
from copy import deepcopy

import torch

from .unified_getup import settings as getup_settings


def settings(**overrides):
    cfg = getup_settings()
    cfg.update(version="robust-v2", nominal_fraction=.25, motor_strength_range=[.85, 1.10],
        leg_kp_scale_range=[.9, 1.1], leg_kd_scale_range=[.85, 1.15],
        spring_strength_range=[.92, 1.08], torque_delay_ms=[0, 3], motor_tau_ms=[0., 5.],
        sensor_delay_policy_ticks=[0, 2], encoder_bias_rad=.003, encoder_noise_rad=.001,
        gyro_bias_rad_s=.015, gyro_noise_rad_s=.01, gravity_noise=.004,
        velocity_noise_rad_s=[.05, .25], sensor_noise_scale=1., domains_enabled=True,
        quiet_action_rate_cost=.15, quiet_motor_speed_cost=.025, quiet_effort_delta_cost=.03,
        quiet_actor_alpha=[.25, .25, .25, .25, .65, .65])
    cfg.update(overrides)
    return cfg


def rigid_body_domain(enabled):
    return dict(enabled=enabled, sampling="startup", enabled_fraction=.75,
        base_body="base_link", wheel_bodies=["L_link3", "R_link3"],
        base_mass_scale=[.9, 1.1], leg_mass_scale=[.95, 1.05], wheel_mass_scale=[.95, 1.05],
        inertia_scale=[.9, 1.1], base_com_offset_m=dict(x=[-.008, .008], y=[-.005, .005], z=[-.005, .005]),
        source_reference="bounded robustness hypotheses, not identified hardware parameters")


class RobustDomain:
    def __init__(self, count, device, cfg, seed, dt):
        self.cfg, self.dt = deepcopy(cfg), dt
        self.device = device
        self.rng = torch.Generator(device=device).manual_seed(seed + 83977)
        self.enabled = torch.rand(count, device=device, generator=self.rng) >= cfg["nominal_fraction"]
        if not cfg["domains_enabled"]:
            self.enabled.zero_()
        self.strength = torch.ones(count, 6, device=device)
        self.kp = torch.ones(count, 1, device=device)
        self.kd = self.kp.clone()
        self.spring = self.kp.clone()
        self.torque_delay = torch.zeros(count, dtype=torch.long, device=device)
        self.sensor_delay = self.torque_delay.clone()
        self.motor_alpha = torch.ones(count, 1, device=device)
        self.bias = torch.zeros(count, 18, device=device)
        self.sigma = self.bias.clone()
        self.torque_history = torch.zeros(5, count, 6, device=device)
        self.torque_state = torch.zeros(count, 6, device=device)
        self.sensor_history = torch.zeros(3, count, 18, device=device)
        self.observed = torch.zeros(count, 18, device=device)
        self.torque_cursor = self.sensor_cursor = 0
        self.last_sensor_tick = -1
        self.rows = torch.arange(count, device=device)

    def uniform(self, ids, bounds, columns):
        low, high = bounds
        return low + (high - low) * torch.rand(len(ids), columns, device=self.device, generator=self.rng)

    def reset(self, ids, sensors):
        cfg = self.cfg
        enabled = self.enabled[ids, None]
        for name, key, columns in (("strength", "motor_strength_range", 6), ("kp", "leg_kp_scale_range", 1),
                                  ("kd", "leg_kd_scale_range", 1), ("spring", "spring_strength_range", 1)):
            getattr(self, name)[ids] = torch.where(enabled, self.uniform(ids, cfg[key], columns), 1.)
        tau = self.uniform(ids, cfg["motor_tau_ms"], 1) * .001
        self.motor_alpha[ids] = torch.where(enabled, self.dt / (self.dt + tau), 1.)
        for name, key in (("torque_delay", "torque_delay_ms"), ("sensor_delay", "sensor_delay_policy_ticks")):
            low, high = cfg[key]
            draw = torch.randint(low, high + 1, (len(ids),), device=self.device, generator=self.rng)
            getattr(self, name)[ids] = torch.where(enabled[:, 0], draw, 0)
        self.bias[ids] = 0.
        self.bias[ids, :3] = self.uniform(ids, [-cfg["gyro_bias_rad_s"] / 2, cfg["gyro_bias_rad_s"] / 2], 3)
        self.bias[ids, 6:10] = self.uniform(ids, [-cfg["encoder_bias_rad"], cfg["encoder_bias_rad"]], 4)
        self.bias[ids] *= enabled
        sigma = sensors.new_tensor([cfg["gyro_noise_rad_s"] / 2] * 3 + [cfg["gravity_noise"]] * 3
            + [cfg["encoder_noise_rad"]] * 4 + [0.] * 2
            + [cfg["velocity_noise_rad_s"][0] * .1] * 4 + [cfg["velocity_noise_rad_s"][1] * .1] * 2)
        self.sigma[ids] = sigma * enabled * cfg["sensor_noise_scale"]
        self.sensor_history[:, ids] = sensors[None]
        self.observed[ids] = sensors
        self.torque_history[:, ids] = 0.
        self.torque_state[ids] = 0.

    def observe(self, sensors, tick):
        if self.last_sensor_tick != tick:
            noisy = sensors + self.bias + torch.randn(sensors.shape, device=self.device, generator=self.rng) * self.sigma
            gravity = noisy[:, 3:6]
            noisy[:, 3:6] = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            self.sensor_history[self.sensor_cursor].copy_(noisy)
            read = (self.sensor_cursor - self.sensor_delay) % len(self.sensor_history)
            self.observed.copy_(self.sensor_history[read, self.rows])
            self.sensor_cursor = (self.sensor_cursor + 1) % len(self.sensor_history)
            self.last_sensor_tick = tick
        return self.observed

    def apply_torque(self, torque, passive):
        requested = torque * self.strength
        requested = torch.where(passive[:, None], 0., requested)
        self.torque_history[self.torque_cursor].copy_(requested)
        read = (self.torque_cursor - self.torque_delay) % len(self.torque_history)
        delayed = self.torque_history[read, self.rows]
        self.torque_state.add_(self.motor_alpha * (delayed - self.torque_state))
        self.torque_state[passive] = 0.
        self.torque_cursor = (self.torque_cursor + 1) % len(self.torque_history)
        return self.torque_state

    def summary(self):
        return dict(enabled_environments=int(self.enabled.sum()), nominal_environments=int((~self.enabled).sum()),
            strength_min=float(self.strength.min()), strength_max=float(self.strength.max()),
            torque_delay_ms_max=int(self.torque_delay.max()), sensor_delay_ms_max=int(self.sensor_delay.max()) * 5,
            domains=deepcopy(self.cfg), parameter_status="robustness assumptions, not real-to-sim identification")

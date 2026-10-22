"""Explicit sim-to-real domains; no privileged domain value enters the actor."""
from copy import deepcopy

import torch

from .unified_getup import settings as getup_settings


def settings(**overrides):
    cfg = getup_settings()
    cfg.update(version="robust-v2.9", stance_aware_recovery=True, recovery_shape_aware=False,
        paired_feedback_winding=True, cyclic_leg_targets=True, leg_action_clip=16., recovery_leg_step=.3,
        smooth_knee_targets=True, recovery_knee_center=[.924544, .924537],
        recovery_leg_speed_cost=.0005,
        nominal_fraction=.25, motor_strength_range=[.85, 1.10],
        leg_kp_scale_range=[.9, 1.1], leg_kd_scale_range=[.85, 1.15],
        spring_strength_range=[.92, 1.08], torque_delay_ms=[0, 3], motor_tau_ms=[0., 5.],
        sensor_delay_policy_ticks=[0, 2], encoder_bias_rad=.003, encoder_noise_rad=.001,
        gyro_bias_rad_s=.015, gyro_noise_rad_s=.01, gravity_noise=.004,
        velocity_noise_rad_s=[.05, .25], sensor_noise_scale=1., domains_enabled=True,
        quiet_action_rate_cost=.15, quiet_motor_speed_cost=.025, quiet_effort_delta_cost=.03,
        quiet_actor_alpha=[.25, .25, .25, .25, .65, .65],
        arc_fraction_within_motion=.4, boundary_xy_m=[40., 40.], push_torque_direction=-1.,
        hold_motion_during_recovery=False, canonical_recovery=True, motion_residual_fraction=.25,
        domain_scale=1., pd_feedback_delay_ms=[0, 3], pd_feedback_period_ms=[1, 5], pd_encoder_bias_rad=.003,
        reward_gamma=.998, reward_potential_gain=5., push_torque_nm=32., push_duration_seconds=.3,
        quiet_frozen=False, quiet_training=False, quiet_sway_m=.05,
        quiet_spectral_cost=False, quiet_critic_anchor=False,
        quiet_velocity_feature_scale=None, quiet_wheel_bound=.5, arc_tuning_fraction=0.,
        standing_support_training=False, quiet_position_weight=.5, quiet_height_probabilities=[.5, .25, .25],
        quiet_high_common_alpha=None, quiet_high_leg_prediction_s=0., standing_early_position_weight=0.,
        fast_forward_reference=False, fast_forward_reference_range=[1.5, 2.],
        fast_forward_reference_yaw_fade=.25)
    cfg.update(overrides)
    scale = cfg["domain_scale"]
    if not 0. <= scale <= 1.:
        raise ValueError("Robust domain scale must be in [0,1]")
    for key in ("motor_strength_range", "leg_kp_scale_range", "leg_kd_scale_range", "spring_strength_range"):
        cfg[key] = [1. + scale * (v - 1.) for v in cfg[key]]
    for key in ("encoder_bias_rad", "encoder_noise_rad", "gyro_bias_rad_s", "gyro_noise_rad_s", "gravity_noise", "pd_encoder_bias_rad"):
        cfg[key] *= scale
    cfg["velocity_noise_rad_s"] = [v * scale for v in cfg["velocity_noise_rad_s"]]
    cfg["motor_tau_ms"] = [v * scale for v in cfg["motor_tau_ms"]]
    for key in ("torque_delay_ms", "sensor_delay_policy_ticks", "pd_feedback_delay_ms"):
        cfg[key] = [0, round(cfg[key][1] * scale)]
    cfg["pd_feedback_period_ms"] = [1, max(1, round(1. + scale * (cfg["pd_feedback_period_ms"][1] - 1.)))]
    return cfg


def rigid_body_domain(enabled, scale=1.):
    cfg = dict(enabled=enabled, sampling="startup", enabled_fraction=.75,
        base_body="base_link", wheel_bodies=["L_link3", "R_link3"],
        base_mass_scale=[.9, 1.1], leg_mass_scale=[.95, 1.05], wheel_mass_scale=[.95, 1.05],
        inertia_scale=[.9, 1.1], base_com_offset_m=dict(x=[-.008, .008], y=[-.005, .005], z=[-.005, .005]),
        source_reference="bounded robustness hypotheses, not identified hardware parameters")
    for key in ("base_mass_scale", "leg_mass_scale", "wheel_mass_scale", "inertia_scale"):
        cfg[key] = [1. + scale * (v - 1.) for v in cfg[key]]
    cfg["base_com_offset_m"] = {axis: [v * scale for v in values] for axis, values in cfg["base_com_offset_m"].items()}
    return cfg


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
            immediate = getattr(self, "evaluation_immediate_features", None)
            if immediate is not None:
                self.observed[:, immediate] = noisy[:, immediate]
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

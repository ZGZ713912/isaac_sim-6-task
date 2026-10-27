"""Opt-in real-robot gap pilot. These are hypotheses, not identified parameters."""
import math

import torch


def apply_profile(task, severity=1.):
    if not math.isfinite(severity) or not 0. < severity <= 1.:
        raise ValueError("Gap severity must be finite within (0,1]")
    timing = task.get("gap_actuator_timing_profile", "stress")
    if timing not in ("stress", "native"):
        raise ValueError("Unknown actuator timing profile")
    def around(low, high):
        return [1.+severity*(low-1.), 1.+severity*(high-1.)]
    task.update(gap_robustness_version="gap-v1", gap_severity=severity,
        gap_correct_pd_scaling=True,
        upright_entry_recovery=True, version="robust-gap-v1", motion_fraction=.5,
        recovery_family_weights=[.4, .05, .05, .125, .125, .2, .05],
        gap_floor_births=True, recovery_leg_speed_cost=.003,
        gap_leg_quality=dict(speed_weight=.004, saturation_weight=.5, slew_weight=.02,
                             speed_soft_limit_rad_s=6., saturation_fraction=.95),
        imu_mount=dict(fixed_degrees=[3.*severity, 3.*severity, severity],
                       wobble_degrees=[0., .5*severity], wobble_hz=[1., 8.]),
        motor_strength_range=around(.8, 1.2), leg_kp_scale_range=around(.85, 1.15),
        leg_kd_scale_range=around(.7, 1.3), spring_strength_range=around(.85, 1.15),
        motor_tau_ms=[0., 8.*severity], torque_delay_ms=[0, round(3.*severity)],
        pd_feedback_delay_ms=[0, round(3.*severity)], pd_feedback_period_ms=[1, max(1, round(1.+2.*severity))],
        encoder_bias_rad=.005*severity, pd_encoder_bias_rad=.005*severity,
        sensor_delay_policy_ticks=[0, 1],
        gap_parameter_status="bounded robustness hypotheses; no hardware identification claim")
    if timing == "native":
        # Isolate IMU/mass/COM/gain robustness from extra, unmeasured timing
        # perturbations. The authenticated native actuator response remains.
        task.update(motor_tau_ms=[0.,0.],torque_delay_ms=[0,0],
                    pd_feedback_delay_ms=[0,0],pd_feedback_period_ms=[1,1])
    return task


def rigid_profile(config, severity):
    config = dict(config)
    # User-confirmed full mass domain: each URDF component within +/-10%.
    for key, radius in (("base_mass_scale", .10), ("leg_mass_scale", .10),
                        ("wheel_mass_scale", .10), ("inertia_scale", .2)):
        config[key] = [1.-radius*severity, 1.+radius*severity]
    config["base_com_offset_m"] = {a: [-v*severity, v*severity] for a,v in dict(x=.015, y=.008, z=.01).items()}
    return config


def randomized_leg_pd(error, velocity, kp_scale, kd_scale, kp, kd):
    """Scale the requested PD torque before clipping, including at saturation."""
    requested = kp_scale*kp*error-kd_scale*kd*velocity
    return requested.clamp(-40., 40.), requested


def leg_quality(speed_peak, saturation_fraction, slew_rms, active, cfg):
    """Physical 1 ms statistics; recovery may use brief torque saturation."""
    speed = (speed_peak-cfg["speed_soft_limit_rad_s"]).clamp_min(0.).square()
    cost = (cfg["speed_weight"]*speed + cfg["saturation_weight"]*saturation_fraction
            + cfg["slew_weight"]*(slew_rms/10.).square())
    return cost * active

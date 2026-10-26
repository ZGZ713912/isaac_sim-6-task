"""Resolve one explicit, inspectable domain study after nominal task creation.

This opt-in path never calls the legacy gap overlay. It does not train, load a
checkpoint or change defaults. Limits below are research bounds, not hardware
identification or a deployment qualification.
"""
from copy import deepcopy
import math


GROUPS = ("observation_noise", "encoder_bias", "imu_alignment", "imu_wobble",
          "actuator_gains", "rigid_body", "ground_friction")


def unit(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0. <= value <= 1.:
        raise ValueError(f"{name} must be finite within [0,1]")
    return float(value)


def resolve(spec, profile=None, strength=1.):
    """Return final task/mechanism values; no second severity can overwrite them."""
    if spec.get("schema") != "wheel-leg-randomization-study-v1" or tuple(spec.get("groups", ())) != GROUPS:
        raise ValueError("Unsupported randomization study or group order")
    strength = unit(strength, "strength")
    nominal = unit(spec["nominal_fraction"], "nominal_fraction")
    profile = spec["default_profile"] if profile is None else profile
    if profile not in spec["profiles"]:
        raise ValueError("Unknown randomization profile")
    values = spec["profiles"][profile]
    if not isinstance(values, dict) or set(values) != set(GROUPS):
        raise ValueError("One explicit scale is required per group")
    scales = {name: unit(values[name], name)*strength for name in GROUPS}
    timing = deepcopy(spec["timing"])
    limits = dict(torque_delay_ms=(0, 4), sensor_delay_policy_ticks=(0, 2),
                  pd_feedback_delay_ms=(0, 7), pd_feedback_period_ms=(1, 7), motor_tau_ms=(0., 8.))
    if set(timing) != set(limits):
        raise ValueError("All discrete timing fields must be explicit")
    for key, (minimum, maximum) in limits.items():
        value = timing[key]
        if (not isinstance(value, list) or len(value) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
                or not minimum <= value[0] <= value[1] <= maximum
                or key != "motor_tau_ms" and any(type(v) is not int for v in value)):
            raise ValueError("Invalid explicit timing range: " + key)
    def around(radius, scale):
        return [1.-radius*scale, 1.+radius*scale]
    noise, bias, align, wobble, gains, rigid, friction = (scales[k] for k in GROUPS)
    active = any(scales.values()) or any(v != [1, 1] if k == "pd_feedback_period_ms" else v != [0, 0]
                                        for k, v in timing.items())
    task = dict(domains_enabled=active, nominal_fraction=nominal,
        motor_strength_range=around(.2, gains), leg_kp_scale_range=around(.15, gains),
        leg_kd_scale_range=around(.3, gains), spring_strength_range=around(.15, gains),
        encoder_bias_rad=.005*bias, pd_encoder_bias_rad=.005*bias,
        encoder_noise_rad=.001*noise, gyro_bias_rad_s=.015*noise,
        gyro_noise_rad_s=.01*noise, gravity_noise=.004*noise,
        velocity_noise_rad_s=[.05*noise, .25*noise], sensor_noise_scale=1.,
        gap_correct_pd_scaling=True, **timing)
    # IMU mounting adds its own angular velocity; it is never silently folded
    # into the much smaller gyro white-noise sigma.
    task["imu_mount"] = (dict(fixed_degrees=[3.*align, 3.*align, align],
                              wobble_degrees=[0., .5*wobble], wobble_hz=[1., 8.])
                         if align or wobble else None)
    dynamics = dict(enabled=active, sampling="startup", enabled_fraction=1.-nominal,
        base_body="base_link", wheel_bodies=["L_link3", "R_link3"],
        base_mass_scale=around(.1, rigid), leg_mass_scale=around(.1, rigid),
        wheel_mass_scale=around(.1, rigid), inertia_scale=around(.2, rigid),
        base_com_offset_m={axis: [-radius*rigid, radius*rigid]
                           for axis, radius in dict(x=.015, y=.008, z=.01).items()},
        source_reference="explicit randomization study; bounded hypotheses, not identified hardware")
    ground = [.5-.2*friction, .5]
    motor = task["motor_strength_range"]
    effective_kp = [a*b for a, b in zip(motor, task["leg_kp_scale_range"], strict=True)]
    effective_kd = [a*b for a, b in zip(motor, task["leg_kd_scale_range"], strict=True)]
    return dict(schema="wheel-leg-resolved-randomization-study-v1", profile=profile, strength=strength,
        group_scales=scales, task=task, dynamics_randomization=dynamics, ground_friction_range=ground,
        effective_wheel_ground_friction_range=[(v+.1)/2. for v in ground],
        effective_unsaturated_leg_kp_scale_bounds=effective_kp,
        effective_unsaturated_leg_kd_scale_bounds=effective_kd,
        imu_wobble_single_axis_gyro_peak_bound_rad_s=math.radians(.5*wobble)*2.*math.pi*8.,
        policy_PD_encoder_bias_disagreement_bound_rad=2.*task["encoder_bias_rad"],
        physical_encoder_white_noise_coverage=False,
        nominal_mask_scope="rigid-body/actuator/sensor shared mask; ground friction independently covers all lanes",
        hardware_identified=False, behavior_qualified=False)


def apply(config, task, resolved):
    if task.get("gap_robustness_version") or task.get("randomization_study"):
        raise ValueError("Do not stack two domain profiles or apply a study twice")
    task.update(deepcopy(resolved["task"]))
    task.pop("domain_scale", None)
    task["version"] += "-domain-study-v1"
    task["randomization_study"] = deepcopy(resolved)
    config["dynamics_randomization"] = deepcopy(resolved["dynamics_randomization"])
    for key in ("ground_static_friction_range", "ground_dynamic_friction_range"):
        config["contact_material"][key] = deepcopy(resolved["ground_friction_range"])


def scalar_metrics(resolved):
    """Stable physical-unit configuration tags, distinct from behavior results."""
    values = {f"group_scale/{key}": value for key, value in resolved["group_scales"].items()}
    task = resolved["task"]
    for key in ("encoder_noise_rad", "encoder_bias_rad", "pd_encoder_bias_rad", "gyro_bias_rad_s",
                "gyro_noise_rad_s", "gravity_noise", "nominal_fraction"):
        values[key] = task[key]
    for key in ("motor_strength_range", "leg_kp_scale_range", "leg_kd_scale_range", "spring_strength_range",
                "sensor_delay_policy_ticks", "torque_delay_ms", "motor_tau_ms",
                "pd_feedback_delay_ms", "pd_feedback_period_ms"):
        for label, value in zip(("min", "max"), task[key], strict=True):
            values[f"{key}/{label}"] = value
    values["imu_wobble_single_axis_gyro_peak_bound_rad_s"] = resolved["imu_wobble_single_axis_gyro_peak_bound_rad_s"]
    return values

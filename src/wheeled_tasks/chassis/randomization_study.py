"""Resolve one explicit, inspectable domain study after nominal task creation.

This opt-in path never calls the legacy gap overlay. It does not train, load a
checkpoint or change defaults. Limits below are research bounds, not hardware
identification or a deployment qualification.
"""
from copy import deepcopy
import math


GROUPS = ("observation_noise", "encoder_bias", "imu_alignment", "imu_wobble",
          "actuator_gains", "rigid_body", "ground_friction")
REQUESTED_GROUPS = ("observation_noise", "encoder_bias", "imu_alignment", "motor_strength",
                    "mass_com", "independent_inertia", "ground_friction", "timing")


def unit(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0. <= value <= 1.:
        raise ValueError(f"{name} must be finite within [0,1]")
    return float(value)


def resolve(spec, profile=None, strength=1.):
    """Return final task/mechanism values; no second severity can overwrite them."""
    if spec.get("schema") == "wheel-leg-randomization-study-v2":
        return _resolve_requested(spec, profile, strength)
    return _resolve_v1(spec, profile, strength)


def _resolve_v1(spec, profile, strength):
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


def _resolve_requested(spec, profile, strength):
    """User-requested final ranges, independent of the historical gap severity."""
    if tuple(spec.get("groups", ())) != REQUESTED_GROUPS:
        raise ValueError("Unsupported requested-domain groups or order")
    if (spec.get("fixed") != dict(leg_kp_scale=1, leg_kd_scale=1, spring_strength_scale=1, imu_wobble=False)
            or type(spec["fixed"]["imu_wobble"]) is not bool):
        raise ValueError("Requested P/D/spring must be one and IMU wobble must be disabled")
    strength = unit(strength, "strength")
    nominal = unit(spec["nominal_fraction"], "nominal_fraction")
    profile = spec["default_profile"] if profile is None else profile
    values = spec["profiles"].get(profile)
    if not isinstance(values, dict) or set(values) != set(REQUESTED_GROUPS):
        raise ValueError("One explicit scale is required per requested domain group")
    scales = {name: unit(values[name], name)*strength for name in REQUESTED_GROUPS}
    reference = spec["reference"]
    radii = ("motor_strength_fraction", "mass_fraction", "independent_inertia_fraction",
             "encoder_bias_rad", "encoder_noise_rad", "gyro_bias_rad_s", "gyro_noise_rad_s",
             "gravity_noise", "leg_velocity_noise_rad_s", "wheel_velocity_noise_rad_s")
    if set(reference) != set(radii) | {"base_com_offset_m", "imu_fixed_degrees", "ground_friction_range"}:
        raise ValueError("All requested-domain reference ranges must be explicit")
    for name in radii:
        unit(reference[name], name)
    if any(reference[name] >= 1. for name in radii[:3]):
        raise ValueError("Mass, inertia and motor strengths must stay strictly positive")
    com = reference["base_com_offset_m"]
    if set(com) != {"x", "y", "z"}:
        raise ValueError("Three explicit COM radii are required")
    for axis, radius in com.items():
        if not 0. <= unit(radius, "COM " + axis) <= .05:
            raise ValueError("COM radius exceeds the supported study range")
    imu = reference["imu_fixed_degrees"]
    if (not isinstance(imu, list) or len(imu) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0. <= v <= 10. for v in imu)):
        raise ValueError("Invalid fixed IMU alignment bounds")
    ground = reference["ground_friction_range"]
    if (not isinstance(ground, list) or len(ground) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in ground)
            or not 0. < ground[0] <= ground[1] <= 1.):
        raise ValueError("Invalid ground friction range")
    if values["timing"] not in (0., 1.):
        raise ValueError("Discrete timing is explicitly on or off, not an amplitude")
    # Reuse only v1's timing/history validation and neutral construction. None
    # of its gain/noise/dynamics radii or legacy gap overlays are applied.
    neutral = dict(schema="wheel-leg-randomization-study-v1", groups=list(GROUPS),
        nominal_fraction=nominal, default_profile="neutral",
        profiles={"neutral": dict.fromkeys(GROUPS, 0.)}, timing=deepcopy(spec["timing"]))
    result = _resolve_v1(neutral, "neutral", 1.)
    timing = {key: deepcopy(result["task"][key]) for key in spec["timing"]}
    if not scales["timing"]:
        timing = {key: [1, 1] if key == "pd_feedback_period_ms" else [0, 0] for key in timing}
    active = any(scales.values())
    noise, bias, align, motor, mass, inertia, friction, _ = (scales[k] for k in REQUESTED_GROUPS)
    def around(radius, scale):
        return [1.-radius*scale, 1.+radius*scale]
    task = result["task"]
    task.update(domains_enabled=active, motor_strength_range=around(reference["motor_strength_fraction"], motor),
        leg_kp_scale_range=[1., 1.], leg_kd_scale_range=[1., 1.], spring_strength_range=[1., 1.],
        encoder_bias_rad=reference["encoder_bias_rad"]*bias,
        pd_encoder_bias_rad=reference["encoder_bias_rad"]*bias,
        encoder_noise_rad=reference["encoder_noise_rad"]*noise,
        gyro_bias_rad_s=reference["gyro_bias_rad_s"]*noise,
        gyro_noise_rad_s=reference["gyro_noise_rad_s"]*noise,
        gravity_noise=reference["gravity_noise"]*noise,
        velocity_noise_rad_s=[reference["leg_velocity_noise_rad_s"]*noise,
                              reference["wheel_velocity_noise_rad_s"]*noise], **timing)
    # Keep identical zero-amplitude mount draws in every active v2 profile.
    # Otherwise removing alignment shifts the later PD bias/white-noise RNG
    # stream and confounds a supposed single-group counterfactual.
    task["imu_mount"] = (dict(fixed_degrees=[v*align for v in imu],
        wobble_degrees=[0., 0.], wobble_hz=[1., 1.]) if active else None)
    dynamics = result["dynamics_randomization"]
    dynamics.update(enabled=active,
        base_mass_scale=around(reference["mass_fraction"], mass),
        leg_mass_scale=around(reference["mass_fraction"], mass),
        wheel_mass_scale=around(reference["mass_fraction"], mass),
        inertia_scale=around(reference["independent_inertia_fraction"], inertia),
        base_com_offset_m={axis: [-radius*mass, radius*mass] for axis, radius in com.items()},
        source_reference="2026-10-26 requested final domains; independent inertia radius is final, not gap-scaled")
    ground = [.5+(v-.5)*friction for v in ground]
    result.update(schema="wheel-leg-resolved-randomization-study-v2", profile=profile, strength=strength,
        group_scales=scales, ground_friction_range=ground,
        effective_wheel_ground_friction_range=[(v+.1)/2. for v in ground],
        effective_unsaturated_leg_kp_scale_bounds=deepcopy(task["motor_strength_range"]),
        effective_unsaturated_leg_kd_scale_bounds=deepcopy(task["motor_strength_range"]),
        imu_wobble_single_axis_gyro_peak_bound_rad_s=0.,
        policy_PD_encoder_bias_disagreement_bound_rad=2.*task["encoder_bias_rad"],
        imu_wobble_enabled=False, leg_gain_randomization_enabled=False,
        spring_strength_randomization_enabled=False)
    return result


def apply(config, task, resolved):
    if task.get("gap_robustness_version") or task.get("randomization_study"):
        raise ValueError("Do not stack two domain profiles or apply a study twice")
    task.update(deepcopy(resolved["task"]))
    task.pop("domain_scale", None)
    task["version"] += "-domain-study-" + resolved["schema"].rsplit("-", 1)[-1]
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
    for key in ("base_mass_scale", "leg_mass_scale", "wheel_mass_scale", "inertia_scale"):
        for label, value in zip(("min", "max"), resolved["dynamics_randomization"][key], strict=True):
            values[f"{key}/{label}"] = value
    for axis, values_m in resolved["dynamics_randomization"]["base_com_offset_m"].items():
        for label, value in zip(("min", "max"), values_m, strict=True):
            values[f"base_com_offset_m/{axis}/{label}"] = value
    mount = task["imu_mount"]
    for axis, value in zip(("roll", "pitch", "yaw"), mount["fixed_degrees"] if mount else [0., 0., 0.], strict=True):
        values[f"imu_fixed_bound_deg/{axis}"] = value
    for label, value in zip(("min", "max"), resolved["ground_friction_range"], strict=True):
        values[f"ground_friction/{label}"] = value
    return values


def runtime_samples(env):
    """Store actual startup draws, not only requested endpoints or averages."""
    d = env.domain
    fields = dict(enabled=d.enabled, motor_strength=d.strength, leg_kp_scale=d.kp,
        leg_kd_scale=d.kd, spring_strength=d.spring, policy_sensor_delay_ticks=d.sensor_delay,
        policy_bias=d.bias, policy_noise_sigma=d.sigma, pd_encoder_bias=env.getup_feedback.bias)
    if d.imu_mount is not None:
        fields.update(d.imu_mount.parameters())
    rigid = env.dynamics_randomization
    if rigid is not None:
        fields.update(rigid_enabled=rigid.enabled, mass_scale=rigid.mass_scale,
            independent_inertia_scale=rigid.inertia_scale, base_com_offset_m=rigid.com_offset)
    return dict(schema="wheel-leg-domain-startup-samples-v1",
        samples={key: value.detach().cpu().tolist() for key, value in fields.items()},
        rigid_summary=rigid.summary() if rigid is not None else None,
        contact_material=env.contact_material.summary() if env.contact_material is not None else None,
        scope="actual sampled model arrays; does not replace native property or behavior verification")

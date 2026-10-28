"""Task binding for the independently measured new wheels, without old priors."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import torch

from .identified_wheel_response import WHEEL_NAMES, configure_identified_wheels


def configure(root, config, declaration, settings):
    """Authenticate the delivered model/control/evidence before deriving a task."""
    for key in ("model", "control", "evidence"):
        item = declaration[key]
        if hashlib.sha256((root/item["file"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Identified wheel input SHA differs: " + key)
    control = json.loads((root/declaration["control"]["file"]).read_text())
    result, matched = configure_identified_wheels(config, control, declaration["model"]["file"], root)
    if matched != control:
        raise ValueError("Delivered wheel control does not match the recorded controller")
    result["control_math_source"] = declaration["control"]["file"]
    result["control_math_sha256"] = declaration["control"]["sha256"]
    result["actuator_response_evidence"] = deepcopy(declaration["evidence"])
    validate_timing(settings)
    return result, control


def validate_timing(settings):
    # The effective measured I/O delay is the sole nominal low-level delay.
    # Policy observation age remains a separately declared uncertainty in v2.
    for key, expected in (("torque_delay_ms", [0, 0]), ("motor_tau_ms", [0., 0.]),
                          ("pd_feedback_delay_ms", [0, 0]), ("pd_feedback_period_ms", [1, 1])):
        if settings[key] != expected:
            raise ValueError("Aggregate wheel delay would overlap a low-level model: " + key)


def scaled_parameters(wheels, dynamics, count):
    """Reuse the wheel body's existing draw for the whole effective inertia."""
    scale = torch.ones(count, 2, device=wheels.device, dtype=wheels.dtype)
    if dynamics is not None:
        scale = (dynamics.mass_scale * dynamics.inertia_scale)[:, dynamics.wheel_ids].to(scale)
    added = scale * scale.new_tensor([row["additional_armature_kg_m2"] for row in wheels.rows])
    total = scale * scale.new_tensor([row["effective_total_inertia_kg_m2"] for row in wheels.rows])
    return scale, added, total


def install(env):
    """Keep the calibrated inertia dominant under the declared body-domain draw."""
    import warp as wp

    wheels = env.identified_wheels
    if not wheels.enabled:
        raise ValueError("Prone-motion training requires the measured new-wheel component")
    validate_timing(env.getup_cfg)
    ids = [env.robot.joint_names.index(name) for name in WHEEL_NAMES]
    dynamics = env.dynamics_randomization
    if dynamics is not None and [env.robot.body_names[i] for i in dynamics.wheel_ids] != ["L_link3", "R_link3"]:
        raise ValueError("Wheel inertia domain has a different left/right body order")
    scale, added, total = scaled_parameters(wheels, dynamics, env.num_envs)
    body_ids = [env.robot.body_names.index(name) for name in ("L_link3", "R_link3")]
    inertia = wp.to_torch(env.robot.root_view.get_inertias())[:, body_ids].to(added).reshape(env.num_envs, 2, 3, 3)
    # PhysX may relabel principal axes, but their eigenvalues are invariant.
    # These axisymmetric wheels have the axle moment as their largest moment.
    cad = torch.linalg.eigvalsh(inertia).amax(-1)
    expected_cad = scale * scale.new_tensor([row["cad_inertia_kg_m2"] for row in wheels.rows])
    if not torch.allclose(cad, expected_cad, rtol=1e-5, atol=1e-8):
        raise RuntimeError("Native wheel CAD inertia differs from the identified subtraction")
    env.robot.write_joint_armature_to_sim_index(armature=added, joint_ids=ids)
    actual = wp.to_torch(env.robot.root_view.get_dof_armatures())[:, ids].to(added)
    if not torch.allclose(actual, added, rtol=1e-6, atol=1e-8):
        raise RuntimeError("Measured wheel armature domain readback differs")
    wheels.inertia = total
    env.wheel_inertia_scale = scale
    env.startup_report["training_identified_wheels"] = dict(
        enabled=True, model_sha256=wheels.cfg["model_sha256"], native_properties_verified=True,
        joint_names=list(WHEEL_NAMES), physical_call_hz=round(1./env.dt), controller_velocity_p=.5,
        baseline_low_level_delay_ms=[5.5, 3.5], old_wheel_dynamics_removed=True,
        body_and_armature_share_one_existing_draw=True, extra_random_draws=0,
        added_armature_kg_m2=actual.cpu().tolist(), effective_inertia_kg_m2=total.cpu().tolist(),
        native_cad_axial_inertia_kg_m2=cad.cpu().tolist(),
        inertia_scale=scale.cpu().tolist(), independent_inertia_range_unchanged=True,
        policy_observation_age="Existing v2 0..1 policy ticks, additional to calibrated base I/O approximation",
        loaded_torque_speed_curve="Explicit delivered prior; not identified under load")

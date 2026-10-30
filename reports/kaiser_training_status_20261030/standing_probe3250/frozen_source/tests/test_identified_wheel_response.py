"""Physical invariants and timing of the new gearbox's actual RL component."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.identified_wheel_response import (
    IdentifiedWheelResponse, WHEEL_NAMES, configure_identified_wheels, load_wheel_response, validate_wheel_model,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model/wheel_identified_1562_20261028/model.json"


def settings(delay=False):
    return {"enabled": True, "model_path": str(MODEL),
            "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(), "apply_aggregate_delay": delay}


def response(delay=False):
    return IdentifiedWheelResponse(2, "cpu", .001, settings(delay), root=ROOT, dtype=torch.float64)


def test_new_hardware_units_and_independent_inertia_are_authenticated():
    _, model = load_wheel_response(settings(), ROOT)
    assert model["gear_ratio"] == 15.62
    assert model["bus_voltage_v"] == 24.
    assert all(s["new_gearbox_confirmed"] for s in model["source_files"])
    assert model["axes"][WHEEL_NAMES[0]]["moving_friction_nm"] != model["axes"][WHEEL_NAMES[1]]["moving_friction_nm"]
    for joint in WHEEL_NAMES:
        bad = copy.deepcopy(model)
        bad["axes"][joint]["additional_armature_kg_m2"] = bad["axes"][joint]["effective_total_inertia_kg_m2"]
        with pytest.raises(ValueError, match="existing wheel CAD"):
            validate_wheel_model(bad)
    bad_settings = settings()
    bad_settings["model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        load_wheel_response(bad_settings, ROOT)


def test_passive_losses_dissipate_and_are_not_counted_as_motor_current():
    model = response()
    torque = torch.zeros(2, 6, dtype=torch.float64)
    for speed in (.001, .1, 1., 6., 20., 55., 100.):
        velocity = torch.zeros_like(torque)
        velocity[:, (2, 5)] = torch.tensor([[speed, -speed], [-speed, speed]], dtype=torch.float64)
        motor = model.apply_torque(torque, velocity)
        assert torch.equal(motor, torque)
        assert torch.all(model.passive_effort * velocity[:, (2, 5)] <= 0)
        assert torch.all(model.passive_effort.abs() <= model.inertia * speed / .001 + 1e-12)
    model.apply_torque(torque, torch.zeros_like(torque))
    assert torch.count_nonzero(model.passive_effort) == 0


def test_native_floor_plus_passive_residual_equals_identified_curve_once():
    model = response()
    velocity = torch.zeros(2, 6, dtype=torch.float64)
    velocity[0, (2, 5)], velocity[1, (2, 5)] = 6., -6.
    model.apply_torque(torch.zeros_like(velocity), velocity)
    native = model.native_parameters()
    for k, joint in enumerate(WHEEL_NAMES):
        row = model.model["axes"][joint]
        point = row["friction_speed_knots_rad_s"].index(6.)
        for env, sign in ((0, 1.), (1, -1.)):
            observed = native["dynamic_friction"][joint] + abs(model.passive_effort[env, k].item())
            assert observed == pytest.approx(row["moving_friction_nm"][point] + sign * row["direction_bias_nm"], abs=1e-12)


def test_fractional_frame_to_rpm_delay_and_selected_reset():
    model = response(True)
    command = torch.zeros(2, 6, dtype=torch.float64)
    command[:, (2, 5)] = 1.
    velocity = torch.zeros_like(command)
    observed = torch.stack([model.apply_torque(command, velocity)[:, (2, 5)] for _ in range(10)])
    for k, joint in enumerate(WHEEL_NAMES):
        delay = model.model["axes"][joint]["aggregate_delay_s"] / .001
        whole, fraction = int(delay), delay - int(delay)
        assert torch.count_nonzero(observed[:whole, :, k]) == 0
        assert observed[whole, 0, k].item() == pytest.approx(1. - fraction)
        assert observed[whole + 1, 0, k].item() == 1.
    model.reset([0])
    updated = model.apply_torque(command, velocity)
    assert torch.count_nonzero(updated[0, (2, 5)]) == 0
    assert torch.equal(updated[1, (2, 5)], command[1, (2, 5)])


def test_motoring_limit_does_not_remove_high_speed_braking():
    model = response()
    velocity = torch.zeros(2, 6, dtype=torch.float64)
    velocity[0, (2, 5)], velocity[1, (2, 5)] = 80., -80.
    command = torch.zeros_like(velocity)
    command[:, (2, 5)] = velocity[:, (2, 5)].sign() * 4.5
    motoring = model.apply_torque(command, velocity)
    assert torch.count_nonzero(motoring[:, (2, 5)]) == 0
    braking = model.apply_torque(-command, velocity)
    assert torch.equal(braking[:, (2, 5)], -command[:, (2, 5)])
    assert torch.equal(braking[:, (0, 1, 3, 4)], command[:, (0, 1, 3, 4)])


def test_disabled_component_is_exact_identity_and_has_no_delay_state():
    model = IdentifiedWheelResponse(2, "cpu", .001)
    effort = torch.ones(2, 6)
    assert model.apply_torque(effort, torch.zeros_like(effort)) is effort
    assert not hasattr(model, "history")
    model.reset([0])


def test_installation_preserves_leg_parameters_and_removes_old_wheel_dynamics():
    mechanism = json.loads((ROOT / "contracts/scratch_standing_base_mechanism_v2.json").read_text())
    control = json.loads((ROOT / "contracts/flat_base_200hz_control_v2.json").read_text())
    for key in ("armature", "frictionloss", "damping", "time_constant_s"):
        mechanism["actuator_response"][key][0] = .0123
    original, original_control = copy.deepcopy(mechanism), copy.deepcopy(control)
    configured, prior = configure_identified_wheels(mechanism, control, MODEL, ROOT)
    assert mechanism == original and control == original_control
    assert prior["actuators"]["wheel"]["kd"] == .5
    from wheeled_tasks.chassis.control_contract import validate_chassis_control
    validate_chassis_control(prior)
    for key in ("armature", "frictionloss", "damping", "time_constant_s"):
        assert configured["actuator_response"][key][0] == .0123
        assert configured["actuator_response"][key][2] == 0
        assert configured["actuator_response"][key][5] == 0


def test_frozen_candidate_authenticates_actual_new_native_model_and_rejects_extra_delay():
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight, validate_response_evidence

    folder = MODEL.parent
    mechanism = json.loads((folder / "mechanism.json").read_text())
    control = load_chassis_control(folder / "control.json")
    assert control["timing"]["policy_dt"] == .005
    assert control["timing"]["physics_dt"] == .001
    receipt = design_preflight(mechanism, control, ROOT, stage="speed", role="train", seed=191019, num_envs=4096)
    assert receipt["status"] == "static_passed"
    changed = copy.deepcopy(mechanism)
    changed["identified_wheel_response"]["apply_aggregate_delay"] = False
    with pytest.raises(ValueError, match="frozen research evidence"):
        validate_response_evidence(changed, ROOT)
    changed = copy.deepcopy(control)
    changed["actuators"]["wheel"]["kd"] = .6
    from wheeled_tasks.chassis.control_contract import validate_chassis_control
    with pytest.raises(ValueError, match="kd differs"):
        validate_chassis_control(changed)


def test_production_native_readback_handles_independent_static_and_dynamic_wheel_losses():
    from test_response_preflight import native_readback_fixture

    code, scope, armature, friction, ids = native_readback_fixture()
    wheels = IdentifiedWheelResponse(3, "cpu", .001, settings(), root=ROOT)
    scope["self"].identified_wheels = wheels
    scope["WHEEL_NAMES"] = WHEEL_NAMES
    fields = wheels.native_parameters()
    for key in ("armature", "frictionloss", "damping", "time_constant_s"):
        for index in (2, 5):
            scope["self"].actuator_response.cfg[key][index] = 0.
    for joint, index in zip(WHEEL_NAMES, (2, 5)):
        armature[:, ids[index]] = fields["armature"][joint]
        for channel, field in enumerate(("frictionloss", "dynamic_friction", "damping")):
            friction[:, ids[index], channel] = fields[field][joint]
    exec(code, scope)
    assert scope["self"].startup_report["actuator_response"]["native_properties_verified"]
    friction[2, ids[5], 1] += .01
    with pytest.raises(RuntimeError, match="readback differs"):
        exec(code, scope)

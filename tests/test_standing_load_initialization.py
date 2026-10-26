"""Protect fresh-bias units, motor order and native evidence provenance."""
import hashlib
import json

import pytest
import torch

from wheeled_algo.standing_load_initialization import (
    CONTROL_ORDER, initial_action, load_prior, validate_values,
)


@pytest.fixture
def prior_files(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    control = dict(leg_kp=160., leg_position_scale=.25, action_clip=3.)
    mechanism = dict(v5_control=control, asset_directory="asset", asset_manifest_sha256="asset-sha",
                     control_math_sha256="control-sha", contact_material={}, physics_dt=.001, policy_dt=.005)
    bank = save("source/pose_bank.json", {})
    config = save("source/config/contract.json", mechanism)
    geometry = [.3, -.06, 0., -.3, .06, 0.]
    offset = [.025, -.025, 0., -.025, .025, 0.]
    reference = dict(schema="scratch-standing-nominal-static-load-diagnostic-v1", training_modified=False,
        control_joint_order=CONTROL_ORDER, height_m=.305, initial_geometry_action_control=geometry,
        hypothetical_static_load_offset_rad=offset,
        hypothetical_biased_action_control=[a+b/.25 for a, b in zip(geometry, offset)],
        static_load=dict(static_hold_torque_nm=[v*160. for v in offset]),
        sources_sha256={str(p.relative_to(tmp_path)): sha(p) for p in (bank, config)})
    reference_path = save("reference.json", reference)
    evidence = [save(f"native{i}/result.json", dict(completed_updates=0,
        diagnostic_reference_sha256=sha(reference_path), diagnostic_target=mode))
        for i, mode in enumerate(("constant_geometry", "constant_geometry_plus_offline_static_load_over_Kp"))]
    pair = save("pair.json", dict(schema="scratch-static-load-native-pair-v1", all_checks_pass=True,
        checks=dict(same_mechanics=True), new_PPO=0,
        diagnostic_result_sha256={str(p.relative_to(tmp_path)): sha(p) for p in evidence}))
    return reference_path, pair, mechanism, bank, tmp_path, reference, evidence


def load(files):
    ref, pair, mechanism, bank, root, *_ = files
    return load_prior(ref, pair, mechanism=mechanism, bank_path=bank, root=root)


def test_static_torque_becomes_trainable_action_bias_in_policy_order(prior_files):
    prior = load(prior_files)
    nominal = torch.tensor([.3, -.06, -.3, .06, 0., 0.])
    actual = initial_action(nominal, prior)
    torch.testing.assert_close(actual, torch.tensor([.4, -.16, -.4, .16, 0., 0.]))
    assert torch.equal(nominal, torch.tensor([.3, -.06, -.3, .06, 0., 0.]))
    assert prior["runtime_compensation"] is False and prior["all_outputs_trainable"] is True


def test_changed_source_or_native_evidence_is_rejected(prior_files):
    prior_files[-1][0].write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        load(prior_files)
    prior_files[3].write_text("{\"different\":true}")
    with pytest.raises(ValueError, match="source changed"):
        load(prior_files)


def test_changed_pd_gain_cannot_reuse_old_offsets(prior_files):
    prior_files[2]["v5_control"]["leg_kp"] = 161.
    with pytest.raises(ValueError, match="divided by Kp"):
        load(prior_files)


def test_wheel_propulsion_cannot_be_hidden_in_leg_prior(prior_files):
    reference = prior_files[5]
    reference["hypothetical_static_load_offset_rad"][2] = .001
    reference["static_load"]["static_hold_torque_nm"][2] = .16
    reference["hypothetical_biased_action_control"][2] = .004
    with pytest.raises(ValueError, match="wheel propulsion"):
        validate_values(reference, prior_files[2]["v5_control"])


@pytest.mark.parametrize("fault", ("nonfinite", "order"))
def test_invalid_torque_or_joint_order_is_rejected(prior_files, fault):
    reference = prior_files[5]
    if fault == "nonfinite":
        reference["static_load"]["static_hold_torque_nm"][1] = float("nan")
    else:
        reference["control_joint_order"] = list(reversed(CONTROL_ORDER))
    with pytest.raises(ValueError):
        validate_values(reference, prior_files[2]["v5_control"])


def test_different_reset_geometry_cannot_use_the_validated_prior(prior_files):
    prior = load(prior_files)
    with pytest.raises(ValueError, match="geometry differs"):
        initial_action(torch.zeros(6), prior)

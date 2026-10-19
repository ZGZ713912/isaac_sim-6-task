"""Prevent intermediate, stale or wrong-clock trials becoming release evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_flat_base import validate_release


def evidence(tmp_path):
    plan = json.loads((Path(__file__).resolve().parents[1] / "contracts/flat_base_finetune_v4.json").read_text())
    names = plan["pilot_case_names"]
    extra = ["forward_1", "backward_1", "forward_2", "backward_2", "forward_3", "backward_3",
             "rotate_4", "rotate_4_reverse", "curve_low", "curve_low_reverse", "height_scan",
             "start_stop_1", "start_stop_2", "start_stop_3"]
    catalogue = json.loads((Path(__file__).resolve().parents[1] / plan["source_worker_file"]).read_text())["evaluation"]["cases"]
    commands = {c["name"]:c["command"] for c in catalogue}
    def cohort(hz, seed, extended=False):
        case = dict(command=[0., 0., .305], episodes=4, requested_episodes=4, full_horizon_episodes=4,
            passed=True, checks=dict(survival=True, mechanics=True, velocity=True), vx_mae_m_s=.01,
            yaw_mae_rad_s=.01, height_mae_m=.001, closure_gap_max_m=.0001, tilt_max_deg=2., stand_drift_max_m=.01)
        cases = {n:{**deepcopy(case), "command":deepcopy(commands[n])} for n in names + (extra if extended else [])}
        return dict(status="evaluated", checkpoint_sha256="candidate", source_onnx_sha256=plan["source_onnx_sha256"],
            asset_manifest_sha256=plan["asset_manifest_sha256"], model_spec_sha256=plan["model_spec_sha256"],
            policy_hz=50, physics_hz=hz, seed=seed, new_optimizer_updates=0,
            settings=dict(protocol_id="flat-base-paired107-envelope-v2" if extended else "flat-base-paired107-fixed-v1",
                          cases=[dict(name=n, command=deepcopy(c["command"])) for n,c in cases.items()]),
            result=dict(passed=True, cases=cases))
    primary = cohort(200, plan["evaluation_seed"])
    reference = dict(schema="flat-rmcs-command-reference-v1", planar_slew="none; next policy tick",
                     height_command_is_reference=True, source_sha256={"fixture": "bound-by-export-entry"})
    default = dict(schema="flat-rmcs-packaged-default-height-v1", height_m=.23, source_sha256="fixture",
                   source_file="fixture.yaml", command_height_bounds_m=[.23, .42])
    maximum = {**deepcopy(default), "schema": "flat-rmcs-packaged-maximum-height-v1", "height_m": .42}
    manifest = dict(schema="flat-base-behavior-release-v4", checkpoint_sha256="candidate", evaluations={},
                    command_reference=reference, default_height_contract=default, maximum_height_contract=maximum)
    for name, hz, seed, extended in (
            ("training_envelope", 200, plan["evaluation_seed"], True),
            ("deployment_basic", 1000, plan["evaluation_seed"], False),
            ("deployment_envelope", 1000, plan["evaluation_seed"], True),
            ("confirmation_envelope", 1000, plan["confirmation_seed"], True),
            ("rmcs_reference_envelope", 1000, plan["evaluation_seed"], True),
            ("rmcs_reference_confirmation", 1000, plan["confirmation_seed"], True),
            ("rmcs_default_height_envelope", 1000, plan["evaluation_seed"], True),
            ("rmcs_default_height_confirmation", 1000, plan["confirmation_seed"], True),
            ("rmcs_maximum_height_envelope", 1000, plan["evaluation_seed"], True),
            ("rmcs_maximum_height_confirmation", 1000, plan["confirmation_seed"], True)):
        path = tmp_path / (name + ".json")
        data = cohort(hz, seed, extended)
        if name.startswith("rmcs_"):
            data["settings"]["protocol_id"] = "flat-base-paired107-rmcs-reference-envelope-v1"
            data["rmcs_command_reference"] = deepcopy(reference)
        if name.startswith("rmcs_default_height_"):
            data["settings"]["protocol_id"] = "flat-base-paired107-rmcs-default-height-envelope-v1"
            data["packaged_default_height"] = deepcopy(default)
            for case in [*data["result"]["cases"].values(), *data["settings"]["cases"]]:
                if any(abs(v) > 0 for v in case["command"][:2]): case["command"][2] = .23
        if name.startswith("rmcs_maximum_height_"):
            data["settings"]["protocol_id"] = "flat-base-paired107-rmcs-maximum-height-envelope-v1"
            data["packaged_maximum_height"] = deepcopy(maximum)
            for case in [*data["result"]["cases"].values(), *data["settings"]["cases"]]:
                if any(abs(v) > 0 for v in case["command"][:2]): case["command"][2] = .42
        path.write_text(json.dumps(data))
        manifest["evaluations"][name] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return primary, manifest, plan


def mutate(tmp_path, manifest, cohort, change):
    entry = manifest["evaluations"][cohort]
    path = tmp_path / entry["file"]
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_exact_clock_and_heldout_cohorts_are_required(tmp_path):
    primary, manifest, plan = evidence(tmp_path)
    assert len(validate_release(primary, manifest, plan, "candidate", tmp_path)) == 10
    manifest["checkpoint_sha256"] = "another-checkpoint"
    with pytest.raises(ValueError, match="exact checkpoint"):
        validate_release(primary, manifest, plan, "candidate", tmp_path)


@pytest.mark.parametrize("cohort,field,value", [
    ("deployment_envelope", "physics_hz", 200),
    ("confirmation_envelope", "seed", 191029),
    ("deployment_basic", "checkpoint_sha256", "stale-checkpoint"),
])
def test_wrong_clock_reused_seed_and_stale_model_are_rejected(tmp_path, cohort, field, value):
    primary, manifest, plan = evidence(tmp_path)
    mutate(tmp_path, manifest, cohort, lambda data: data.update({field: value}))
    with pytest.raises(ValueError, match="different model, clock or seed"):
        validate_release(primary, manifest, plan, "candidate", tmp_path)


@pytest.mark.parametrize("field,value", [("full_horizon_episodes", 3), ("vx_mae_m_s", float("nan")),
                                       ("stand_drift_max_m", .1)])
def test_aggregate_pass_does_not_hide_censoring_nonfinite_or_inaccurate_results(tmp_path, field, value):
    primary, manifest, plan = evidence(tmp_path)
    mutate(tmp_path, manifest, "deployment_envelope", lambda data:
        data["result"]["cases"]["stand_305mm"].update({field: value}))
    with pytest.raises(ValueError, match="unqualified or censored"):
        validate_release(primary, manifest, plan, "candidate", tmp_path)


def test_pd_clock_alone_does_not_prove_the_rmcs_command_reference_path(tmp_path):
    primary, manifest, plan = evidence(tmp_path)
    mutate(tmp_path, manifest, "rmcs_reference_envelope", lambda data: data.pop("rmcs_command_reference"))
    with pytest.raises(ValueError, match="RMCS command-reference path"):
        validate_release(primary, manifest, plan, "candidate", tmp_path)


def test_short_branch_diagnostic_cannot_qualify_release(tmp_path):
    primary, manifest, plan = evidence(tmp_path)
    mutate(tmp_path, manifest, "deployment_envelope", lambda data:
        data["settings"].update(protocol_id="flat-base-paired107-diagnostic-ablation-v1", diagnostic_only=True))
    with pytest.raises(ValueError, match="incomplete behavior coverage"):
        validate_release(primary, manifest, plan, "candidate", tmp_path)


def test_nominal_height_result_cannot_stand_in_for_packaged_default_height(tmp_path):
    primary, manifest, plan = evidence(tmp_path)
    mutate(tmp_path, manifest, "rmcs_default_height_envelope", lambda data:
        data["result"]["cases"]["forward_05"].update(command=[.5, 0., .305]))
    with pytest.raises(ValueError, match="different command or body height"):
        validate_release(primary, manifest, plan, "candidate", tmp_path)


def test_missing_default_height_heldout_evidence_is_rejected(tmp_path):
    primary, manifest, plan = evidence(tmp_path)
    del manifest["evaluations"]["rmcs_default_height_confirmation"]
    with pytest.raises(ValueError, match="missing a required cohort"):
        validate_release(primary, manifest, plan, "candidate", tmp_path)


def test_case_names_cannot_hide_replaced_motion_commands(tmp_path):
    primary, manifest, plan = evidence(tmp_path)
    mutate(tmp_path, manifest, "training_envelope", lambda data:
        data["result"]["cases"]["forward_3"].update(command=[0., 0., .305]))
    with pytest.raises(ValueError, match="different command or body height"):
        validate_release(primary, manifest, plan, "candidate", tmp_path)


def test_missing_maximum_height_heldout_evidence_is_rejected(tmp_path):
    primary, manifest, plan = evidence(tmp_path)
    del manifest["evaluations"]["rmcs_maximum_height_confirmation"]
    with pytest.raises(ValueError, match="missing a required cohort"):
        validate_release(primary, manifest, plan, "candidate", tmp_path)


def test_nominal_motion_cannot_satisfy_maximum_height_coverage(tmp_path):
    primary, manifest, plan = evidence(tmp_path)
    mutate(tmp_path, manifest, "rmcs_maximum_height_envelope", lambda data:
        data["result"]["cases"]["forward_3"].update(command=[3., 0., .305]))
    with pytest.raises(ValueError, match="different command or body height"):
        validate_release(primary, manifest, plan, "candidate", tmp_path)


def test_maximum_height_must_match_the_authenticated_package_boundary(tmp_path):
    primary, manifest, plan = evidence(tmp_path)
    manifest["maximum_height_contract"]["height_m"] = .305
    with pytest.raises(ValueError, match="authenticated packaged maximum height"):
        validate_release(primary, manifest, plan, "candidate", tmp_path)

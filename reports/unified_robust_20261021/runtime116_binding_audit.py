"""Read-only audit of runtime085 evidence; not permission to export a policy."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from iteration045_runner import write
from wheeled_algo.standing_reference_lifecycle import validate_observation_contract


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_configuration(configuration, contract):
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    bridge = configuration["rl_bridge"]["ros__parameters"]
    controller = configuration["wheel_leg_controller"]["ros__parameters"]
    executor = configuration["rmcs_executor"]["ros__parameters"]
    expected_bridge = dict(policy_rate=200., history_length=1, rl_obs_size=35, rl_feature_size=35, rl_action_size=6,
        expected_observation_contract_sha256=contract_sha, model_type="wheel_leg_v7", leg_winding_version="paired_branch107_v1",
        expected_model_spec_sha256=contract["model_spec_sha256"], enable_default=False)
    if any(bridge.get(k) != v for k, v in expected_bridge.items()):
        raise ValueError("Runtime rate, shape, geometry or observation contract differs")
    if (controller.get("unified_full_rl") is not True or controller.get("recovery_enabled") is not False
            or controller.get("pd_frequency") != 1000. or executor.get("update_rate") != 1000.
            or controller.get("leg_winding_version") != "paired_branch107_v1"):
        raise ValueError("Runtime direct six-axis RL or host feedback contract differs")
    if bridge.get("observation_terms") != ["index=0 path=/wheel_leg/rl/input/frame take=array type=double_array dim=35 name=frame"]:
        raise ValueError("Runtime must route the complete coherent observation frame")
    names = ["left_hip_joint", "left_knee_joint", "right_hip_joint", "right_knee_joint", "left_wheel", "right_wheel"]
    actions = [f"index={i} output=/wheel_leg/rl/target/{name} scale={'.25' if i < 4 else '10.0'} clip={'-4:4' if i < 4 else '-90:90'}"
               for i, name in enumerate(names)]
    # Compare tokens numerically to accept YAML's equivalent0.25 spelling.
    for actual, expected in zip(bridge.get("action_terms", []), actions, strict=True):
        a, b = [dict(token.split("=", 1) for token in row.split()) for row in (actual, expected)]
        if a.keys() != b.keys() or any(a[k] != b[k] for k in ("index", "output", "clip")) or float(a["scale"]) != float(b["scale"]):
            raise ValueError("Runtime action order, scale or limit differs")
    return contract_sha


def main():
    torch.set_num_threads(1)
    checkpoint = ROOT / "runs/unified_robust_20261022/composed_getup115/model_migrated.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    task = state["infos"]["settings"]
    validate_observation_contract(task)
    evidence_path = REPORTS / "runtime085_final.json"
    evidence = json.loads(evidence_path.read_text())
    runtime = Path(evidence["rmcs_worktree"])
    assert subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip() == evidence["rmcs_commit"]
    child = runtime / "rmcs_ws/src/rmcs_rl"
    assert subprocess.check_output(["git", "-C", str(child), "rev-parse", "HEAD"], text=True).strip() == evidence["child_commit"]
    assert not subprocess.check_output(["git", "-C", str(runtime), "status", "--porcelain", "-uno"], text=True).strip()
    for relative, expected in evidence["source_files"].items():
        assert sha(runtime / relative) == expected, relative
    test_count = 0
    for relative, expected in evidence["gtest_xml_sha256"].items():
        path = runtime / relative
        assert sha(path) == expected, relative
        root = ET.parse(path).getroot()
        assert root.get("failures", "0") == "0" and root.get("errors", "0") == "0"
        test_count += len(root.findall(".//testcase"))
    assert test_count == evidence["gtest_count"] == 38
    for relative, expected in evidence["files"].items():
        assert sha(REPORTS / relative) == expected, relative
    parity = evidence["frame_parity"]
    assert parity["frames"] == 35158 and parity["valid_frames"] == 35137 and parity["invalid_rejection_exact"]
    assert parity["full35_maximum_error"] < 1e-5 and parity["passive_history_maximum_error"] < 1e-5
    yaml_path = runtime / "rmcs_ws/src/rmcs_bringup/config/wheel-leg-infantry-unified-rl.yaml"
    config = yaml.safe_load(yaml_path.read_text())
    contract = task["standing_position_observation"]
    contract_sha = verify_configuration(config, contract)
    assert contract_sha == evidence["model_contract_sha256"]
    negatives = []
    for name, change in (
        ("wrong50Hz", lambda c: c["rl_bridge"]["ros__parameters"].update(policy_rate=50.)),
        ("missing_contract", lambda c: c["rl_bridge"]["ros__parameters"].update(expected_observation_contract_sha256="")),
        ("scripted_handoff", lambda c: c["wheel_leg_controller"]["ros__parameters"].update(unified_full_rl=False)),
        ("wrong_PD_rate", lambda c: c["wheel_leg_controller"]["ros__parameters"].update(pd_frequency=200.)),
        ("wrong_action_scale", lambda c: c["rl_bridge"]["ros__parameters"]["action_terms"].__setitem__(0, "index=0 output=/wheel_leg/rl/target/left_hip_joint scale=1.0 clip=-4:4")),
    ):
        bad = deepcopy(config)
        change(bad)
        try:
            verify_configuration(bad, contract)
        except ValueError:
            negatives.append(name)
        else:
            raise AssertionError("Accepted negative runtime contract " + name)
    plan = dict(schema="runtime116-live-binding-audit-v1", status="current085_runtime_matches115_observation_ABI",
        candidate_checkpoint_sha256=sha(checkpoint), observation_contract_sha256=contract_sha,
        rmcs_worktree=str(runtime), rmcs_commit=evidence["rmcs_commit"], child_commit=evidence["child_commit"],
        recorded085_sources_all_live_exact=True, recorded38_gtest_outputs_all_live_exact=True,
        frame_parity=parity, yaml=str(yaml_path), yaml_sha256=sha(yaml_path), configuration_negative_cases_rejected=negatives,
        reference_evidence_sha256=sha(evidence_path), native_runtime_or_hardware_started=False,
        behavior_qualified=False, export_implemented=False, active_delivery_changed=False, new_macro_updates=0,
        remaining_export_work=[
            "Wait for115 long standing, full random, two matched nominal and stationary lifecycle admission",
            "Replace unconditional experimentalABI rejection with exact runtime/contract binding validation",
            "Embed rmcs_observation_contract_sha256 in ONNX and source geometry/position contract in delivery receipt",
            "Describe slots14:16 as20x filtered standing displacement; do not reuse old zero-channel metadata",
            "Describe per-height5/5/10cm and recenter evidence, not old position_input_to_actor=false",
            "Verify exact candidate graph against Torch on all evaluated behaviors and frozen C++35D frame fixtures",
            "Keep legacy313 delivery untouched; user performs hardware tests of the reviewed separate package"],
        interpretation="Existing compiled interface evidence only.079 diagnostic graph parity does not qualify115 dynamics or115 graph.")
    path = REPORTS / "runtime116_binding_audit.json"
    assert not path.exists()
    write(path, plan)
    print(json.dumps(plan), flush=True)


if __name__ == "__main__":
    main()

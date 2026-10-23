"""Test the exact trial graph through production C++ with memory-only sensors."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_unified_test_export import verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="001", choices=("001", "002", "003"))
    args = parser.parse_args()
    package = ROOT / "models/unified_getup115_test_200hz_20261023"
    evidence = package / "evidence" / ("runtime119_" + args.revision)
    evidence.mkdir(exist_ok=False)
    if (evidence / "runtime119_validation.json").exists():
        raise ValueError("Preserve existing validation receipt")
    positive = verify(package)
    original = yaml.safe_load((package / "wheel-leg-infantry-unified115-test.yaml").read_text())
    negatives = []
    for name, change in (
        ("50Hz", lambda c: c["rl_bridge"]["ros__parameters"].update(policy_rate=50.)),
        ("missing_observation_hash", lambda c: c["rl_bridge"]["ros__parameters"].update(expected_observation_contract_sha256="")),
        ("wrong_history", lambda c: c["rl_bridge"]["ros__parameters"].update(history_length=2)),
        ("old_zero_slot_route", lambda c: c["rl_bridge"]["ros__parameters"].update(observation_terms=["index=0 path=/old/obs dim=35"])),
        ("wrong_action_scale", lambda c: c["rl_bridge"]["ros__parameters"]["action_terms"].__setitem__(0,
            "index=0 output=/wheel_leg/rl/target/left_hip_joint scale=1.0 clip=-4:4")),
        ("scripted_recovery", lambda c: c["wheel_leg_controller"]["ros__parameters"].update(recovery_enabled=True)),
        ("wrong_host_PD", lambda c: c["wheel_leg_controller"]["ros__parameters"].update(pd_frequency=200.)),
    ):
        with tempfile.TemporaryDirectory(prefix="unified119-negative-") as temporary:
            destination = Path(temporary)
            # Copy only the files consumed by the portable verifier.
            for filename in ("policy.onnx", "checkpoint.pt", "deployment_contract.json", "inference_vectors.json"):
                shutil.copy2(package / filename, destination / filename)
            bad = deepcopy(original)
            change(bad)
            (destination / "wheel-leg-infantry-unified115-test.yaml").write_text(yaml.safe_dump(bad))
            try:
                verify(destination)
            except ValueError as error:
                negatives.append(dict(name=name, rejected=True, reason=str(error)))
            else:
                raise ValueError("Negative configuration accepted: " + name)
    runtime = Path("/home/noir/Documents/workspace/RMCS/.codex/worktrees/unified-standing-v3")
    container_root = "/workspaces/RMCS/.codex/worktrees/unified-standing-v3"
    fixture_name = "unified_runtime119_" + args.revision
    fixture = runtime / "rmcs_ws/build/rmcs_core" / fixture_name
    fixture.mkdir(exist_ok=False)
    shutil.copy2(package / "policy.onnx", fixture / "unified115_test.onnx")
    model_inside = container_root + "/rmcs_ws/build/rmcs_core/" + fixture_name + "/unified115_test.onnx"
    # Keep ROS YAML scalar quoting: PyYAML's unquoted string nan is read as a
    # floating-point scalar by rcl_yaml_param_parser, unlike the source YAML.
    profile = (package / "wheel-leg-infantry-unified115-test.yaml").read_text()
    old_path = 'rl_model_path: "models/wheel_leg/unified115_test.onnx"'
    assert old_path in profile
    profile = profile.replace(old_path, 'rl_model_path: "' + model_inside + '"')
    (fixture / "memory_test.yaml").write_text(profile)
    profile_inside = container_root + "/rmcs_ws/build/rmcs_core/" + fixture_name + "/memory_test.yaml"
    tests = []
    for package_name, binary in (("rmcs_rl", "policy_model_contract"), ("rmcs_rl", "observation_frame"),
                                  ("rmcs_core", "wheel_leg_unified_observation"),
                                  ("rmcs_core", "wheel_leg_unified_policy_integration")):
        binary_inside = container_root + "/rmcs_ws/build/" + package_name + "/" + binary
        xml_inside = container_root + "/rmcs_ws/build/rmcs_core/" + fixture_name + "/" + binary + ".xml"
        shell = "source /opt/ros/jazzy/setup.bash && source " + shlex.quote(container_root + "/rmcs_ws/install/setup.bash")
        shell += " && exec " + shlex.quote(binary_inside) + " " + shlex.quote("--gtest_output=xml:" + xml_inside)
        command = ["docker", "exec", "-e", "RMCS_UNIFIED_TEST_PROFILE=" + profile_inside,
                   "rmcs_devcontainer-rmcs-develop-1", "bash", "-lc", shell]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (evidence / (binary + ".log")).write_text(result.stdout)
        if result.returncode:
            raise ValueError(f"C++ test failed: {binary}, exit {result.returncode}; see preserved log")
        xml_path = fixture / (binary + ".xml")
        shutil.copy2(xml_path, evidence / (binary + ".xml"))
        xml = ET.parse(xml_path).getroot()
        if any(int(xml.get(field, "0")) for field in ("failures", "errors", "disabled")):
            raise ValueError("C++ result has failures")
        if any(case.find("skipped") is not None for case in xml.iter("testcase")):
            raise ValueError("C++ test silently skipped")
        tests.append(dict(binary=binary, cases=int(xml.get("tests")), failures=0, skipped=0, command=command))
        print(json.dumps(tests[-1]), flush=True)
    result = dict(schema="unified119-trial-offline-validation-v1", model_sha256=sha(package / "policy.onnx"),
        portable_verifier=positive, negative_configurations=negatives, cpp_tests=tests,
        cpp_test_cases=sum(row["cases"] for row in tests), exact115_graph_loaded_by_production_bridge=True,
        production_controller_inputs="in-memory sensors only; no WheelLeg hardware component instantiated",
        memory_profile_sha256=sha(fixture / "memory_test.yaml"), compiled_runtime_rebuilt=False,
        behavior_qualified=False, hardware_tested=False, hardware_process_launched=False, new_ppo_updates=0,
        revision=args.revision)
    (evidence / "runtime119_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("model_sha256", "cpp_test_cases", "hardware_tested")}), flush=True)


if __name__ == "__main__":
    main()

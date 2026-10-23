#!/usr/bin/env python3
"""Verify a portable full-RL trial package offline; never opens vehicle devices."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    import numpy as np
    import onnxruntime as ort
    import yaml

    directory = Path(directory)
    contract = json.loads((directory / "deployment_contract.json").read_text())
    vectors = json.loads((directory / "inference_vectors.json").read_text())
    if (contract["schema"] != "unified-getup-user-trial-v1"
            or contract["behavior_qualified"] is not False or contract["hardware_tested"] is not False
            or contract["policy_hz"] != 200 or contract["physical_pd_hz"] != 1000
            or contract["history_length"] != 1 or contract["actor_dim"] != 35 or contract["action_dim"] != 6):
        raise ValueError("Trial status, dimensions or clocks differ")
    observation = contract["standing_position_observation"]
    observation_sha = hashlib.sha256(json.dumps(
        observation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (observation_sha != contract["observation_contract_sha256"]
            or observation_sha != "db0fc6121c73121dcd39eef0524ecfeab4b3aa69aa7f5700d4205499af2402a9"):
        raise ValueError("Standing observation contract differs")
    policy = directory / "policy.onnx"
    model_sha = digest(policy)
    if model_sha != contract["model_sha256"] or model_sha != vectors["model_sha256"]:
        raise ValueError("Model SHA differs")
    if digest(directory / "checkpoint.pt") != contract["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA differs")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    required = dict(policy_hz="200", observation_dim="35", action_dim="6",
        source_checkpoint_sha256=contract["checkpoint_sha256"],
        rmcs_observation_contract_sha256=observation_sha,
        leg_winding_version="paired_branch107_v1",
        leg_target_model_spec_sha256=observation["model_spec_sha256"],
        usage="USER_AUTHORIZED_EXPERIMENTAL_VEHICLE_TEST", behavior_qualified="false")
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("ONNX metadata differs")
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].name != "obs"
            or inputs[0].type != "tensor(float)" or inputs[0].shape != [1, 35]
            or outputs[0].name != "actions" or outputs[0].type != "tensor(float)"
            or outputs[0].shape != [1, 6]):
        raise ValueError("Expected float32 1x35 -> 1x6")
    configuration = yaml.safe_load((directory / "wheel-leg-infantry-unified115-test.yaml").read_text())
    bridge = configuration["rl_bridge"]["ros__parameters"]
    controller = configuration["wheel_leg_controller"]["ros__parameters"]
    hardware = configuration["wheel_leg"]["ros__parameters"]
    if (bridge["policy_rate"] != 200. or bridge["history_length"] != 1
            or [bridge[key] for key in ("rl_obs_size", "rl_feature_size", "rl_action_size")] != [35, 35, 6]
            or bridge["expected_observation_contract_sha256"] != observation_sha
            or bridge["rl_model_path"] != "models/wheel_leg/unified115_test.onnx"
            or bridge["expected_model_spec_sha256"] != observation["model_spec_sha256"]
            or bridge["leg_winding_version"] != "paired_branch107_v1"
            or bridge["enable_default"] is not False
            or controller["unified_full_rl"] is not True or controller["recovery_enabled"] is not False
            or controller["pd_frequency"] != 1000.
            or configuration["rmcs_executor"]["ros__parameters"]["update_rate"] != 1000.
            or hardware["joint_control_mode"] != "torque"):
        raise ValueError("Runtime configuration differs")
    if bridge["observation_terms"] != [
            "index=0 path=/wheel_leg/rl/input/frame take=array type=double_array dim=35 name=frame"]:
        raise ValueError("Runtime must consume the coherent frame")
    names = ["left_hip_joint", "left_knee_joint", "right_hip_joint", "right_knee_joint", "left_wheel", "right_wheel"]
    if len(bridge["action_terms"]) != 6:
        raise ValueError("Six ordered action terms required")
    for i, term in enumerate(bridge["action_terms"]):
        values = dict(token.split("=", 1) for token in term.split())
        if (values["index"] != str(i) or values["output"] != "/wheel_leg/rl/target/" + names[i]
                or float(values["scale"]) != (.25 if i < 4 else 10.)
                or values["clip"] != ("-4:4" if i < 4 else "-90:90")):
            raise ValueError("Action order, scaling or limits differ")
    if not vectors["vectors"]:
        raise ValueError("Missing frozen inference vectors")
    bounds = np.asarray([16.] * 4 + [9.] * 2)
    worst = 0.
    for row in vectors["vectors"]:
        obs = np.asarray(row["observation"], dtype=np.float32)[None]
        expected = np.asarray(row["expected_raw_action"], dtype=np.float32)[None]
        actual = session.run(None, {"obs": obs})[0]
        error = float(np.max(np.abs(actual - expected)))
        if (not np.isfinite(actual).all() or not np.isfinite(error)
                or error > vectors["max_abs_tolerance"] or (np.abs(actual) > bounds + 1e-5).any()):
            raise ValueError("Inference vector or action bound mismatch")
        worst = max(worst, error)
    return dict(status="offline_check_passed", model_sha256=model_sha,
                observation_contract_sha256=observation_sha, vectors=len(vectors["vectors"]),
                maximum_raw_action_error=worst, hardware_tested=False, behavior_qualified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))

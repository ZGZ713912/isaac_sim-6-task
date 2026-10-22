#!/usr/bin/env python3
"""Offline check of a packaged ONNX policy; never connects to the vehicle."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(directory):
    import numpy as np
    import onnxruntime as ort
    directory = Path(directory)
    contract = json.loads((directory / "deployment_contract.json").read_text())
    vectors = json.loads((directory / "inference_vectors.json").read_text())
    policy = directory / "policy.onnx"
    sha = hashlib.sha256(policy.read_bytes()).hexdigest()
    if sha != contract["model_sha256"] or sha != vectors["model_sha256"]:
        raise ValueError("Policy SHA does not match the contract and test vectors")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    if (contract["policy_hz"] != 200 or contract["history_length"] != 1
            or metadata.get("policy_hz") != "200"
            or metadata.get("source_checkpoint_sha256") != contract["checkpoint_sha256"]
            or metadata.get("leg_target_model_spec_sha256") != contract["leg_target_model_spec_sha256"]):
        raise ValueError("Policy metadata and runtime contract differ")
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].name != "obs"
            or inputs[0].shape != [1, 35] or inputs[0].type != "tensor(float)"
            or outputs[0].shape != [1, 6]):
        raise ValueError("Expected a single float32 1x35 -> 1x6 policy")
    bounds = np.asarray([contract["leg_action_clip"]] * 4 + [contract["wheel_action_clip"]] * 2)
    errors = []
    if not vectors["vectors"]:
        raise ValueError("No frozen inference vectors")
    for row in vectors["vectors"]:
        observation = np.asarray(row["observation"], dtype=np.float32)[None]
        expected = np.asarray(row["expected_raw_action"], dtype=np.float32)[None]
        actual = session.run(None, {"obs": observation})[0]
        error = float(np.max(np.abs(actual - expected)))
        if (not np.isfinite(actual).all() or not np.isfinite(error)
                or error > vectors["max_abs_tolerance"] or (np.abs(actual) > bounds + 1e-5).any()):
            raise ValueError("ONNX failed a frozen numeric/range check")
        errors.append(error)
    return dict(offline_numeric_check="passed", vectors=len(errors), max_abs_error=max(errors),
                model_sha256=sha, hardware_tested=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))

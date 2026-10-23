"""Unqualified ONNX for offline RMCS interface tests; never a delivery export."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from export_robust_getup import load_mean


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    parent = ROOT / "runs/unified_robust_20261022/pilot033_dynamic_recovery079/model_final.pt"
    source = ROOT / "runs/unified_robust_20261022/dynamic_recovery_full079/robust/trajectory.npz"
    output = ROOT / "runs/unified_robust_20261022/runtime085_diagnostic"
    output.mkdir(exist_ok=False)
    graph = output / "diagnostic_runtime_v3_079.onnx"
    mean, info = load_mean(parent)
    contract = info["settings"]["standing_position_observation"]
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert digest == "db0fc6121c73121dcd39eef0524ecfeab4b3aa69aa7f5700d4205499af2402a9"
    torch.onnx.export(mean, torch.zeros(1, 35), str(graph), input_names=["obs"], output_names=["actions"],
        opset_version=17, dynamo=False)
    model = onnx.load(graph)
    onnx.helper.set_model_props(model, dict(policy_hz="200", observation_dim="35", action_dim="6",
        source_checkpoint_sha256=sha(parent), leg_winding_version="paired_branch107_v1",
        leg_target_model_spec_sha256=contract["model_spec_sha256"],
        rmcs_observation_contract_sha256=digest,
        usage="UNQUALIFIED_OFFLINE_INTERFACE_TEST_ONLY"))
    onnx.save(model, graph)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
    with np.load(source) as data:
        flat = data["observations"].reshape(-1, 35)
        obs = flat[np.linspace(0, len(flat) - 1, 2048, dtype=int)].copy()
    maximum = 0.
    with torch.inference_mode():
        for row in obs:
            expected = mean(torch.from_numpy(row[None])).numpy()
            actual = session.run(None, {"obs": row[None]})[0]
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    assert maximum < 1e-5, maximum
    receipt = dict(schema="runtime085-diagnostic-graph-v1", checkpoint_sha256=sha(parent),
        observation_contract_sha256=digest, graph=str(graph.relative_to(ROOT)), graph_sha256=sha(graph),
        input_trajectory_sha256=sha(source), checked_inputs=len(obs), maximum_raw_action_error=maximum,
        delivery_export=False, behavior_qualified=False, hardware_ready=False, native_steps=0, new_ppo_updates=0)
    (Path(__file__).parent / "runtime085_graph.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

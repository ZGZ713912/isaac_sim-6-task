"""Audit trained adaptation scope and a run-only diagnostic graph, no delivery."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration036_compare import read
from wheeled_algo.robust_getup import quiet_gate


def main():
    torch.set_num_threads(1)
    checkpoint = RUNS / "pilot028_all_prediction053/model_final.pt"
    parent, _ = load_mean(RUNS / "quiet_all_prediction052/model_migrated.pt")
    after, info = load_mean(checkpoint)
    x = torch.from_numpy(np.concatenate([read(name)[3]["observations"][::20].reshape(-1, 35).copy()
        for name in ("side_confirmation032", "all_prediction_static052")]))
    with torch.inference_mode():
        frozen = quiet_gate(x).ravel() == 0
        assert torch.equal(parent(x)[frozen], after(x)[frozen])
    graph = checkpoint.parent / "diagnostic_adapted_prediction.onnx"
    torch.onnx.export(after, torch.zeros(1, 35), str(graph), input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = onnx.load(graph)
    onnx.helper.set_model_props(model, dict(policy_hz="200", observation_dim="35", action_dim="6",
        source_checkpoint_sha256=digest, quiet_leg_prediction_scope="all_quiet_heights",
        quiet_leg_prediction_seconds="0.005", qualification="diagnostic_only_not_admitted"))
    onnx.save(model, graph)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
    probes = x[::max(1, len(x)//3072)][:3072].clone()
    probes[::7, 3] = .333 * 5.
    with torch.inference_mode():
        expected = after(probes).numpy()
    actual = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in probes.numpy()])
    error = float(np.max(np.abs(actual - expected)))
    assert error < 2e-4
    result = dict(schema="prediction053-numerical-isolation-v1", checkpoint_sha256=digest,
        clock=info["completed_updates"], actual_observations=len(x), nonquiet_actions_bitwise_exact=int(frozen.sum()),
        learned_feedback_weights={k: v.flatten().tolist() for k, v in after.quiet_velocity.state_dict().items()},
        onnx_probes=len(probes), onnx_max_raw_error=error, diagnostic_graph=str(graph.relative_to(ROOT)),
        diagnostic_graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),
        optimizer_updates=0, native_runs=0, new_delivery=False, long_training_admitted=False)
    (REPORTS / "iteration053_numeric.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

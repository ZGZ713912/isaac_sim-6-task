"""Verify the nonzero learned path is representable in the35to6 ONNX contract.

The artificial head weights are confined to a copied diagnostic graph; the
training checkpoint and deployment package are never changed.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from export_robust_getup import load_mean


def main():
    torch.set_num_threads(1)
    checkpoint = RUNS / "quiet_rolling050/model_migrated.pt"
    model, _ = load_mean(checkpoint)
    model = deepcopy(model)
    with torch.no_grad():
        for name, factor in (("mid", 1.), ("low", .7), ("high", 1.2)):
            model.quiet_velocity[name].weight.copy_(torch.tensor([[.2], [-.1], [.1], [-.2]]) * factor)
    collected = []
    for run in ("quiet_velocity_baseline036", "side_confirmation032", "rolling_command_feedback049"):
        data = np.load(RUNS / run / "robust/trajectory.npz")
        collected.append(data["observations"][::80].reshape(-1, 35)[:1024].copy())
    x = np.concatenate(collected)
    x[::7, 3] = np.float32(.333) * np.float32(5.)
    path = checkpoint.parent / "diagnostic_learned_path.onnx"
    torch.onnx.export(model, torch.from_numpy(x[:1]), str(path), input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    graph = onnx.load(str(path))
    onnx.helper.set_model_props(graph, dict(scope="artificial_nonzero_heads_export_check_only_not_a_trained_motor_policy",
        policy_hz="200", diagnostic_only="true", source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(torch.from_numpy(x)).numpy()
    actual, latency = [], []
    for row in x:
        start = time.perf_counter()
        actual.append(session.run(None, {"obs": row[None]})[0][0])
        latency.append(time.perf_counter() - start)
    error = float(np.abs(np.asarray(actual) - expected).max())
    assert np.isfinite(actual).all() and error < 2e-4
    result = dict(schema="rolling050-nonzero-path-onnx-check-v1", real_and_intermediate_height_inputs=len(x),
        max_raw_action_error=error, input_shape=[1,35], output_shape=[1,6], policy_hz=200,
        cpu_single_thread_p95_ms=float(np.quantile(latency, .95) * 1000.),
        diagnostic_graph=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        scope="Artificial nonzero weights only in a deep copy; not training, behavior, hardware latency qualification, or deployment admission",
        optimizer_updates=0)
    (REPORTS / "iteration050_onnx_path_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

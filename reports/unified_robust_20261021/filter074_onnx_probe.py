"""Numerical ONNX support probe; standing ABI delivery guard stays closed."""
import hashlib
import json
from pathlib import Path
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    checkpoint = RUNS / "quiet_whole_wheel074/model_migrated.pt"
    mean, info = load_mean(checkpoint)
    arrays, sources = [], {}
    for name in ("quiet_whole_static074", "high_recenter_full066"):
        path = RUNS / name / "robust/trajectory.npz"
        with np.load(path) as data:
            rows = data["observations"].reshape(-1, 35)
            arrays.append(rows[np.linspace(0, len(rows) - 1, 1024, dtype=int)].copy())
        sources[str(path.relative_to(ROOT))] = sha(path)
    x = np.concatenate(arrays)
    x[::7, 3] = np.float32(.333) * np.float32(5.)
    path = checkpoint.parent / "diagnostic_filter074.onnx"
    if path.exists():
        raise ValueError("Refuse to overwrite ONNX evidence")
    torch.onnx.export(mean, torch.from_numpy(x[:1]), str(path), input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    graph = onnx.load(path)
    onnx.helper.set_model_props(graph, dict(diagnostic_only="true", scope="numerical_export_probe_no_deployment_contract",
        policy_hz="200", source_checkpoint_sha256=sha(checkpoint),
        required_observation_abi=info["settings"]["standing_position_observation"]["schema"],
        quiet_output_wheel_alpha="0.25"))
    onnx.checker.check_model(graph)
    onnx.save(graph, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = mean(torch.from_numpy(x)).numpy()
    actual, latency = [], []
    for row in x:
        start = time.perf_counter()
        actual.append(session.run(None, {"obs": row[None]})[0][0])
        latency.append((time.perf_counter() - start) * 1000.)
    error = float(np.abs(np.asarray(actual) - expected).max())
    assert np.isfinite(actual).all() and error < 5e-4
    result = dict(schema="filter074-numerical-onnx-probe-v1", checkpoint_sha256=sha(checkpoint),
        graph=str(path.relative_to(ROOT)), graph_sha256=sha(path), input_shape=[1, 35], output_shape=[1, 6],
        inputs=len(x), max_raw_action_error=error, cpu_single_thread_p95_ms=float(np.quantile(latency, .95)),
        sources=sources, weights_modified=False, new_macro_updates=0,
        full_training_admitted=False, deployment_export_guard_unchanged=True,
        scope="Offline numerical support only; experimental observation ABI not installed in RMCS")
    (REPORTS / "filter074_onnx_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

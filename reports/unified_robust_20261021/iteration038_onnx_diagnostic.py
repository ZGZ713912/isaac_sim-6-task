"""Check the wheel-only architecture's numeric export; no deployment admission."""
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from export_robust_getup import load_mean

torch.set_num_threads(1)
run = ROOT / "runs/unified_robust_20261022/pilot023_wheel_velocity038"
checkpoint = run / "model_final.pt"
mean, info = load_mean(checkpoint)
target = run / "diagnostic_wheel_velocity038.onnx"
assert not target.exists()
torch.onnx.export(mean, torch.zeros(1, 35), str(target), input_names=["obs"], output_names=["actions"],
                  opset_version=17, dynamo=False)
model = onnx.load(target)
onnx.helper.set_model_props(model, dict(purpose="numeric_diagnostic_only", hardware_ready="false",
    source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    quiet_velocity_feedback_mode="wheels", policy_hz="200"))
onnx.save(model, target)
onnx.checker.check_model(model)
options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
session = ort.InferenceSession(str(target), sess_options=options, providers=["CPUExecutionProvider"])
rows = []
for name in ("quiet_velocity_baseline036", "side_confirmation032"):
    observations = np.load(ROOT / f"runs/unified_robust_20261022/{name}/robust/trajectory.npz")["observations"].reshape(-1, 35)
    rows.append(observations[np.linspace(0, len(observations) - 1, 256, dtype=int)])
observations = np.concatenate(rows)
errors = {}
with torch.inference_mode():
    for height in (None, .23, .2675, .305, .3625, .42):
        values = observations.copy()
        if height is not None:
            values[:, 3] = np.float32(height) * np.float32(5.)
        worst = 0.
        for value in values:
            expected = mean(torch.from_numpy(value[None])).numpy()
            actual = session.run(None, {"obs": value[None]})[0]
            assert np.isfinite(actual).all()
            worst = max(worst, float(np.abs(expected - actual).max()))
        errors[str(height)] = worst
assert max(errors.values()) <= 5e-4
value = observations[:1]
for _ in range(20):
    session.run(None, {"obs": value})
timings = []
for _ in range(512):
    start = time.perf_counter()
    session.run(None, {"obs": value})
    timings.append((time.perf_counter() - start) * 1000.)
receipt = dict(schema="quiet038-onnx-diagnostic-v1", completed_updates=info["completed_updates"],
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    onnx_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    shape="1x35 ->1x6 deterministic mean", numeric_observations=512 * len(errors),
    maximum_raw_action_error_by_height=errors,
    cpu_inference_p50_ms=float(np.median(timings)), cpu_inference_p95_ms=float(np.percentile(timings, 95)),
    cpu_inference_max_ms=max(timings), local_desktop_timing_only=True,
    hardware_ready=False, deployment_package_replaced=False,
    scope="Architecture/numeric check only; original two-seed behavioral exporter gate remains mandatory")
(Path(__file__).parent / "iteration038_onnx_diagnostic.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))

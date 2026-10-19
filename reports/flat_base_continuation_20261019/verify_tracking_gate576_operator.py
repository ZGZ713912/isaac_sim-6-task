"""Actual parent tensors and current RMCS loader for a declared gate counterfactual."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.guarded_feedback_residual import TRACKING_KIND
from wheeled_algo.onnx_actor_import import recover_actor

torch.set_num_threads(4)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
checkpoint = ROOT / "runs/flat_base_20261019/pilot012/model_000576.pt"
source = ROOT / "models/wheel_leg_flat_base_20261018/source.onnx"
parent = torch.load(checkpoint, map_location="cpu", weights_only=True)
legacy = mean_from_checkpoint(parent, source)
declared = deepcopy(parent)
declared["infos"]["actor_adaptation"] = TRACKING_KIND
mean = mean_from_checkpoint(declared, source)
traces = np.load(ROOT / "runs/flat_base_20261019/tracking_gate576_native001/traces.npz")
values = traces["observations"][::5].reshape(-1, 35)
x = torch.from_numpy(values)
static = (x[:, :3] == 0).all(-1)
with torch.no_grad():
    expected = mean(x).numpy()
    static_identity = torch.equal(mean(x[static]), legacy(x[static]))
    moving_changed = not torch.equal(mean(x[~static]), legacy(x[~static]))
graph = mean.export_graph(source)
original = onnx.load(str(source))
original_mean, _ = recover_actor(source)
source_exact = all(a.SerializeToString() == b.SerializeToString()
                   for a,b in zip(original.graph.initializer, graph.graph.initializer))
# Copy the already audited I/O metadata, retaining the new gate declaration.
metadata = {item.key:item.value for item in onnx.load(str(
    ROOT / "models/wheel_leg_flat_feedback576_20261019/policy.onnx")).metadata_props}
metadata.update({item.key:item.value for item in graph.metadata_props
                 if item.key in ("actor_adaptation", "feedback_residual")})
metadata.update(diagnostic_counterfactual="tracking-gate-only-with-parent576-tensors",
                qualification_eligible="false", training_release="false")
onnx.helper.set_model_props(graph, metadata)
onnx.checker.check_model(graph, full_check=True)
directory = Path(__file__).with_name("tracking_gate576_operator_fixture002")
directory.mkdir(exist_ok=False)
output = directory / "diagnostic_counterfactual.onnx"
onnx.save(graph, output)
options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
actual = session.run(None, {"obs":values})[0]
np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
rmcs = Path("/home/noir/Documents/workspace/RMCS/rmcs_ws")
runtime = rmcs / "build/rmcs_rl/onnxruntime-linux-x64-1.20.0"
rl = rmcs / "src/rmcs_rl"
sources = [ROOT / "scripts/cpp/flat_base_rmcs_check.cpp", rl / "src/policy/onnx_runtime.cpp",
           rl / "src/policy/policy_model.cpp"]
binary = directory / "rmcs_policy_check"
subprocess.run(["g++", "-std=c++23", "-O2", f"-I{rl/'src'}", f"-I{rl/'include'}",
    f"-I{runtime/'include'}", *map(str,sources), f"-L{runtime/'lib'}", f"-Wl,-rpath,{runtime/'lib'}",
    "-lonnxruntime", "-o", str(binary)], check=True)
command = [str(binary), str(output), "35", metadata["rmcs_model_spec_sha256"], metadata["leg_winding_version"]]
fixture = "\n".join(" ".join(format(float(v), ".9g") for v in row) for row in values)
native = subprocess.run(command, input=fixture, text=True, capture_output=True, check=True)
cpp = np.fromstring(native.stdout, sep=" ").reshape(-1, 6)
np.testing.assert_allclose(cpp, actual, rtol=2e-5, atol=2e-5)
negative = {}
for name, index, value in (("wrong_obs_size", 2, "34"), ("wrong_model_spec", 3, "0"*64),
                            ("wrong_winding", 4, "independent_nearest_v1")):
    bad = command.copy(); bad[index] = value
    result = subprocess.run(bad, input="", text=True, capture_output=True)
    negative[name] = dict(rejected=result.returncode != 0, error=result.stderr.strip())
checks = dict(weights_all_exact=all(torch.equal(v, mean.state_dict()[n]) for n,v in legacy.state_dict().items()),
    original_eight_initializer_bytes_exact=source_exact, static_mean_exact=static_identity,
    moving_mean_is_declared_change=moving_changed, actual_ORT_and_CPP_close=True,
    invalid_contracts_rejected=all(row["rejected"] for row in negative.values()))
report = dict(schema="flat-tracking-gate576-operator-fixture-v1", checks=checks, passed=all(checks.values()),
    parent_checkpoint_sha256=sha(checkpoint), diagnostic_onnx_sha256=sha(output),
    new_optimizer_updates=0, qualification_eligible=False, actual_observation_samples=len(values),
    torch_ORT_max_error=float(abs(expected-actual).max()), CPP_ORT_max_error=float(abs(cpp-actual).max()),
    current_CPP_source_sha256={str(p):sha(p) for p in sources}, negatives=negative,
    gate_contract=mean.verify_original(original_mean)["feedback_zero_gate"])
(directory / "verification.json").write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
if not report["passed"]:
    raise SystemExit(1)

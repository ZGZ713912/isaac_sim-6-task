"""Offline identity/Adam and actual RMCS loader proof; no PPO or physics."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.guarded_feedback_residual import (GuardedFeedbackActor,
    configure_guarded_feedback_training, extend_feedback_parent)
from wheeled_tasks.chassis.flat_base_continuation import digest, encode

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
torch.set_num_threads(2)
source = ROOT / "models/wheel_leg_flat_base_20261018/source.onnx"
checkpoint = ROOT / "runs/flat_base_20261019/pilot010/model_000448.pt"
parent = torch.load(checkpoint, map_location="cpu", weights_only=True)
old_mean = mean_from_checkpoint(parent, source)
obs = TensorDict(dict(policy=torch.zeros(8, 35), critic=torch.zeros(8, 83)), batch_size=[8])
groups = dict(actor=["policy"], critic=["critic"])
old_actor = MLPModel(obs, groups, "actor", 6, hidden_dims=[256, 128, 64], activation="elu",
    obs_normalization=False, distribution_cfg=dict(class_name="GaussianDistribution", init_std=.04, std_type="scalar"))
old_actor.mlp = old_mean
old_actor.load_state_dict(parent["actor_state_dict"], strict=True)
critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[256, 128, 64], activation="elu", obs_normalization=False)
critic.load_state_dict(parent["critic_state_dict"], strict=True)
actor = deepcopy(old_actor)
actor.mlp = GuardedFeedbackActor(old_actor.mlp)
optimizer = torch.optim.Adam([dict(params=actor.parameters(), lr=1e-4, name="actor"),
                              dict(params=critic.parameters(), lr=5e-4, name="critic")])
adapted, proof = extend_feedback_parent(parent, actor, critic, optimizer, list(dict(old_actor.named_parameters())))
actor.load_state_dict(adapted["actor_state_dict"], strict=True)
optimizer.load_state_dict(adapted["optimizer_state_dict"])
values = np.load(ROOT / "runs/flat_base_20261019/rmcs448_envelope001/traces.npz")["observations"][::50].reshape(-1, 35)
inputs = torch.from_numpy(values)
with torch.no_grad():
    assert torch.equal(actor.mlp(inputs), old_actor.mlp(inputs))
assert adapted["infos"] == parent["infos"]
assert torch.equal(adapted["torch_rng_state"], parent["torch_rng_state"])
old_ids = parent["optimizer_state_dict"]["param_groups"][0]["params"]
new_ids = adapted["optimizer_state_dict"]["param_groups"][0]["params"]
for group in (0, 1):
    old = parent["optimizer_state_dict"]["param_groups"][group]["params"]
    new = adapted["optimizer_state_dict"]["param_groups"][group]["params"]
    for a, b in zip(old, new):
        if a in parent["optimizer_state_dict"]["state"]:
            assert all(torch.equal(v, adapted["optimizer_state_dict"]["state"][b][k])
                       for k, v in parent["optimizer_state_dict"]["state"][a].items())
assert all(k not in adapted["optimizer_state_dict"]["state"] for k in (13, 14, 15, 16))
scope = configure_guarded_feedback_training(actor, True)
assert scope["trainable_parameters"] == 1484
mean = actor.mlp.eval()
# A nonzero synthetic branch exercises the operators that zero initialization
# could otherwise mask. It is never saved as a trained checkpoint.
with torch.no_grad():
    mean.feedback[2].weight.normal_(std=.12)
    mean.feedback[2].bias.normal_(std=.05)
graph = mean.export_graph(source)
template = onnx.load(ROOT / "models/wheel_leg_flat_residual448_20261019/policy.onnx")
metadata = {x.key:x.value for x in template.metadata_props}
metadata.update({x.key:x.value for x in graph.metadata_props})
metadata["offline_fixture_only"] = "true; synthetic nonzero feedback, no training"
helper.set_model_props(graph, metadata)
original = onnx.load(source)
assert all(a.SerializeToString() == b.SerializeToString()
           for a,b in zip(original.graph.initializer, graph.graph.initializer))
model = OUT / "synthetic_operator_fixture.onnx"
onnx.save(graph, model)
options = ort.SessionOptions();options.intra_op_num_threads = options.inter_op_num_threads = 1
session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
with torch.no_grad():
    expected = mean(inputs).numpy()
actual = session.run(None, {"obs":values})[0]
np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
binary = ROOT / "models/wheel_leg_flat_residual448_20261019/rmcs_policy_check"
previous = json.loads((binary.parent / "verification.json").read_text())
for path, sha in previous["source_loader_sha256"].items():
    assert digest(path) == sha
command = [str(binary), str(model), "35", metadata["rmcs_model_spec_sha256"], "paired_branch107_v1"]
fixture = "\n".join(" ".join(format(float(x), ".9g") for x in row) for row in values)
native = subprocess.run(command, input=fixture, text=True, capture_output=True, check=True)
cpp = np.fromstring(native.stdout, sep=" ").reshape(-1, 6)
np.testing.assert_allclose(cpp, actual, rtol=2e-5, atol=2e-5)
result = dict(schema="flat-feedback-parent448-offline-proof-v1", parent_checkpoint_sha256=digest(checkpoint),
    source_onnx_sha256=digest(source), new_optimizer_updates=0, synthetic_branch_fixture=True,
    real_trace_samples=len(inputs), actor_identity_before_synthetic_mutation="bit_exact",
    original_source_initializer_bytes="all_eight_exact", inherited_Adam="all_existing_slots_bit_exact",
    new_feedback_Adam="empty", parent_counters_and_saved_RNG="bit_exact", migration=proof,
    trainable_parameters=scope["trainable_parameters"],
    torch_onnx_max_abs_error=float(np.abs(actual-expected).max()),
    rmcs_cpp_ort_max_abs_error=float(np.abs(cpp-actual).max()), hardware_motion_started=False,
    code_sha256={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),
        ROOT / "src/wheeled_algo/guarded_feedback_residual.py", ROOT / "src/wheeled_algo/command_residual.py")})
(OUT / "verification.json").write_bytes(encode(result))
print(json.dumps(result, indent=2))

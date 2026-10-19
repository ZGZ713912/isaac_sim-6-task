"""Recover a trainable mean while rejecting silently changed policy graphs."""
import hashlib

import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest
import torch

from wheeled_algo.onnx_actor_import import FORMAT, RecoveredActor, load_actor_mean, recover_actor, verify_outputs


@pytest.fixture
def source(tmp_path):
    torch.manual_seed(67)
    actor = RecoveredActor().eval()
    nodes, tensors = [], []
    previous = "observations"
    for index in (0, 2, 4, 6):
        weight, bias = f"exported_weight_{index}", f"exported_bias_{index}"
        tensors += [numpy_helper.from_array(actor.state_dict()[f"mlp.{index}.weight"].numpy(), weight),
                    numpy_helper.from_array(actor.state_dict()[f"mlp.{index}.bias"].numpy(), bias)]
        output = "commands" if index == 6 else f"linear_{index}"
        nodes.append(helper.make_node("Gemm", [previous, weight, bias], [output], transB=1))
        previous = output
        if index != 6:
            previous = f"activation_{index}"
            nodes.append(helper.make_node("Elu", [output], [previous]))
    graph = helper.make_graph(nodes, "actor",
        [helper.make_tensor_value_info("observations", onnx.TensorProto.FLOAT, ["batch", 35])],
        [helper.make_tensor_value_info("commands", onnx.TensorProto.FLOAT, ["batch", 6])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    path = tmp_path / "source.onnx"
    onnx.save(model, path)
    return path, actor


def test_exact_parameters_independent_runtime_and_trainability(source):
    path, original = source
    before = path.read_bytes()
    actor, report = recover_actor(path)
    assert report["parameter_count"] == 50758
    assert report["source_sha256"] == hashlib.sha256(before).hexdigest()
    for name, value in original.state_dict().items():
        assert torch.equal(value, actor.state_dict()[name])
    assert verify_outputs(actor, path)["sample_count"] == 1624
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    optimizer.zero_grad()
    actor(torch.ones(2, 35)).square().mean().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in actor.parameters())
    optimizer.step()
    assert not torch.equal(actor.mlp[0].weight, original.mlp[0].weight)
    assert path.read_bytes() == before
    assert "critic" in report["not_recoverable_from_this_graph"]


@pytest.mark.parametrize("mutation", ["activation", "gemm_scale", "extra_preprocessing"])
def test_reject_unrepresented_graph_semantics(source, mutation):
    path, _ = source
    model = onnx.load(path)
    if mutation == "activation":
        model.graph.node[1].op_type = "Relu"
    elif mutation == "gemm_scale":
        model.graph.node[0].attribute.append(helper.make_attribute("alpha", 2.0))
    else:
        model.graph.initializer.append(numpy_helper.from_array(np.ones(35, dtype=np.float32), "scale"))
        model.graph.node[0].input[0] = "normalized"
        model.graph.node.insert(0, helper.make_node("Mul", ["observations", "scale"], ["normalized"]))
    onnx.save(model, path)
    with pytest.raises(ValueError):
        recover_actor(path)


def test_stock_rsl_actor_mean_import_preserves_fresh_std_and_rng(source):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    path, _ = source
    recovered, report = recover_actor(path)
    obs = TensorDict({"policy": torch.zeros(1, 35)}, batch_size=[1])
    target = MLPModel(obs, {"actor": ["policy"]}, "actor", 6, hidden_dims=[256, 128, 64],
                      activation="elu", obs_normalization=False,
                      distribution_cfg={"class_name": "GaussianDistribution", "init_std": .1, "std_type": "scalar"})
    others = {key: value.clone() for key, value in target.state_dict().items() if not key.startswith("mlp.")}
    assert others
    rng_before = torch.get_rng_state().clone()
    result = load_actor_mean(target, {"format": FORMAT, "actor_mean_state_dict": recovered.state_dict(),
                                     "source_sha256": report["source_sha256"]})
    assert torch.equal(torch.get_rng_state(), rng_before)
    for key, value in others.items():
        assert torch.equal(target.state_dict()[key], value)
    for key, value in recovered.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)
    inputs = torch.randn(32, 35)
    torch.testing.assert_close(target.as_onnx(verbose=False)(inputs), recovered(inputs), rtol=0, atol=0)
    assert result["exploration"] == "retained_from_target_initialization"


def test_wrong_target_or_missing_parameters_are_rejected_without_partial_copy(source):
    path, _ = source
    recovered, report = recover_actor(path)
    target = RecoveredActor()
    payload = {"format": FORMAT, "source_sha256": report["source_sha256"],
               "actor_mean_state_dict": recovered.state_dict()}
    target.mlp[5] = torch.nn.ReLU()
    before = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="activation"):
        load_actor_mean(target, payload)
    for key, value in before.items():
        assert torch.equal(target.state_dict()[key], value)
    payload["actor_mean_state_dict"].pop("mlp.6.bias")
    with pytest.raises(ValueError, match="parameter set"):
        load_actor_mean(RecoveredActor(), payload)

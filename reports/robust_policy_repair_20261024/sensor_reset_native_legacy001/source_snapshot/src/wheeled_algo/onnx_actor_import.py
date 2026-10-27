"""Recover the deterministic chassis MLP from an ONNX inference graph.

This deliberately produces an actor mean, not a native learning checkpoint.
Observation semantics and any preprocessing outside the graph remain unknown.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn


FORMAT = "chassis_onnx_actor_mean_v1"
WIDTHS = (35, 256, 128, 64, 6)


class RecoveredActor(nn.Module):
    """Trainable deterministic mean with the original RSL parameter names."""

    def __init__(self):
        super().__init__()
        layers = []
        for index, (inputs, outputs) in enumerate(zip(WIDTHS, WIDTHS[1:])):
            layers.append(nn.Linear(inputs, outputs))
            if index < len(WIDTHS) - 2:
                layers.append(nn.ELU(alpha=1.0))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs):
        return self.mlp(obs)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def recover_actor(path):
    """Authenticate the complete supported graph before copying float32 tensors."""
    data = Path(path).read_bytes()
    model = onnx.load_model_from_string(data)
    onnx.checker.check_model(model, full_check=True)
    graph = model.graph
    _require(len(graph.input) == len(graph.output) == 1, "Expected one input and one output")
    for value, width in ((graph.input[0], WIDTHS[0]), (graph.output[0], WIDTHS[-1])):
        tensor = value.type.tensor_type
        _require(tensor.elem_type == onnx.TensorProto.FLOAT, "Expected float32 observations/actions")
        _require(len(tensor.shape.dim) == 2 and tensor.shape.dim[1].dim_value == width,
                 f"Expected a batch by {width} tensor")
    _require(not graph.sparse_initializer and not model.functions, "Unsupported graph extensions")
    _require([node.op_type for node in graph.node] ==
             ["Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"],
             "Only the complete four-layer Gemm/Elu actor graph is supported")
    tensors = {value.name: value for value in graph.initializer}
    _require(len(tensors) == len(graph.initializer) == 8, "Expected exactly eight parameter tensors")
    state, parameter_info, used = {}, [], set()
    previous = graph.input[0].name
    for node_index, node in enumerate(graph.node):
        _require(node.domain in ("", "ai.onnx"), "Unsupported operator domain")
        _require(len(node.output) == 1 and node.input[0] == previous, "Graph is not one sequential chain")
        attrs = {value.name: onnx.helper.get_attribute_value(value) for value in node.attribute}
        if node.op_type == "Elu":
            _require(len(node.input) == 1 and set(attrs) <= {"alpha"}
                     and attrs.get("alpha", 1.0) == 1.0, "Expected ELU with alpha=1")
        else:
            _require(len(node.input) == 3, "Expected Gemm weight and bias inputs")
            _require(set(attrs) <= {"alpha", "beta", "transA", "transB"}
                     and attrs.get("alpha", 1.0) == attrs.get("beta", 1.0) == 1.0
                     and attrs.get("transA", 0) == 0 and attrs.get("transB", 0) == 1,
                     "Unsupported Gemm scaling or transposition")
            layer = node_index // 2
            for suffix, name, shape in (("weight", node.input[1], (WIDTHS[layer + 1], WIDTHS[layer])),
                                        ("bias", node.input[2], (WIDTHS[layer + 1],))):
                _require(name in tensors and name not in used, "Missing or reused parameter")
                tensor = tensors[name]
                _require(tensor.data_location != onnx.TensorProto.EXTERNAL, "External parameters are unsupported")
                array = numpy_helper.to_array(tensor).copy()
                _require(array.dtype == np.float32 and array.shape == shape and np.isfinite(array).all(),
                         f"Invalid parameter: {name}")
                key = f"mlp.{node_index}.{suffix}"
                state[key] = torch.from_numpy(array)
                used.add(name)
                parameter_info.append({"onnx_name": name, "torch_name": key, "shape": list(shape),
                                       "sha256": hashlib.sha256(array.tobytes()).hexdigest()})
        previous = node.output[0]
    _require(previous == graph.output[0].name and used == tensors.keys(), "Unconsumed output or parameters")
    actor = RecoveredActor().cpu().eval()
    actor.load_state_dict(state, strict=True)
    metadata = {item.key: item.value for item in model.metadata_props}
    _require(len(metadata) == len(model.metadata_props), "Duplicate model metadata keys")
    report = {"format": FORMAT, "source_sha256": hashlib.sha256(data).hexdigest(),
              "input_name": graph.input[0].name, "output_name": graph.output[0].name,
              "input_dim": WIDTHS[0], "output_dim": WIDTHS[-1], "widths": list(WIDTHS),
              "activation": "elu", "parameter_count": sum(value.numel() for value in state.values()),
              "parameters": parameter_info, "metadata": metadata,
              "normalization_in_graph": False, "external_preprocessing": "unknown",
              "observation_semantics_in_graph": False,
              "not_recoverable_from_this_graph": ["critic", "exploration_std", "optimizer",
                  "training_updates", "reward", "observation_semantics", "action_semantics", "policy_rate"]}
    return actor, report


def verify_outputs(actor, source, *, seed=20261018):
    """Compare independent ONNX and PyTorch engines on multiple input scales."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(source), sess_options=options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    batches = [np.zeros((1, WIDTHS[0]), dtype=np.float32),
               np.eye(WIDTHS[0], dtype=np.float32), -np.eye(WIDTHS[0], dtype=np.float32)]
    batches += [(rng.normal(size=(count, WIDTHS[0])) * scale).astype(np.float32)
                for count, scale in ((17, .1), (512, 1.), (512, 10.), (512, 100.))]
    results = []
    with torch.inference_mode():
        for values in batches:
            expected = session.run(None, {session.get_inputs()[0].name: values})[0]
            actual = actor(torch.from_numpy(values)).numpy()
            _require(np.isfinite(expected).all() and np.isfinite(actual).all(), "Nonfinite verification output")
            # Cancellation near zero needs an absolute float32 error allowance
            # proportional to the stress input scale, in addition to rtol.
            atol = 2e-5 * max(1.0, float(np.abs(values).max()) / 10.0)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=atol)
            results.append({"count": len(values), "input_abs_max": float(np.abs(values).max()),
                            "atol": atol, "max_abs_error": float(np.abs(expected - actual).max())})
    return {"passed": True, "seed": seed, "rtol": 2e-5,
            "sample_count": sum(item["count"] for item in results), "batches": results,
            "max_abs_error": max(item["max_abs_error"] for item in results)}


def load_actor_mean(target, payload):
    """Copy the recovered mean into an RSL MLPModel; keep its fresh distribution.

    The caller still has to validate the environment's observation/action and
    physical contracts. A successful tensor load does not establish those.
    """
    _require(payload.get("format") == FORMAT, "Expected an ONNX actor-mean bundle, not a resume checkpoint")
    state = payload["actor_mean_state_dict"]
    _require(hasattr(target, "mlp"), "Target actor has no MLP")
    # Sequential iteration retains repeated uses of one shared ELU instance.
    layers = tuple(target.mlp)
    _require(len(layers) == 7, "Target actor architecture differs")
    for index, layer in enumerate(layers):
        if index % 2:
            _require(type(layer) is nn.ELU and layer.alpha == 1.0, "Target actor activation differs")
        else:
            _require(type(layer) is nn.Linear, "Target actor contains unsupported transforms")
    normalizer = getattr(target, "obs_normalizer", None)
    _require(normalizer is None or type(normalizer) is nn.Identity, "Target observation normalization is enabled")
    expected = {f"mlp.{index}.{suffix}": shape
                for index, (inputs, outputs) in zip((0, 2, 4, 6), zip(WIDTHS, WIDTHS[1:]))
                for suffix, shape in (("weight", (outputs, inputs)), ("bias", (outputs,)))}
    _require(state.keys() == expected.keys(), "Recovered actor parameter set differs")
    target_state = target.mlp.state_dict()
    _require(set(target_state) == {key.removeprefix("mlp.") for key in expected}, "Target actor parameter set differs")
    for key, shape in expected.items():
        _require(isinstance(state[key], torch.Tensor) and tuple(state[key].shape) == shape
                 and state[key].dtype == torch.float32 and torch.isfinite(state[key]).all().item(),
                 f"Invalid recovered state: {key}")
        _require(tuple(target_state[key.removeprefix("mlp.")].shape) == shape
                 and target_state[key.removeprefix("mlp.")].dtype == torch.float32,
                 f"Target actor shape/dtype differs: {key}")
    target.mlp.load_state_dict({key.removeprefix("mlp."): value for key, value in state.items()}, strict=True)
    return {"actor_mean": "exact_source_tensors", "exploration": "retained_from_target_initialization",
            "critic_optimizer": "must_be_initialized_by_training", "source_sha256": payload["source_sha256"]}

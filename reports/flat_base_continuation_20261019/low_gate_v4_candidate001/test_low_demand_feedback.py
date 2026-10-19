"""Gate endpoints, original state retention, native reproduction and ONNX parity."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import onnxruntime as ort
import pytest
import torch
from wheeled_algo.command_residual import CommandResidualActor, mean_from_checkpoint
from wheeled_algo.guarded_feedback_residual import GuardedFeedbackActor
from wheeled_algo.onnx_actor_import import RecoveredActor
from wheeled_algo.separated_feedback import SeparatedFeedbackActor
from test_guarded_feedback_residual import write_source

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('wheeled_algo.low_demand_feedback_candidate', HERE / 'low_demand_feedback.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Actor, configure = module.LowDemandFeedbackActor, module.configure_low_demand_feedback_training


@pytest.fixture
def parent():
    torch.manual_seed(702)
    result = SeparatedFeedbackActor(GuardedFeedbackActor(CommandResidualActor(RecoveredActor()), tracking_gate=True))
    with torch.no_grad():
        for branch in (result.feedback, result.fast_feedback):
            branch[2].weight.normal_(std=.2)
            branch[2].bias.normal_(std=.2)
    return result


def observations():
    generator = torch.Generator().manual_seed(501)
    obs = torch.randn(263, 35, generator=generator)
    obs[:, 0] *= 3
    obs[:, 2] *= 4
    obs[:9, :3] = torch.tensor([[0.,0.,0.],[.5,0.,0.],[-.5,0.,0.],
        [0.,0.,1.],[0.,0.,-1.],[.75,0.,0.],[1.5,0.,0.],[0.,0.,3.],[3.,0.,0.]])
    return obs


def test_extension_keeps_all_parameter_names_values_order_and_rng(parent):
    rng = torch.get_rng_state().clone()
    child = Actor(parent)
    assert torch.equal(torch.get_rng_state(), rng)
    assert list(child.state_dict()) == list(parent.state_dict())
    assert list(dict(child.named_parameters())) == list(dict(parent.named_parameters()))
    assert all(torch.equal(v, child.state_dict()[n]) for n, v in parent.state_dict().items())


def test_same_gate_as_the_native_counterfactual(parent):
    child, obs = Actor(parent), observations()
    demand = torch.maximum(obs[:, :1].abs()/.75, obs[:, 2:3].abs()/1.5)
    expected = torch.maximum(parent.tracking_feedback_gate(obs), (2.-demand).clamp(0,1))
    assert torch.equal(child.tracking_feedback_gate(obs), expected)
    assert torch.all(expected[:6] == 1)
    assert torch.all((expected >= 0) & (expected <= 1))


def test_static_and_fast_demand_means_are_bit_exact(parent):
    child, obs = Actor(parent), observations()
    demand = torch.maximum(obs[:,0].abs()/.75, obs[:,2].abs()/1.5)
    selected = (demand == 0) | (demand >= 2)
    assert torch.equal(child(obs[selected]), parent(obs[selected]))
    assert (child.feedback_actions(obs).abs() <= child.output_scale).all()


def test_training_scope_retains2252_feedback_parameters_and_frozen_command(parent):
    actor = torch.nn.Module(); actor.add_module('mlp', Actor(parent))
    actor.register_parameter('std', torch.nn.Parameter(torch.ones(6)))
    for p in actor.parameters():
        p.grad = torch.ones_like(p)
    result = configure(actor, True)
    assert result['trainable_parameters'] == 2252
    assert len(result['trainable_names']) == 8
    assert all(n.startswith(('mlp.feedback.', 'mlp.fast_feedback.')) for n in result['trainable_names'])
    assert all(p.grad is None for p in actor.parameters())
    configure(actor, False)
    assert all(not p.requires_grad for p in actor.parameters())


def test_full_high_command_loss_has_zero_direct_low_feedback_gradient(parent):
    child, obs = Actor(parent), observations()
    obs[:, 0] = 3.; obs[:, 2] = 0
    child(obs).square().mean().backward()
    assert all(p.grad is not None and not p.grad.count_nonzero() for p in child.feedback.parameters())
    assert any(p.grad.count_nonzero() for p in child.fast_feedback.parameters())
    assert all(p.grad is None for p in child.command.parameters())


def test_dynamic_onnx_matches_all_gate_regions_and_keeps_original_initializers(parent, tmp_path):
    child, obs = Actor(parent), observations()
    source = tmp_path / 'source.onnx'
    original = write_source(source, SimpleNamespace(mlp=parent.command.base))
    graph = child.export_graph(source)
    assert all(a.SerializeToString() == b.SerializeToString() for a,b in zip(original.graph.initializer, graph.graph.initializer))
    assert {p.key:p.value for p in graph.metadata_props}['actor_adaptation'] == module.KIND
    options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    for count in (1,9,263):
        np.testing.assert_allclose(session.run(None, {'obs':obs[:count].numpy()})[0], child(obs[:count]).detach().numpy(),
                                   atol=2e-5, rtol=2e-5)


def test_export_rejects_changed_source_or_bounds(parent, tmp_path):
    child = Actor(parent); source = tmp_path/'source.onnx'
    write_source(source, SimpleNamespace(mlp=parent.command.base))
    with torch.no_grad(): child.output_scale[4] *= 2
    with pytest.raises(ValueError, match='bound'): child.export_graph(source)


def test_actual1264_child_reproduces_the_native_gate_counterfactual():
    checkpoint = ROOT / 'runs/flat_base_20261019/pilot024/model_001264.pt'
    source = ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx'
    parent = mean_from_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=True), source)
    child = Actor(parent)
    trace = np.load(ROOT / 'runs/flat_base_20261019/low_gate1264_floor_native001/traces.npz')
    selected = trace['active'].astype(bool)
    obs, actions = torch.from_numpy(trace['observations'][selected]), torch.from_numpy(trace['actions'][selected])
    with torch.inference_mode():
        assert float((child(obs)-actions).abs().max()) < 5e-6
    trace.close()

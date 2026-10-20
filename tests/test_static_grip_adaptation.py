"""Static adaptation must have a gradient without changing moving-head ownership."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import nn

from wheeled_algo.command_residual import CommandResidualActor, mean_from_checkpoint
from wheeled_algo.guarded_feedback_residual import GuardedFeedbackActor
from wheeled_algo.low_demand_feedback import LowDemandFeedbackActor
from wheeled_algo.onnx_actor_import import RecoveredActor
from wheeled_algo.separated_feedback import SeparatedFeedbackActor, KIND as RECALL_KIND
from wheeled_algo.static_feedback import StaticFeedbackActor
from wheeled_algo.static_grip import StaticGripActor, KIND, configure_static_grip_training
from wheeled_algo.wheel_authority import WheelAuthorityActor, KIND as PARENT_KIND
from test_guarded_feedback_residual import write_source


@pytest.fixture
def means(tmp_path):
    torch.manual_seed(191040)
    shared = GuardedFeedbackActor(CommandResidualActor(RecoveredActor()), tracking_gate=True)
    recalled = SeparatedFeedbackActor(shared)
    with torch.no_grad():
        recalled.feedback[2].weight.normal_(std=.01)
    recalled_path = tmp_path / 'recalled.pt'
    torch.save(dict(infos=dict(actor_adaptation=RECALL_KIND), actor_state_dict={
        'mlp.' + n: v for n, v in recalled.state_dict().items()}), recalled_path)
    recall_sha = hashlib.sha256(recalled_path.read_bytes()).hexdigest()
    moving = LowDemandFeedbackActor(recalled)
    parent = WheelAuthorityActor(StaticFeedbackActor(moving, recalled, recall_sha))
    with torch.no_grad():
        parent.wheel_authority[2].weight.normal_(std=.01)
    return parent, StaticGripActor(parent), recalled_path


def observations():
    obs = torch.randn(257, 35)
    obs[:, :3] = 0.
    obs[:129, 0] = torch.linspace(-3., 3., 129)
    obs[129:, 2] = torch.linspace(-4., 4., 128)
    return obs


def test_identity_entry_preserves_all_parameters_buffers_and_actions(means):
    parent, child, _ = means
    assert list(parent.state_dict()) == list(child.state_dict())
    assert list(dict(parent.named_parameters())) == list(dict(child.named_parameters()))
    assert all(torch.equal(v, child.state_dict()[n]) for n, v in parent.state_dict().items())
    obs = observations()
    assert torch.equal(parent(obs), child(obs))


def test_static_gradient_changes_only_static_head_and_preserves_moving_function(means):
    parent, child, _ = means
    actor = nn.Module()
    actor.mlp = child
    actor.register_parameter('std', nn.Parameter(torch.full((6,), .04)))
    proof = configure_static_grip_training(actor, True)
    assert len(proof['trainable_names']) == 4
    assert all(n.startswith('mlp.static_feedback.') for n in proof['trainable_names'])
    obs = torch.randn(128, 35)
    obs[:, :3] = 0.
    moving = observations()
    full = child.moving_mix(moving).squeeze(-1) == 1
    before = deepcopy(actor.state_dict())
    static_before = child(obs).detach().clone()
    optimizer = torch.optim.Adam(actor.parameters(), lr=.0001)
    optimizer.zero_grad(set_to_none=True)
    (child(obs) - static_before - .01).square().mean().backward()
    assert all(p.grad is not None and p.grad.count_nonzero() for p in child.static_feedback.parameters())
    assert all(p.grad is None for name, p in actor.named_parameters()
               if not name.startswith('mlp.static_feedback.'))
    optimizer.step()
    assert not torch.equal(child(obs), static_before)
    assert torch.equal(child(moving[full]), parent(moving[full]))
    assert all(torch.equal(actor.state_dict()[name], before[name]) for name in proof['frozen_state_names'])
    disabled = configure_static_grip_training(actor, False)
    assert disabled['trainable_parameters'] == 0
    assert all(p.grad is None for p in actor.parameters())


def test_adapted_static_export_matches_pytorch_across_blends(means, tmp_path):
    parent, child, _ = means
    with torch.no_grad():
        child.static_feedback[2].bias.add_(.025)
    original = SimpleNamespace(mlp=parent.command.base)
    child.verify_original(original)
    with pytest.raises(ValueError, match='Frozen recalled'):
        parent.static_feedback.load_state_dict(child.static_feedback.state_dict())
        parent.verify_original(original)
    source = tmp_path / 'source.onnx'
    write_source(source, original)
    graph = child.export_graph(source)
    metadata = {item.key: item.value for item in graph.metadata_props}
    assert metadata['actor_adaptation'] == KIND
    assert json.loads(metadata['static_feedback_recall'])['static_trainable']
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    obs = observations()
    obs[:65, 0] = torch.linspace(-.1, .1, 65)
    obs[65:129, :3] = 0.
    with torch.inference_mode():
        expected = child(obs).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: obs.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=3e-6)


def test_checkpoint_reader_distinguishes_adapted_static_from_frozen_v6(means, tmp_path):
    parent, child, recalled = means
    with torch.no_grad():
        child.static_feedback[2].bias.add_(.01)
    source = tmp_path / 'source.onnx'
    write_source(source, SimpleNamespace(mlp=parent.command.base))
    checkpoint = dict(infos=dict(actor_adaptation=KIND,
        static_feedback_recall=dict(checkpoint=str(recalled), sha256=child.recalled_sha256)),
        actor_state_dict={'mlp.' + name: value for name, value in child.state_dict().items()})
    recovered = mean_from_checkpoint(checkpoint, source)
    obs = observations()
    assert type(recovered) is StaticGripActor
    assert torch.equal(recovered(obs), child(obs))
    checkpoint['infos']['actor_adaptation'] = PARENT_KIND
    with pytest.raises(ValueError, match='Frozen recalled'):
        mean_from_checkpoint(checkpoint, source)

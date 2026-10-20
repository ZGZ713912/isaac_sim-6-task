"""Verify the unadopted motion-only scope without touching replay sources."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import nn

from test_static_grip_adaptation import means, observations
from test_guarded_feedback_residual import write_source

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / 'reports/flat_base_continuation_20261019/motion_grip_candidate001/proposed_motion_grip.py'
spec = importlib.util.spec_from_file_location('proposed_motion_grip', PROPOSAL)
proposal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proposal)


def motion(means):
    parent = means[1]
    with torch.no_grad():
        parent.static_feedback[2].bias.add_(.015)
        parent.fast_feedback[2].weight.normal_(std=.01)
    return parent, proposal.MotionGripActor(parent, 'a' * 64)


def test_identity_retains_existing_parameter_order_and_function(means):
    parent, child = motion(means)
    assert list(parent.state_dict()) == list(child.state_dict())
    assert list(dict(parent.named_parameters())) == list(dict(child.named_parameters()))
    assert all(torch.equal(value, child.state_dict()[name]) for name, value in parent.state_dict().items())
    obs = observations()
    assert torch.equal(parent(obs), child(obs))


def test_gradient_updates_motion_heads_and_preserves_stable_static_function(means):
    parent, child = motion(means)
    actor = nn.Module()
    actor.mlp = child
    actor.register_parameter('std', nn.Parameter(torch.full((6,), .04)))
    proof = proposal.configure_motion_grip_training(actor, True)
    assert proof['trainable_parameters'] == 1126 * 2 + 994
    assert len(proof['trainable_names']) == 12
    obs = observations()
    static = torch.randn(64, 35)
    static[:, :3] = 0.
    static[:, 20:22] = .1
    before = deepcopy(actor.state_dict())
    optimizer = torch.optim.Adam(actor.parameters(), lr=.0001)
    output = child(obs).detach()
    (child(obs) - output - .02).square().mean().backward()
    assert all(p.grad is not None and p.grad.count_nonzero() for name, p in actor.named_parameters()
               if name in proof['trainable_names'])
    assert all(p.grad is None for name, p in actor.named_parameters() if name not in proof['trainable_names'])
    optimizer.step()
    assert all(not torch.equal(actor.state_dict()[name], before[name]) for name in proof['trainable_names'])
    assert all(torch.equal(actor.state_dict()[name], before[name]) for name in proof['frozen_state_names'])
    assert torch.equal(parent(static), child(static))
    assert not torch.equal(parent(obs), child(obs))
    assert proposal.configure_motion_grip_training(actor, False)['trainable_parameters'] == 0
    assert all(p.grad is None for p in actor.parameters())


def test_export_matches_adapted_motion_and_preserves_static_reference(means, tmp_path):
    parent, child = motion(means)
    original = SimpleNamespace(mlp=parent.command.base)
    with torch.no_grad():
        child.feedback[2].bias.add_(.02)
        child.fast_feedback[2].bias.sub_(.02)
        child.wheel_authority[2].bias.add_(.03)
    source = tmp_path / 'source.onnx'
    write_source(source, original)
    graph = child.export_graph(source)
    metadata = {item.key: item.value for item in graph.metadata_props}
    assert metadata['actor_adaptation'] == proposal.KIND
    assert 'static_grip_adaptation' not in metadata
    assert not json.loads(metadata['static_feedback_recall'])['static_trainable']
    assert json.loads(metadata['motion_grip_adaptation'])['parent_checkpoint_sha256'] == 'a' * 64
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=options,
                                  providers=['CPUExecutionProvider'])
    obs = observations()
    with torch.inference_mode():
        expected = child(obs).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: obs.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    with torch.no_grad():
        child.static_feedback[2].bias.add_(.001)
    with pytest.raises(ValueError, match='Frozen adapted static'):
        child.verify_original(original)


def test_rejects_other_parent_and_undeclared_command_training(means):
    with pytest.raises(ValueError, match='exact v7'):
        proposal.MotionGripActor(means[0], 'a' * 64)
    _, child = motion(means)
    with pytest.raises(ValueError, match='freezes the command'):
        child.set_trainable(True, train_command=True)

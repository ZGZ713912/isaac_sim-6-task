"""Moving optimization must preserve the recalled stationary controller."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch

from wheeled_algo.command_residual import CommandResidualActor
from wheeled_algo.guarded_feedback_residual import GuardedFeedbackActor
from wheeled_algo.low_demand_feedback import LowDemandFeedbackActor
from wheeled_algo.onnx_actor_import import RecoveredActor
from wheeled_algo.separated_feedback import SeparatedFeedbackActor
from wheeled_algo.static_feedback import StaticFeedbackActor
from test_guarded_feedback_residual import write_source


@pytest.fixture
def means():
    torch.manual_seed(191036)
    shared = GuardedFeedbackActor(CommandResidualActor(RecoveredActor()), tracking_gate=True)
    with torch.no_grad():
        shared.feedback[2].weight.normal_(std=.1)
    recalled = SeparatedFeedbackActor(shared)
    moving = LowDemandFeedbackActor(recalled)
    with torch.no_grad():
        moving.feedback[2].bias.add_(.15)
        moving.fast_feedback[2].weight.add_(.1)
    return moving, recalled, StaticFeedbackActor(moving, recalled, 'a' * 64)


def test_Adam_updates_moving_feedback_without_changing_static_actions(means):
    moving, recalled, child = means
    stationary = torch.randn(128, 35)
    stationary[:, :3] = 0.
    moving_obs = torch.randn(128, 35)
    moving_obs[:, 0] = torch.linspace(-3., 3., 128)
    stationary_before = child(stationary).detach().clone()
    moving_before = child(moving_obs).detach().clone()
    frozen_before = deepcopy(child.static_feedback.state_dict())
    optimizer = torch.optim.Adam(child.parameters(), lr=.003)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (child(moving_obs) - moving_before - .03).square().mean().backward()
        optimizer.step()
    assert not torch.equal(child(moving_obs), moving_before)
    assert torch.equal(child(stationary), stationary_before)
    assert torch.equal(child(stationary), recalled(stationary))
    assert all(torch.equal(v, child.static_feedback.state_dict()[n]) for n, v in frozen_before.items())
    assert all(p.grad is None and not p.requires_grad for p in child.static_feedback.parameters())


def test_static_loss_cannot_update_either_moving_branch(means):
    _, _, child = means
    obs = torch.randn(41, 35)
    obs[:, :3] = 0.
    child(obs).square().mean().backward()
    assert all(p.grad is not None and p.grad.count_nonzero() == 0
               for head in (child.feedback, child.fast_feedback) for p in head.parameters())


def test_changed_reference_command_and_static_tensor_are_rejected(means):
    moving, recalled, child = means
    with torch.no_grad():
        recalled.command.residual[2].bias.add_(.1)
    with pytest.raises(ValueError, match='unchanged command'):
        StaticFeedbackActor(moving, recalled, 'a' * 64)
    with torch.no_grad():
        child.static_feedback[2].bias.add_(.1)
    with pytest.raises(ValueError, match='Frozen recalled'):
        child.verify_original(SimpleNamespace(mlp=moving.command.base))


def test_export_preserves_static_moving_and_transition_actions(means, tmp_path):
    moving, _, child = means
    source = tmp_path / 'source.onnx'
    write_source(source, SimpleNamespace(mlp=moving.command.base))
    model = child.export_graph(source)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    obs = torch.randn(257, 35)
    obs[:, :3] = 0.
    obs[:129, 0] = torch.linspace(-.1, .1, 129)
    obs[129:, 2] = torch.linspace(-4., 4., 128)
    with torch.inference_mode():
        expected = child(obs).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: obs.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=3e-6)

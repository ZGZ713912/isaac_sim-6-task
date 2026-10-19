"""Keep the parent function and low-command feedback intact when adding a branch."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch

from wheeled_algo.command_residual import CommandResidualActor, mean_from_checkpoint
from wheeled_algo.guarded_feedback_residual import GuardedFeedbackActor, TRACKING_KIND
from wheeled_algo.onnx_actor_import import RecoveredActor
from wheeled_algo.separated_feedback import (
    KIND, SeparatedFeedbackActor, configure_separated_feedback_training,
    extend_separated_feedback_parent,
)
from test_guarded_feedback_residual import write_source


@pytest.fixture
def parent():
    torch.manual_seed(1319)
    mean = GuardedFeedbackActor(CommandResidualActor(RecoveredActor()), tracking_gate=True)
    with torch.no_grad():
        mean.feedback[2].weight.normal_(std=.2)
        mean.feedback[2].bias.fill_(.1)
    return mean


def observations():
    x = torch.randn(257, 35)
    x[:, 0] *= 3
    x[:, 2] *= 4
    x[:8, :3] = torch.tensor([[0., 0., 0.], [.5, 0., 0.], [-.5, 0., 0.],
        [0., 0., 1.], [0., 0., -1.], [.75, 0., 1.5], [1.125, 0., 0.], [3., 0., 0.]])
    return x


def test_clone_preserves_parent_at_endpoints_and_interpolation(parent):
    before = deepcopy(parent.state_dict())
    child = SeparatedFeedbackActor(parent)
    x = observations()
    assert torch.equal(child(x), parent(x))
    assert all(torch.equal(v, child.state_dict()[n]) for n, v in before.items())
    with torch.no_grad():
        child.fast_feedback[2].bias.add_(.7)
    assert torch.equal(child(x[:6]), parent(x[:6]))
    assert all(torch.equal(v, parent.state_dict()[n]) for n, v in before.items())
    assert (child.feedback_actions(x).abs() <= child.output_scale).all()


def test_high_command_teacher_has_no_direct_low_branch_gradient(parent):
    child = SeparatedFeedbackActor(parent)
    x = observations()
    x[:4, :3] = torch.tensor([[3., 0., 0.], [-3., 0., 0.], [0., 0., 4.], [0., 0., -4.]])
    low_before = child(x[4:6]).detach().clone()
    (child(x[:4]) - .3).square().mean().backward()
    assert all(p.grad is not None and p.grad.count_nonzero() == 0 for p in child.feedback.parameters())
    assert any(p.grad.count_nonzero() for p in child.fast_feedback.parameters())
    with torch.no_grad():
        for p in child.fast_feedback.parameters():
            p.add_(p.grad, alpha=-.001)
    assert torch.equal(child(x[4:6]), low_before)
    assert all(p.grad is None for p in child.command.parameters())


def test_training_scope_clears_frozen_gradients(parent):
    actor = torch.nn.Module()
    actor.add_module('mlp', SeparatedFeedbackActor(parent))
    actor.register_parameter('std', torch.nn.Parameter(torch.ones(6)))
    for p in actor.parameters():
        p.grad = torch.ones_like(p)
    scope = configure_separated_feedback_training(actor, True)
    assert scope['trainable_parameters'] == 2252
    assert all(n.startswith(('mlp.feedback.', 'mlp.fast_feedback.')) for n in scope['trainable_names'])
    assert all(p.grad is None for p in actor.parameters())
    configure_separated_feedback_training(actor, False)
    assert all(not p.requires_grad for p in actor.parameters())
    with pytest.raises(ValueError, match='freezes'):
        actor.mlp.set_trainable(True, train_command=True)


@pytest.mark.parametrize('sparse', [False, True])
def test_adam_identity_mapping_retains_critic_and_inherited_counters(parent, sparse):
    actor = torch.nn.Module(); actor.add_module('mlp', parent)
    actor.register_parameter('std', torch.nn.Parameter(torch.ones(6)))
    critic = torch.nn.Linear(83, 1)
    old_names = list(dict(actor.named_parameters()))
    opt = torch.optim.Adam([dict(params=actor.parameters(), lr=3e-4), dict(params=critic.parameters(), lr=5e-4)])
    state = opt.state_dict()
    params = list(actor.parameters()) + list(critic.parameters())
    for i, p in enumerate(params):
        if not sparse or i % 2:
            state['state'][i] = dict(step=torch.tensor(float(i+1)), exp_avg=torch.full_like(p, i+.1),
                                     exp_avg_sq=torch.full_like(p, i+.2))
    checkpoint = dict(actor_state_dict=deepcopy(actor.state_dict()), critic_state_dict=deepcopy(critic.state_dict()),
        optimizer_state_dict=state, torch_rng_state=torch.get_rng_state(),
        infos=dict(actor_adaptation=TRACKING_KIND, feedback_branch_actor_updates=304,
                   successful_updates_total=832, consumed_optimizer_updates_total=833))
    actor.mlp = SeparatedFeedbackActor(parent)
    new_opt = torch.optim.Adam([dict(params=actor.parameters(), lr=3e-4), dict(params=critic.parameters(), lr=5e-4)])
    adapted, proof = extend_separated_feedback_parent(checkpoint, actor, critic, new_opt, old_names)
    new_names = list(dict(actor.named_parameters()))
    old_ids = dict(zip(old_names, state['param_groups'][0]['params'], strict=True))
    for n, i in zip(new_names, adapted['optimizer_state_dict']['param_groups'][0]['params'], strict=True):
        old_id = old_ids[n.replace('mlp.fast_feedback.', 'mlp.feedback.', 1)]
        a, b = adapted['optimizer_state_dict']['state'].get(i), state['state'].get(old_id)
        assert (a is None) == (b is None)
        if a is not None:
            assert all(torch.equal(a[k], v) for k, v in b.items())
    for a, b in zip(state['param_groups'][1]['params'], adapted['optimizer_state_dict']['param_groups'][1]['params'], strict=True):
        if a in state['state']:
            assert all(torch.equal(v, adapted['optimizer_state_dict']['state'][b][k]) for k, v in state['state'][a].items())
    assert adapted['infos'] == checkpoint['infos']
    assert proof['post_split_actor_updates'] == 0
    assert torch.equal(adapted['torch_rng_state'], checkpoint['torch_rng_state'])
    with pytest.raises(ValueError, match='order'):
        extend_separated_feedback_parent(checkpoint, actor, critic, new_opt, list(reversed(old_names)))


def test_diverged_branches_roundtrip_and_dynamic_onnx(parent, tmp_path):
    child = SeparatedFeedbackActor(parent)
    with torch.no_grad():
        child.fast_feedback[2].bias.add_(.2)
    source = tmp_path / 'source.onnx'
    original = write_source(source, SimpleNamespace(mlp=parent.command.base))
    checkpoint = dict(actor_state_dict={'mlp.'+n:v for n,v in child.state_dict().items()}, infos=dict(actor_adaptation=KIND))
    restored = mean_from_checkpoint(checkpoint, source)
    x = observations()
    assert torch.equal(child(x), restored(x))
    graph = restored.export_graph(source)
    assert all(a.SerializeToString() == b.SerializeToString() for a,b in zip(original.graph.initializer, graph.graph.initializer))
    options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    for size in (1, 17, 257):
        np.testing.assert_allclose(session.run(None, {'obs':x[:size].numpy()})[0], child(x[:size]).detach().numpy(),
                                   atol=2e-5, rtol=2e-5)
    with torch.no_grad():
        child.output_scale[0] *= 2
    with pytest.raises(ValueError, match='bound'):
        child.export_graph(source)

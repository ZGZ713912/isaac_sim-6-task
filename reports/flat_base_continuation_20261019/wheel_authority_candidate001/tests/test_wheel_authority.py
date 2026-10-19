"""A learned wheel extension must preserve existing low-command behavior."""
from copy import deepcopy
import pytest
import torch

from wheeled_algo.command_residual import CommandResidualActor
from wheeled_algo.guarded_feedback_residual import GuardedFeedbackActor
from wheeled_algo.low_demand_feedback import LowDemandFeedbackActor
from wheeled_algo.onnx_actor_import import RecoveredActor
from wheeled_algo.separated_feedback import SeparatedFeedbackActor
from wheeled_algo.static_feedback import StaticFeedbackActor
from wheeled_algo.wheel_authority import WheelAuthorityActor, configure_wheel_authority_training


@pytest.fixture
def means():
    torch.manual_seed(191037)
    shared = GuardedFeedbackActor(CommandResidualActor(RecoveredActor()), tracking_gate=True)
    with torch.no_grad():
        shared.feedback[2].weight.normal_(std=.1)
    static = SeparatedFeedbackActor(shared)
    moving = LowDemandFeedbackActor(static)
    parent = StaticFeedbackActor(moving, static, 'a'*64)
    return parent, WheelAuthorityActor(parent)


def test_actual_Adam_updates_only_new_head_and_preserves_low_commands(means):
    parent, child = means
    x = torch.randn(128, 35)
    x[:64, 0], x[:64, 2] = -3., 0.
    x[64:, :3] = 0.
    x[64:, 20:22] = torch.tensor([2., -2.])
    low = x.clone()
    low[:, 0], low[:, 2] = -.5, 0.
    static = low.clone()
    static[:, :3], static[:, 20:22] = 0., 0.
    expected = parent(x).detach()
    assert torch.equal(child(x), expected)
    before = deepcopy(child.state_dict())
    optimizer = torch.optim.Adam(child.parameters(), lr=.003)
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        (child(x)[:, 4:6] - expected[:, 4:6] - .1).square().mean().backward()
        optimizer.step()
    assert not torch.equal(child(x), expected)
    assert torch.equal(child(low), parent(low))
    assert torch.equal(child(static), parent(static))
    assert all(torch.equal(v, child.state_dict()[n]) for n,v in before.items()
               if not n.startswith('wheel_authority.'))
    assert all(p.grad is None for n,p in child.named_parameters() if not n.startswith('wheel_authority.'))


def test_scope_and_braking_gate_remain_explicit(means):
    _, child = means
    actor = torch.nn.Module()
    actor.mlp = child
    actor.std = torch.nn.Parameter(torch.ones(6))
    scope = configure_wheel_authority_training(actor, True)
    assert scope['trainable_parameters'] == 994
    assert all(n.startswith('mlp.wheel_authority.') for n in scope['trainable_names'])
    obs = torch.zeros(4, 35)
    obs[:, 20] = torch.tensor([.39, .4, .6, .8])/.6
    torch.testing.assert_close(child.authority_gate(obs).squeeze(), torch.tensor([0., 0., .5, 1.]), atol=1e-6, rtol=0.)
    obs[:, 0] = .5
    assert not child.authority_gate(obs).count_nonzero()
    configure_wheel_authority_training(actor, False)
    assert not any(p.requires_grad for p in actor.parameters())


def test_existing_controller_bounds_are_not_encoded_as_new_action_clipping(means):
    parent, child = means
    with torch.no_grad():
        child.wheel_authority[2].bias.fill_(1000.)
    obs = torch.zeros(16, 35)
    obs[:, 0] = 3.
    correction = child.authority_actions(obs)
    assert torch.equal(correction[:, :4], torch.zeros(16, 4))
    assert bool((correction[:, 4:6] == 6.).all())
    assert torch.equal(child(obs), parent(obs)+correction)

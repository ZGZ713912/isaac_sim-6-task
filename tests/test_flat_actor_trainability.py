"""A head update must preserve hidden features, exploration and their Adam state."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from wheeled_algo.flat_actor_trainability import configure_flat_actor_training
from wheeled_algo.onnx_actor_import import RecoveredActor


class Actor(RecoveredActor):
    def __init__(self):
        super().__init__()
        self.distribution = nn.Module()
        self.distribution.std_param = nn.Parameter(torch.full((6,), .025))


def test_a_real_adam_update_changes_the_head_and_preserves_initialized_frozen_slots():
    torch.manual_seed(31)
    actor = Actor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    x = torch.randn(17, 35)
    # Create nonzero momentum for every parameter before freezing it.
    (actor(x).square().mean() + actor.distribution.std_param.sum()).backward()
    optimizer.step()
    before = deepcopy(actor.state_dict())
    slots = {name: deepcopy(optimizer.state[p]) for name, p in actor.named_parameters()}
    result = configure_flat_actor_training(actor, enabled=True, output_layer_only=True)
    assert result["trainable_parameters"] == 390
    assert result["exploration_std_frozen"]
    # Freezing must clear stale gradients, including when the next caller uses
    # zero tensors for active parameters instead of setting them to None.
    optimizer.zero_grad(set_to_none=False)
    actor(x).square().mean().backward()
    optimizer.step()
    for name, parameter in actor.named_parameters():
        if name in result["trainable_names"]:
            assert not torch.equal(before[name], parameter)
        else:
            assert parameter.grad is None and torch.equal(before[name], parameter)
            assert all(torch.equal(value, optimizer.state[parameter][key]) for key, value in slots[name].items())


def test_warmup_and_normal_training_keep_the_original_all_or_none_behavior():
    actor = Actor()
    assert configure_flat_actor_training(actor, enabled=False, output_layer_only=True)["trainable_parameters"] == 0
    assert not any(p.requires_grad for p in actor.parameters())
    assert configure_flat_actor_training(actor, enabled=True)["trainable_parameters"] == 50764
    assert all(p.requires_grad for p in actor.parameters())


def test_unknown_output_shape_is_rejected():
    with pytest.raises(ValueError, match="recovered"):
        configure_flat_actor_training(nn.Linear(35, 6), enabled=True, output_layer_only=True)

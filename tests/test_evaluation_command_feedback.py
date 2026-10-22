"""Check deployable sensor units, inactive behavior, and zero-intervention parity."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from wheeled_algo.evaluation_command_feedback import StandingCommandFeedbackParent


def states():
    x = torch.zeros(12, 35)
    x[:, 9] = -1.
    x[:, 3] = 1.525
    return x


def test_zero_gain_keeps_parent_weights_rng_and_actions_exact():
    parent = nn.Linear(35, 6)
    rng = torch.get_rng_state().clone()
    wrapped = StandingCommandFeedbackParent(deepcopy(parent), 0.)
    assert torch.equal(torch.get_rng_state(), rng)
    x = torch.randn(12, 35)
    assert torch.equal(parent(x), wrapped(x))
    for name, value in parent.state_dict().items():
        assert torch.equal(value, wrapped.inner.state_dict()[name])


def test_encoder_sign_scale_and_request_bound():
    x = states()
    wrapped = StandingCommandFeedbackParent(nn.Identity(), 2.)
    x[:, 20] = .1
    x[:, 21] = -.1
    torch.testing.assert_close(wrapped(x)[:, 0], torch.full((12,), -.12))
    assert torch.equal(wrapped(x)[:, 1:], x[:, 1:])
    x[:, 20:22] *= 100.
    torch.testing.assert_close(wrapped(x)[:, 0], torch.full((12,), -.15))


@pytest.mark.parametrize("index,value", [(0, .12), (2, .25), (9, 0.), (4, 1.)])
def test_motion_recovery_or_disturbance_retains_original_input(index, value):
    x = states()
    x[:, 20:22] = torch.tensor([.5, -.5])
    x[:, index] = value
    assert torch.equal(StandingCommandFeedbackParent(nn.Identity(), 2.)(x), x)


@pytest.mark.parametrize("gain", [float("nan"), float("inf"), -8.01, 8.01])
def test_invalid_gains_are_rejected(gain):
    with pytest.raises(ValueError):
        StandingCommandFeedbackParent(nn.Identity(), gain)


def test_reference_attribute_and_previous_action_channels_retained():
    parent = nn.Module()
    parent.command = nn.Linear(35, 6)
    wrapped = StandingCommandFeedbackParent(parent, 1.)
    assert wrapped.command is parent.command
    x = states()
    first = wrapped.command_delta(x)
    x[:, 22:28] = 3.
    assert torch.equal(wrapped.command_delta(x), first)

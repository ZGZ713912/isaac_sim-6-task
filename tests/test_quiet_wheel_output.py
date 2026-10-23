"""Whole-output filtering preserves the explicit policy scope and fixed points."""
import pytest
import torch
from torch import nn

from evaluate_robust_getup_batch import quiet_filter_variants
from wheeled_algo.evaluation_quiet_output import QuietWheelOutputFilter


class FixtureMean(nn.Module):
    def forward(self, obs):
        return obs[:, 16:22].clamp(-2., 2.)


def observations():
    generator = torch.Generator().manual_seed(73)
    obs = torch.randn(2000, 35, generator=generator) * .1
    obs[:, :3] = 0.
    obs[:, 4:6] = 0.
    obs[:, 9] = -1.
    return obs


def test_alpha_one_is_exact_and_filter_does_not_consume_rng_or_add_state():
    mean, obs = FixtureMean(), observations()
    before = torch.get_rng_state()
    wrapper = QuietWheelOutputFilter(mean, 1.)
    assert torch.equal(wrapper(obs), mean(obs))
    assert torch.equal(before, torch.get_rng_state())
    assert not wrapper.state_dict()


@pytest.mark.parametrize("alpha", [.05, .1, .25, .5])
def test_legs_and_nonquiet_outputs_exact_with_fixed_point_and_expected_quiet_filter(alpha):
    mean, obs = FixtureMean(), observations()
    action = mean(obs)
    wrapper = QuietWheelOutputFilter(mean, alpha)
    result = wrapper(obs)
    assert torch.equal(result[:, :4], action[:, :4])
    expected = obs[:, 26:28] + alpha * (action[:, 4:] - obs[:, 26:28])
    assert torch.equal(result[:, 4:], expected)
    obs[:, 26:28] = action[:, 4:]
    assert torch.equal(wrapper(obs), action)
    obs[:, 0] = .5
    assert torch.equal(wrapper(obs), action)
    obs[:, 0] = 0.
    obs[:, 9] = 1.
    assert torch.equal(wrapper(obs), action)


@pytest.mark.parametrize("alpha", [0., -.1, .049, 1.01, float("nan"), float("inf"), True])
def test_invalid_filter_coefficients_rejected(alpha):
    with pytest.raises(ValueError):
        QuietWheelOutputFilter(FixtureMean(), alpha)


@pytest.mark.parametrize("alphas", [[], [1., 1.], [1., .5, .25, .1, .05], [float("nan")], [.049]])
def test_invalid_grid_rejected(alphas):
    with pytest.raises(ValueError):
        quiet_filter_variants(alphas)


def test_grid_names_are_unique_and_keep_declared_coefficients():
    values = [1., .5, .25, .1]
    assert list(quiet_filter_variants(values).values()) == values

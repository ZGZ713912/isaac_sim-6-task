"""Preserve old prediction scope and enforce the physical35D velocity units."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from wheeled_algo.robust_getup import RobustGetupActor


def position_parent():
    parent = nn.Linear(35, 6)
    with torch.no_grad():
        parent.weight.zero_()
        parent.bias.zero_()
        parent.weight[:4, 10:14] = torch.eye(4)
    return parent


def test_scope_changes_only_lower_quiet_requests_and_keeps_high_endpoint_exact():
    old = RobustGetupActor(position_parent(), quiet_alpha=(1.,) * 6,
                          quiet_high_leg_prediction_s=.005)
    new = deepcopy(old)
    new.quiet_leg_prediction_all_heights = True
    obs = torch.zeros(6, 35)
    obs[:, 9] = -1.
    obs[:, 3] = torch.tensor([.23, .305, .42, .305, .305, .42]) * 5.
    obs[:, 10:14] = .1
    obs[:, 16:20] = .1  # 1rad/s; a5ms forecast advances q by.005rad.
    obs[3, 0] = .5
    obs[4, 2] = .5
    obs[5, 9] = .1
    before, after = old(obs), new(obs)
    assert torch.equal(before[2:], after[2:])
    torch.testing.assert_close(after[:2, :4] - before[:2, :4], torch.full((2, 4), .005))
    assert torch.equal(before[:, 4:], after[:, 4:])


def test_zero_velocity_is_exact_and_scope_adds_no_parameters_or_buffers():
    old = RobustGetupActor(position_parent(), quiet_high_leg_prediction_s=.005)
    new = deepcopy(old)
    new.quiet_leg_prediction_all_heights = True
    obs = torch.zeros(64, 35)
    obs[:, 9] = -1.
    obs[:, 3] = torch.linspace(.2, .45, 64) * 5.
    obs[:, 10:14] = .3
    assert torch.equal(old(obs), new(obs))
    assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", float("nan")])
def test_scope_requires_boolean(invalid):
    with pytest.raises(ValueError, match="explicit boolean"):
        RobustGetupActor(position_parent(), quiet_high_leg_prediction_s=.005,
                         quiet_leg_prediction_all_heights=invalid)


def test_enabled_scope_requires_positive_prediction_duration():
    with pytest.raises(ValueError, match="positive prediction duration"):
        RobustGetupActor(position_parent(), quiet_leg_prediction_all_heights=True)

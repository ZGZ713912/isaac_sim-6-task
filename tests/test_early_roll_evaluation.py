"""Early authority changes only the inspected roll domain and retains limits."""
import math
from unittest.mock import patch

import pytest
import torch
from torch import nn

from wheeled_algo.evaluation_early_roll import EarlyRollEvaluation, roll_gate
from wheeled_algo.getup_symmetry import canonical_recovery_logits
from wheeled_algo.recovery_coordinates import smooth_recovery_legs
from wheeled_algo.robust_getup import RobustGetupActor


def actor():
    torch.manual_seed(77)
    return RobustGetupActor(nn.Linear(35, 6), canonical_recovery=True, leg_action_clip=16.,
        cyclic_leg_targets=True, smooth_knee_targets=True, stance_aware_recovery=True, recovery_shape_aware=True)


def test_quiet_and_reference_valid_domains_exact():
    mean = actor()
    obs = torch.zeros(100, 35)
    obs[:, 9] = -1.
    old = mean(obs)
    assert torch.equal(EarlyRollEvaluation(mean, 20., 25.)(obs), old)
    assert math.sqrt(1. - .94 ** 2) < math.sin(math.radians(20.))
    assert not bool((roll_gate(mean, obs, 20., 25.)[1] > 0.).any())


@pytest.mark.parametrize("full", [25., 45.])
def test_changed_actions_agree_with_explicit_convex_blend(full):
    mean = actor()
    obs = torch.zeros(100, 35)
    angle = torch.linspace(21., 80., 100) * (math.pi / 180.)
    obs[:, 8], obs[:, 9] = angle.sin(), -angle.cos()
    _, gate = roll_gate(mean, obs, 20., full)
    with patch("wheeled_algo.robust_getup.tilt_gate", lambda x, *_: torch.zeros_like(x[:, :1])):
        motion = mean(obs)
    raw = canonical_recovery_logits(mean.recovery, obs, mean.feature_scale)
    legs = smooth_recovery_legs(raw[:, :4], obs[:, 22:26], obs[:, 10:14],
                                mean.recovery_leg_step, mean.recovery_knee_center)
    fallen = torch.cat((legs, 9. * torch.tanh(raw[:, 4:6])), -1)
    expected = motion + gate * (fallen - motion)
    actual = EarlyRollEvaluation(mean, 20., full)(obs)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=0.)
    assert bool((actual.abs() <= mean.action_bound).all())


@pytest.mark.parametrize("start,full", [(19., 25.), (20., 20.), (20., 61.), (float("nan"), 45.), (20., float("nan"))])
def test_invalid_ranges_rejected(start, full):
    with pytest.raises(ValueError):
        EarlyRollEvaluation(actor(), start, full)

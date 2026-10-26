"""Circular actor requests must survive the deployed paired target decoder."""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from wheeled_algo.paired_recovery import DELTA_BOUNDS, NOMINAL_DELTA, wrap
from wheeled_algo.upright_target_limit import UprightTargetLimitedGapActor, rebase_final_leg_pairs
from wheeled_algo.gap_entry_preservation import EntryPreservingGapActor
from wheeled_tasks.chassis.paired_leg_targets import paired_leg_targets


def test_wrapped_feasible_targets_retain_physical_angles_and_paired_decoder_requests():
    torch.manual_seed(921)
    common = (torch.rand(30000, 2) * 2 - 1) * math.pi
    bounds = torch.tensor(DELTA_BOUNDS)
    knee = bounds[:, 0] + torch.rand(30000, 2) * (bounds[:, 1] - bounds[:, 0])
    difference = torch.tensor([1., -1.]) * knee - torch.tensor(NOMINAL_DELTA)
    continuous = torch.stack((common - .5 * difference, common + .5 * difference), -1).flatten(-2)
    wrapped = wrap(continuous)
    rebased = .25 * rebase_final_leg_pairs(wrapped / .25)
    torch.testing.assert_close(wrap(rebased - continuous), torch.zeros_like(rebased), rtol=0, atol=1e-6)
    assert rebased.abs().max() < 4.
    nominal = torch.tensor([0., NOMINAL_DELTA[0], 0., NOMINAL_DELTA[1]])
    current = nominal + continuous
    actual = paired_leg_targets(nominal + rebased, current)
    expected = paired_leg_targets(nominal + continuous, current)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    old = paired_leg_targets(nominal + wrapped, current)
    assert (old - expected).abs().max() > 3.


def test_non_crossing_targets_are_bitwise_unchanged():
    torch.manual_seed(922)
    legs = torch.rand(4000, 4) * 4 - 2
    assert torch.equal(legs, rebase_final_leg_pairs(legs))


@pytest.mark.parametrize("actor_type", [UprightTargetLimitedGapActor, EntryPreservingGapActor])
def test_opt_in_rebases_only_final_targets_and_keeps_wheels_and_gradients(actor_type):
    torch.manual_seed(923)
    old = actor_type(nn.Linear(35, 6), upright_entry_recovery=True,
        canonical_recovery=True, smooth_knee_targets=True, cyclic_leg_targets=True,
        recovery_leg_step=.3, leg_action_clip=16.)
    new = deepcopy(old)
    new.final_target_winding_version = "paired-v1"
    obs = torch.randn(32, 35) * .1
    obs[:, 9] = -1.
    obs[:, 10:14] = wrap(torch.tensor([3., 3.4, -3., -3.4]))
    obs[:, 22:26] = obs[:, 10:14] / .25
    a, b = old(obs), new(obs)
    assert torch.equal(a[:, 4:], b[:, 4:])
    assert not torch.equal(a[:, :4], b[:, :4])
    torch.testing.assert_close(wrap(.25 * (b[:, :4] - a[:, :4])), torch.zeros_like(b[:, :4]), rtol=0, atol=1e-6)
    b.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)


@pytest.mark.parametrize("bad", [True, False, "", "paired-v2"])
def test_unknown_versions_are_rejected(bad):
    with pytest.raises(ValueError, match="final target winding"):
        UprightTargetLimitedGapActor(nn.Linear(35, 6), final_target_winding_version=bad)


def test_inadequate_bounds_are_rejected_instead_of_truncating_paired_targets():
    with pytest.raises(ValueError, match="bounds covering"):
        UprightTargetLimitedGapActor(nn.Linear(35, 6), cyclic_leg_targets=True,
            leg_action_clip=13., final_target_winding_version="paired-v1")

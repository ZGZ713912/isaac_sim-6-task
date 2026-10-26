"""Previous target winding must not invent a half-turn recovery command."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from wheeled_algo.paired_recovery import wrap
from wheeled_algo.recovery_coordinates import smooth_recovery_legs
from wheeled_algo.gap_entry_handoff import HandoffGapGetupActor
from wheeled_algo.upright_target_limit import UprightTargetLimitedGapActor


def request(previous, measured, version=None):
    return smooth_recovery_legs(torch.zeros_like(previous), previous / .25, measured,
                                .3, (.924544, .924537), previous_common_version=version) * .25


def test_previous_pair_straddling_pi_has_the_same_physical_target():
    continuous = torch.tensor([[3., 3.4, -3., -3.4]])
    wrapped = wrap(continuous)
    fixed = request(wrapped, wrapped, "circular-v1")
    reference = request(continuous, wrapped, "circular-v1")
    torch.testing.assert_close(wrap(fixed - reference), torch.zeros_like(fixed), rtol=0, atol=1e-6)
    assert wrap(request(wrapped, wrapped) - reference).abs().max() > .3


def test_normal_previous_pairs_keep_the_exact_old_arithmetic():
    torch.manual_seed(805)
    previous = torch.rand(256, 4) - .5
    assert torch.equal(request(previous, previous), request(previous, previous, "circular-v1"))


@pytest.mark.parametrize("actor_type", [HandoffGapGetupActor, UprightTargetLimitedGapActor])
def test_circular_recovery_changes_crossing_legs_and_retains_wheels(actor_type):
    torch.manual_seed(806)
    old = actor_type(nn.Linear(35, 6), smooth_knee_targets=True, recovery_leg_step=.3,
                     upright_entry_recovery=True, stance_aware_recovery=True)
    fixed = deepcopy(old)
    fixed.recovery_previous_common_version = "circular-v1"
    obs = torch.zeros(4, 35)
    obs[:, 9] = torch.tensor([-1., -.6, .5, 1.])
    obs[:, 10:14] = wrap(torch.tensor([3., 3.4, -3., -3.4]))
    obs[:, 22:26] = obs[:, 10:14] / .25
    a, b = old(obs), fixed(obs)
    assert not torch.equal(a[:, :4], b[:, :4])
    assert torch.equal(a[:, 4:], b[:, 4:])
    b.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in fixed.parameters() if p.grad is not None)


@pytest.mark.parametrize("bad", [True, False, "", "circular-v2"])
def test_unknown_coordinate_versions_are_rejected(bad):
    with pytest.raises(ValueError, match="previous recovery"):
        HandoffGapGetupActor(nn.Linear(35, 6), recovery_previous_common_version=bad)

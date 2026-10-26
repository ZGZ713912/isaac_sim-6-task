from copy import deepcopy

import pytest
import torch
from torch import nn

from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.gap_entry_preservation import EntryPreservingGapActor
from wheeled_algo.gap_recovery_exploration import GapRecoveryNoiseMean
from wheeled_algo.upright_target_limit import UprightTargetLimitedGapActor
from wheeled_algo.unified_getup import UnifiedGetupActor


def pair(seed):
    torch.manual_seed(seed)
    kwargs = dict(upright_entry_recovery=True, canonical_recovery=True, smooth_knee_targets=True,
                  recovery_leg_step=.3, stance_aware_recovery=True, recovery_shape_aware=True,
                  quiet_height_experts=True, quiet_native_height_anchors=True,
                  quiet_velocity_feedback=True, quiet_velocity_feedback_mode="standing_displacement",
                  quiet_output_wheel_alpha=.32)
    old = UprightTargetLimitedGapActor(nn.Linear(35, 6), **kwargs)
    new = EntryPreservingGapActor(deepcopy(old.parent), **kwargs)
    absent = new.load_state_dict(old.state_dict(), strict=False)
    assert not absent.unexpected_keys and all(k.startswith("entry_recovery.") for k in absent.missing_keys)
    new.entry_recovery.load_state_dict(old.recovery.state_dict(), strict=True)
    return old, new


@pytest.mark.parametrize("seed", [811, 812, 813])
def test_neutral_entry_copy_keeps_the_full_mean_bitwise_exact(seed):
    old, new = pair(seed)
    obs = torch.randn(512, 35)
    obs[:, 3] = (torch.tensor([.23, .305, .42]) * 5.).repeat(171)[:512]
    obs[:, 7:10] /= obs[:, 7:10].norm(dim=-1, keepdim=True)
    with torch.no_grad():
        assert torch.equal(old(obs), new(obs))
    assert sum(p.numel() for p in new.entry_recovery.parameters()) == 21894
    assert not any(p.requires_grad for p in new.entry_recovery.parameters())
    assert [p.shape for p in new.parameters() if p.requires_grad] == [p.shape for p in old.parameters() if p.requires_grad]


def test_fallen_updates_cannot_modify_the_full_entry_authority_function():
    old, new = pair(814)
    obs = torch.zeros(6, 35)
    obs[:, 3] = .305 * 5.
    obs[:, 9] = -1.
    obs[:3, 10:14] = torch.tensor([1., 1., -1., -1.])
    obs[3:, 7:10] = torch.tensor([.1, .3, .9])
    obs[3:, 10:14] = torch.tensor([-.8, .2, .3, -.9])
    obs[:, 22:26] = obs[:, 10:14] / .25
    gate = handoff_entry_gate(obs)[:, 0]
    assert torch.equal(gate, torch.tensor([1., 1., 1., 0., 0., 0.]))
    with torch.no_grad():
        new.recovery[-1].bias.add_(torch.tensor([.03, -.01, .04, .02, .03, -.01]))
        # The preserved legacy blend can incur one rounding error in x+(y-x).
        torch.testing.assert_close(old(obs)[:3], new(obs)[:3], atol=1e-6, rtol=0.)
        assert not torch.equal(old(obs)[3:], new(obs)[3:])
        # With no entry authority, the original fallen function is exact.
        assert torch.equal(UprightTargetLimitedGapActor.forward(new, obs)[3:], new(obs)[3:])
    dist = GapRecoveryNoiseMean(new, .06, .35, .15)
    dist(obs).square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in new.recovery.parameters())
    assert all(p.grad is None for p in new.entry_recovery.parameters())


def test_unified_initializer_handles_only_the_declared_frozen_addition():
    old, new = pair(815)
    unified = UnifiedGetupActor(deepcopy(old.parent))
    new.initialize_from_unified(unified.state_dict())
    assert all(torch.equal(v, new.entry_recovery.state_dict()[k]) for k, v in new.recovery.state_dict().items())
    assert not any(p.requires_grad for p in new.entry_recovery.parameters())


def test_disabled_entry_is_rejected():
    with pytest.raises(ValueError, match="enabled orbit-v2"):
        EntryPreservingGapActor(nn.Linear(35, 6), upright_entry_recovery=False)

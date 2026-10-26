"""The orbit-v2 noise pilot must retain mean, entry and native likelihood."""
from copy import deepcopy
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from wheeled_algo.gap_entry_handoff import HandoffGapConditionedMean, HandoffGapGetupActor
from wheeled_algo.gap_recovery_exploration import GapRecoveryNoiseMean, SampledGapPolicyEvaluation


def setup():
    torch.manual_seed(707)
    mean = HandoffGapGetupActor(nn.Linear(35, 6), upright_entry_recovery=True,
        canonical_recovery=True, smooth_knee_targets=True, recovery_leg_step=.3,
        stance_aware_recovery=True, recovery_shape_aware=True)
    settings = dict(gap_entry_handoff_version="orbit-v2", quiet_exploration_fraction=.06,
                    quiet_wheel_exploration_fraction=.35)
    obs = torch.zeros(4, 35)
    obs[:, 3] = .305 * 5.
    obs[:, 9] = -1.
    obs[1, 0] = 1.
    obs[2, 9] = 1.
    obs[3, 10:14] = obs.new_tensor([1., 1., -1., -1.])
    obs[:, 22:26] = obs[:, 10:14] / .25
    return mean, settings, obs


def test_fraction_one_is_the_exact_existing_handoff_distribution():
    mean, settings, obs = setup()
    old = HandoffGapConditionedMean(mean, .06, .35)
    new = GapRecoveryNoiseMean(deepcopy(mean), .06, .35, 1.)
    new.load_state_dict(old.state_dict(), strict=True)
    assert torch.equal(old(obs), new(obs))


def test_only_fallen_leg_std_changes_and_the_native_distribution_uses_it():
    from rsl_rl.modules import HeteroscedasticGaussianDistribution
    mean, settings, obs = setup()
    old = HandoffGapConditionedMean(mean, .06, .35)
    new = GapRecoveryNoiseMean(deepcopy(mean), .06, .35, .15)
    new.load_state_dict(old.state_dict(), strict=True)
    a, b = old(obs), new(obs)
    assert torch.equal(a[:, 0], b[:, 0])
    assert torch.equal(a[:, 1, 4:], b[:, 1, 4:])
    assert torch.equal(a[[0, 1, 3], 1], b[[0, 1, 3], 1])
    torch.testing.assert_close((b[2, 1, :4] - a[2, 1, :4]).exp(), obs.new_full((4,), .15))
    native = HeteroscedasticGaussianDistribution(6, std_type="log", std_range=(.003, .8))
    native.update(b)
    assert torch.equal(native.mean, b[:, 0])
    assert torch.equal(native.std, b[:, 1].clamp(math.log(.003), math.log(.8)).exp())
    sample = native.sample()
    log_prob = torch.distributions.Normal(native.mean, native.std).log_prob(sample).sum(-1)
    (-log_prob.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)


def test_private_common_draws_preserve_rng_and_wheel_samples(monkeypatch):
    mean, settings, obs = setup()
    old = SampledGapPolicyEvaluation(mean, settings, torch.full((6,), -.57), 708)
    new = SampledGapPolicyEvaluation(mean, dict(settings, gap_recovery_leg_exploration_fraction=.15),
                                     torch.full((6,), -.57), 708)
    rng = torch.get_rng_state()
    for _ in range(8):
        a, b = old(obs), new(obs)
        assert torch.equal(a[:, 4:], b[:, 4:])
        assert torch.equal(a[[0, 1, 3]], b[[0, 1, 3]])
    assert torch.equal(rng, torch.get_rng_state())
    assert old.sampling_receipt()["generator_state_sha256"] == new.sampling_receipt()["generator_state_sha256"]
    with pytest.raises(ValueError, match="batch size"):
        old(obs[:1])
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from export_robust_getup import verify_deterministic_evidence
    with pytest.raises(ValueError, match="Mean qualification"):
        verify_deterministic_evidence(dict(sampled_policy_exposure_diagnostic=new.sampling_receipt()))


@pytest.mark.parametrize("value", [.099, 1.01, True, float("nan"), float("inf")])
def test_invalid_exploration_fraction_is_rejected(value):
    mean, settings, obs = setup()
    with pytest.raises(ValueError, match="noise fraction"):
        GapRecoveryNoiseMean(mean, .06, .35, value)

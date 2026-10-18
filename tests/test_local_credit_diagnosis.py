from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('local_credit_probe',
    Path(__file__).parents[1] / 'scripts/diagnose_local_credit.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_credit_population_excludes_conditioning_and_reports_noise_score():
    a = torch.tensor([1., float('nan'), -1.])
    r = torch.tensor([2., float('nan'), -4.])
    n = torch.tensor([[1., 2.], [float('nan'), float('nan')], [-1., 0.]])
    mask = torch.tensor([True, False, True])
    rng = torch.get_rng_state().clone()
    result = probe.credit_statistics(a, r, n, mask)
    assert result['samples'] == 2
    assert result['advantage_mean'] == 0.
    assert result['advantage_std'] == 1.
    assert result['advantage_positive_fraction'] == .5
    assert result['residual_mean'] == -1.
    assert result['residual_rms'] == pytest.approx(10 ** .5)
    assert result['advantage_times_noise_mean'] == [1., 1.]
    assert result['advantage_times_noise_square_minus_one_mean'] == [0., 2.]
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match='Nonfinite'):
        probe.credit_statistics(a, r, n, torch.ones(3, dtype=torch.bool))


def test_empty_population_has_explicit_zero_samples_and_rejects_bad_masks():
    a = torch.tensor([float('nan')])
    result = probe.credit_statistics(a, a, a[:, None], torch.zeros(1, dtype=torch.bool))
    assert result['samples'] == 0
    assert all(value == 0 for key, value in result.items() if not isinstance(value, list))
    with pytest.raises(ValueError, match='boolean'):
        probe.credit_statistics(a, a, a[:, None], torch.ones(1))


def test_event_credit_respects_script_handoffs_done_and_bootstrap_without_mutation():
    events = torch.tensor([[float('nan'), 0.], [0., -4.], [-4., 0.], [0., 0.]])
    learning = torch.tensor([[False, True], [True, True], [True, True], [True, True]])
    bootstrap = torch.ones_like(learning)
    dones = torch.tensor([[False, False], [False, True], [True, False], [False, False]])
    original = events.clone()
    result = probe.event_advantage(events, learning, bootstrap, dones, .9, .5)
    assert torch.allclose(result, torch.tensor([[0., -1.8], [-1.8, -4.], [-4., 0.], [0., 0.]]))
    assert torch.allclose(events, original, equal_nan=True)
    bootstrap[1, 0] = False
    assert probe.event_advantage(events, learning, bootstrap, dones, .9, .5)[1, 0] == 0
    with pytest.raises(ValueError, match='Nonfinite'):
        probe.event_advantage(events, torch.ones_like(learning), bootstrap, dones, .9, .5)
    with pytest.raises(ValueError, match='aligned'):
        probe.event_advantage(events, learning.float(), bootstrap, dones, .9, .5)


def test_reading_native_masked_gae_preserves_models_adam_storage_and_rng():
    from test_recovery_rollout_mask import algorithm, collect
    from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask
    alg = algorithm()
    adapter = attach_recovery_rollout_mask(alg, num_envs=2)
    mask = torch.tensor([[1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 1]], dtype=torch.bool)
    obs = collect(alg, mask)
    alg.compute_returns(obs)
    before = deepcopy(alg.save())
    arrays = {k: getattr(alg.storage, k).clone() for k in ('advantages', 'returns', 'values', 'actions')}
    rng = torch.get_rng_state().clone()
    statistics = probe.credit_statistics(alg.storage.advantages.squeeze(-1),
        (alg.storage.returns - alg.storage.values).squeeze(-1), alg.storage.actions,
        adapter.learning)
    assert statistics['samples'] == 8
    assert probe.same_tree(before, alg.save())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, getattr(alg.storage, key)) for key, value in arrays.items())

"""Native score-gradient identity and observation-only storage extraction."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch

from test_recovery_rollout_mask import algorithm, collect
from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = script('diagnose_local_credit')
analysis = script('analyze_local_policy_credit')


def native_buffer():
    alg = algorithm()
    adapter = attach_recovery_rollout_mask(alg, num_envs=2)
    mask = torch.tensor([[1, 1], [0, 1], [0, 1], [1, 1], [1, 1], [1, 1]], dtype=torch.bool)
    obs = collect(alg, mask)
    alg.compute_returns(obs)
    return alg, adapter


def test_snapshot_preserves_native_models_optimizer_buffers_rng_and_is_independent():
    alg, _ = native_buffer()
    before = deepcopy(alg.save())
    rng = torch.get_rng_state().clone()
    values = probe.policy_credit_snapshot(alg.storage, 'actor')
    assert torch.equal(values['policy_input'], alg.storage.observations['actor'])
    assert torch.equal(values['native_action'], alg.storage.actions)
    assert torch.equal(values['native_log_prob'], alg.storage.actions_log_prob.squeeze(-1))
    assert probe.same_tree(before, alg.save())
    assert torch.equal(rng, torch.get_rng_state())
    for value in values.values():
        value.zero_()
    assert bool(alg.storage.distribution_params[1].gt(0).all())
    assert bool(alg.storage.actions.abs().sum() > 0)
    assert bool(alg.storage.observations['actor'].abs().sum() > 0)


def test_snapshot_rejects_partial_or_invalid_native_distribution_buffers():
    alg, _ = native_buffer()
    alg.storage.step -= 1
    with pytest.raises(ValueError, match='complete native'):
        probe.policy_credit_snapshot(alg.storage, 'actor')
    alg.storage.step += 1
    alg.storage.distribution_params[1][0, 0, 0] = 0.
    with pytest.raises(ValueError, match='nonpositive'):
        probe.policy_credit_snapshot(alg.storage, 'actor')


def test_command_labels_are_exhaustive_disjoint_and_exclude_conditioning():
    commands = torch.tensor([[0., 0., .23], [0., 0., .305], [0., 0., .43],
        [.5, 0., .305], [-.5, 0., .305], [0., 1., .305], [0., -1., .305],
        [.1, .1, .305], [float('nan'), float('nan'), float('nan')]])
    eligible = torch.tensor([True]*8 + [False])
    masks = analysis.command_buckets(commands, eligible)
    assert all(int(mask.sum()) == 1 for mask in masks.values())
    assert torch.equal(torch.stack(list(masks.values())).sum(0), eligible.int())
    with pytest.raises(ValueError, match='Nonfinite'):
        analysis.command_buckets(commands, torch.ones(9, dtype=torch.bool))
    with pytest.raises(ValueError, match='aligned'):
        analysis.command_buckets(commands, eligible.float())


def test_readonly_score_gradient_matches_native_old_policy_surrogate_including_std():
    from tensordict import TensorDict
    alg, adapter = native_buffer()
    values = probe.policy_credit_snapshot(alg.storage, 'actor')
    m = adapter.learning
    inputs, actions = values['policy_input'][m], values['native_action'][m]
    advantage = alg.storage.advantages.squeeze(-1)[m]
    model = deepcopy(alg.actor)
    # The diagnostic accepts a policy observation key; the fixture model uses actor.
    model.obs_groups = ['policy']
    rng = torch.get_rng_state().clone()
    before = deepcopy(model.state_dict())
    actual = analysis.score_gradient(model, inputs, actions, advantage, len(inputs), chunk=3)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())
    # Reference the installed native stochastic forward/log_prob and ratio-clipped
    # expression. The old log-prob is detached at precisely the same CPU policy.
    model(TensorDict({'policy': inputs}, batch_size=[len(inputs)]), stochastic_output=True)
    logp = model.get_output_log_prob(actions)
    ratio = (logp-logp.detach()).exp()
    loss = torch.maximum(-advantage*ratio, -advantage*ratio.clamp(.8,1.2)).mean()
    expected = torch.cat([v.flatten().double() for v in torch.autograd.grad(loss, tuple(model.parameters()))])
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    assert actual.abs().sum() > 0 and all(p.grad is None for p in model.parameters())
    # Splitting only the saved advantage into centered and baseline components
    # must reconstruct the native gradient without modifying a training buffer.
    avg = advantage.mean()
    within = analysis.score_gradient(model, inputs, actions, advantage-avg, len(inputs), chunk=3)
    base = analysis.score_gradient(model, inputs, actions, torch.full_like(advantage,avg), len(inputs), chunk=3)
    torch.testing.assert_close(within+base, actual, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize('invalid', ['nonfinite', 'length', 'denominator'])
def test_score_gradient_rejects_bad_owned_samples(invalid):
    alg, adapter = native_buffer()
    model = deepcopy(alg.actor)
    model.obs_groups = ['policy']
    m = adapter.learning
    inputs = alg.storage.observations['actor'][m].clone()
    actions = alg.storage.actions[m].clone()
    advantage = alg.storage.advantages.squeeze(-1)[m].clone()
    denominator = len(inputs)
    if invalid == 'nonfinite':
        advantage[0] = float('nan')
    elif invalid == 'length':
        actions = actions[:-1]
    else:
        denominator = 0
    with pytest.raises(ValueError):
        analysis.score_gradient(model, inputs, actions, advantage, denominator)

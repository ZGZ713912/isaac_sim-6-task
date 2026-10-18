"""Native loss/state semantics and fail-closed value-clipping contract checks."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis import local_scratch as local
from wheeled_tasks.chassis.local_value_revision import value_clipping_setting, revision_binding

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_absence_and_exact_named_boolean_only():
    assert value_clipping_setting({}) is True
    assert value_clipping_setting({'local_scratch': {'plan_file': local.RETURN_PLAN_FILE}}) is True
    assert value_clipping_setting({'use_clipped_value_loss': False,
        'local_scratch': {'plan_file': local.VALUE_PLAN_FILE}}) is False


@pytest.mark.parametrize('value', [True, 0, 1, 0., None, 'false', [], {}])
def test_false_is_not_a_general_truthy_or_numeric_override(value):
    with pytest.raises(ValueError, match='named local v9 boolean False'):
        value_clipping_setting({'use_clipped_value_loss': value,
            'local_scratch': {'plan_file': local.VALUE_PLAN_FILE}})


@pytest.mark.parametrize('marker', [None, {}, {'plan_file': local.RETURN_PLAN_FILE},
                                  {'plan_file': local.ENTROPY_AB_PLAN_FILE}])
def test_unrelated_contract_cannot_override_native_value_loss(marker):
    with pytest.raises(ValueError):
        value_clipping_setting({'use_clipped_value_loss': False, 'local_scratch': marker})


def test_no_control_receipt_cannot_generate_a_target_plan(tmp_path, monkeypatch):
    # Real source/decision hashes are checked first; only the control receipt
    # lookup is redirected to a deliberately absent location in this negative.
    from wheeled_tasks.chassis import local_value_revision as revision
    monkeypatch.setattr(revision, 'CONSUMPTION_FILE', str(tmp_path / 'absent.json'))
    with pytest.raises(ValueError, match='actual completed/evaluated control'):
        revision_binding(ROOT)


@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('per_batch', [False, True])
def test_native_value_toggle_preserves_learned_state_actor_sampling_and_adam_clock(masked, per_batch):
    from test_recovery_rollout_mask import algorithm, collect
    from test_native_ppo_audit import assert_tree_equal
    from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask
    parent = algorithm(steps=384, per_batch=per_batch, normalize_obs=True)
    parent.schedule, parent.learning_rate = 'fixed', 1e-4
    for group in parent.optimizer.param_groups:
        group['lr'] = 1e-4
    attach_recovery_rollout_mask(parent, num_envs=2)
    learning = torch.ones(384, 2, dtype=torch.bool)
    obs = collect(parent, learning)
    parent.compute_returns(obs)
    parent.update()
    saved = deepcopy(parent.save())
    if masked:
        learning[:20, 0] = False
        learning[140:160, 1] = False
    arms, arrays = [], []
    for clipped in (True, False):
        alg = algorithm(steps=384, per_batch=per_batch, normalize_obs=True)
        alg.schedule, alg.learning_rate = 'fixed', 1e-4
        alg.value_loss_coef, alg.entropy_coef, alg.max_grad_norm = 4., .0005, 1.
        alg.gamma, alg.lam = .9974905699336811, 1.
        alg.use_clipped_value_loss = clipped
        alg.num_learning_epochs, alg.num_mini_batches = 5, 4
        assert alg.load(deepcopy(saved), None, strict=True)
        assert alg.use_clipped_value_loss is clipped
        assert_tree_equal(alg.save(), saved)
        attach_recovery_rollout_mask(alg, num_envs=2)
        obs = collect(alg, learning)
        alg.compute_returns(obs)
        arrays.append(tuple(getattr(alg.storage, key).clone()
            for key in ('actions', 'rewards', 'returns', 'advantages', 'values')))
        # A synthetic lagging old-value fixture exercises the native flat
        # clipped branch. It is not a claim about a real on-policy rollout.
        alg.storage.values -= 1.
        alg.storage.returns += 20.
        torch.manual_seed(901)
        losses = alg.update()
        arms.append((deepcopy(alg.save()), torch.get_rng_state().clone(), losses))
        assert alg.storage.step == 0
        for parameter, state in alg.optimizer.state_dict()['state'].items():
            assert state['step'] == saved['optimizer_state_dict']['state'][parameter]['step'] + 20
            assert all(torch.isfinite(tensor).all() for tensor in state.values())
    assert_tree_equal(arrays[0], arrays[1])
    assert_tree_equal(arms[0][0]['actor_state_dict'], arms[1][0]['actor_state_dict'])
    assert torch.equal(arms[0][1], arms[1][1])
    assert arms[0][2]['surrogate'] == arms[1][2]['surrogate']
    assert any(not torch.equal(tensor, arms[1][0]['critic_state_dict'][key])
        for key, tensor in arms[0][0]['critic_state_dict'].items())
    actor_count = len(list(parent.actor.parameters()))
    for parameter in range(actor_count):
        assert_tree_equal(arms[0][0]['optimizer_state_dict']['state'][parameter],
                          arms[1][0]['optimizer_state_dict']['state'][parameter])


def test_native_clipped_flat_gradient_and_unclipped_target_gradient():
    # Analytic one-sample values make the independent derivative explicit:
    # current1, old0, target2 => clipped branch (0.2-2)^2 and dL/dV=0;
    # unclipped (1-2)^2 has dL/dV=-2. Compare native update's first gradients.
    from test_recovery_rollout_mask import algorithm, collect
    captured = []
    for clipped in (True, False):
        alg = algorithm(steps=6)
        alg.use_clipped_value_loss = clipped
        alg.num_learning_epochs = alg.num_mini_batches = 1
        for parameter in alg.critic.parameters():
            parameter.data.zero_()
        bias = list(alg.critic.parameters())[-1]
        bias.data.fill_(1.)
        obs = collect(alg, torch.ones(6, 2, dtype=torch.bool))
        alg.compute_returns(obs)
        alg.storage.values.zero_()
        alg.storage.returns.fill_(2.)
        values = []
        hook = bias.register_hook(lambda grad: values.append(grad.detach().clone()))
        try:
            losses = alg.update()
        finally:
            hook.remove()
        captured.append((values[0].item(), losses['value']))
    assert captured[0] == pytest.approx((0., 3.24))
    assert captured[1] == pytest.approx((-2., 1.))

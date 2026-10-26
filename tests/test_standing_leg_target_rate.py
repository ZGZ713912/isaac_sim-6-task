from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.standing_leg_target_rate import contract, density, install, validate_contract


def test_physical_units_small_deltas_and_all_four_leg_axes_are_charged():
    delta = torch.eye(4) * .001
    torch.testing.assert_close(density(delta), torch.full((4,), torch.log(torch.tensor(2.)).item() / 4))
    assert density(torch.full((1, 4), .00001)).item() > 0.
    assert density(torch.zeros(1, 4)).item() == 0.
    torch.testing.assert_close(density(delta), density(-delta))


@pytest.mark.parametrize('value', [True, -.1, 17., float('nan'), float('inf')])
def test_invalid_contract_weight_is_rejected(value):
    with pytest.raises(ValueError):
        contract(value)


def test_contract_rejects_changed_units_and_source():
    value = contract(.2)
    validate_contract(value, value['module_sha256'])
    changed = deepcopy(value); changed['target_delta_scale_rad'] = .01
    with pytest.raises(ValueError):
        validate_contract(changed, value['module_sha256'])
    with pytest.raises(ValueError):
        validate_contract(value, {})


class Env:
    def __init__(self, reset_inside=False):
        self.precise_standing_reward = {}
        self.policy_dt, self.fixed_cases = .005, None
        self.v5 = SimpleNamespace(LEGS=(0, 1, 3, 4))
        self.actions = torch.zeros(3, 6)
        self.target = torch.zeros(3, 6)
        self.reward = torch.ones(3)
        self.mask = torch.tensor([True, True, False])
        self.commands = torch.zeros(3, 3); self.commands[1, 0] = .5
        self.reset_inside = reset_inside
        self.calls = 0

    def step(self, action):
        self.calls += 1
        self.actions.copy_(self.target)
        diagnostics = dict(actions=self.actions.clone(), requested_command=self.commands.clone())
        if self.reset_inside:
            self.actions.zero_()
        return {}, self.reward, torch.zeros(3, dtype=torch.bool), dict(
            diagnostics=diagnostics, policy_learning_mask=self.mask.clone(), log={})


def test_actual_applied_leg_targets_cost_only_eligible_standing_and_do_not_write_actions():
    env = Env(); env.target[:, [0, 1, 3, 4]] = .004
    expected = env.target.clone()
    install(env, .2)
    _, reward, _, extras = env.step(torch.full((3, 6), 99.))
    paid = .005 * .2 * torch.log(torch.tensor(2.))
    torch.testing.assert_close(reward, torch.tensor([1. - paid, 1., 1.]))
    assert torch.equal(env.actions, expected) and env.calls == 1
    assert env.standing_leg_target_rate_last['eligible'].tolist() == [True, False, False]
    assert 'LegTargetRate/paid_cost_per_policy_step' in extras['log']
    # A held target has zero delta on the next policy tick.
    assert env.step(None)[1] is not env.reward
    assert not env.standing_leg_target_rate_last['paid'].any()


def test_autoreset_uses_pre_reset_diagnostics_and_never_bridges_episodes():
    env = Env(reset_inside=True); env.target[:, [0, 1, 3, 4]] = .004
    install(env, .2)
    rewards = [env.step(None)[1] for _ in range(2)]
    torch.testing.assert_close(rewards[0], rewards[1])
    assert rewards[0][0] < 1.
    assert not env.actions.any()


def test_wheels_are_excluded_zero_weight_is_reward_identity_and_fixed_eval_is_unchanged():
    env = Env(); env.target[:, [2, 5]] = 10.
    install(env, .2)
    env.step(None)
    assert not env.standing_leg_target_rate_last['paid'].any()
    env = Env(); install(env, 0.)
    assert env.step(None)[1] is env.reward
    env.fixed_cases = [{}]; env.target.fill_(.5)
    _, reward, _, extras = env.step(None)
    assert reward is env.reward and not extras['log']


def test_installation_requires_precise_200hz_and_rejects_duplicates():
    env = Env(); install(env, .2)
    with pytest.raises(ValueError):
        install(env, .2)
    env = Env(); env.policy_dt = .02
    with pytest.raises(ValueError):
        install(env, .2)


@pytest.mark.parametrize('delta', [torch.zeros(1, 6), torch.full((1, 4), float('nan'))])
def test_invalid_target_samples_are_rejected(delta):
    with pytest.raises(ValueError):
        density(delta)

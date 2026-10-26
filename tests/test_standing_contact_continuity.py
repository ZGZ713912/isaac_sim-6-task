from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.standing_contact_continuity import break_counts, contract, install, validate_contract


def test_substep_breaks_count_events_and_preserve_boundary_memory():
    present = torch.tensor([[False, False, True], [False, False, False], [True, False, True],
                            [False, False, False], [True, False, True]])
    assert break_counts(torch.tensor([True, False, True]), present).tolist() == [2, 0, 2]
    assert break_counts(torch.tensor([False, True]), torch.tensor([[False, True]])).tolist() == [0, 0]


def test_invalid_units_or_changed_provenance_are_rejected():
    for value in (True, -.01, .21, float('nan')):
        with pytest.raises(ValueError):
            contract(value)
    c = contract(.05)
    validate_contract(c, c['module_sha256'])
    changed = deepcopy(c)
    changed['startup_grace_s'] = .2
    with pytest.raises(ValueError):
        validate_contract(changed, c['module_sha256'])
    with pytest.raises(ValueError):
        validate_contract(c, {})


class ExampleEnv:
    def __init__(self, reset_inside=False):
        self.num_envs, self.device = 1, 'cpu'
        self.fixed_cases = None
        self.standing_contact_training_contract = {}
        self.grounded_standing_reset = {}
        self.recovery_lane = self.passive = torch.zeros(1, dtype=torch.bool)
        self.pending_command = torch.zeros(1, 3)
        self.standing_contact_monitor = SimpleNamespace(last=None)
        self.reward = torch.ones(1)
        self.reset_inside = reset_inside
        self.present = torch.zeros(5, 1, dtype=torch.bool)

    def reset(self, ids):
        pass

    def step(self, action):
        force = torch.zeros(5, 1, 2, 3)
        force[..., 2] = self.present[:, :, None] * 70.
        self.standing_contact_monitor.last = {'force': force}
        if self.reset_inside:
            self.reset(torch.tensor([0]))
        return {}, self.reward, torch.tensor([self.reset_inside]), {'policy_learning_mask': torch.ones(1, dtype=torch.bool), 'log': {}}


def test_reset_during_native_step_does_not_merge_support_events_across_episodes():
    env = ExampleEnv(reset_inside=True)
    install(env, .05)
    for _ in range(2):
        _, reward, _, _ = env.step(None)
        torch.testing.assert_close(reward, torch.tensor([.95]))
        assert env.standing_contact_continuity_last['break_count'].item() == 1
        assert not env.standing_contact_continuity_last['same_episode'].item()


def test_continued_outage_is_dense_cost_only_and_zero_event_cost_retains_reward_identity():
    env = ExampleEnv()
    install(env, 0.)
    assert env.step(None)[1] is env.reward
    assert env.standing_contact_continuity_last['break_count'].item() == 1
    assert env.step(None)[1] is env.reward
    assert env.standing_contact_continuity_last['break_count'].item() == 0


def test_physical_substep_objective_supports_each_first_ms_break():
    env = ExampleEnv()
    del env.standing_contact_training_contract
    env.physical_standing_reward = {}
    env.present[:, 0] = torch.tensor([False, True, False, True, True])
    install(env, .1)
    _, reward, _, extras = env.step(None)
    torch.testing.assert_close(reward, torch.tensor([.8]))
    assert env.standing_contact_continuity_last['break_count'].item() == 2
    assert extras['log']['StandingContinuity/reward_cost_per_break'] == .1


def test_fixed_evaluation_reward_and_raw_contact_remain_native():
    env = ExampleEnv()
    install(env, .1)
    env.fixed_cases = [{}]
    _, reward, _, extras = env.step(None)
    assert reward is env.reward
    assert not env.standing_contact_monitor.last['force'].any()
    assert not extras['log']

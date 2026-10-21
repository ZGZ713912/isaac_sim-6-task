import torch
from wheeled_tasks.chassis.getup_reward import potential_reward


def test_a_crash_cannot_keep_more_potential_reward_by_first_throwing_body_up():
    gamma, gain = .998, 5.
    paths = torch.tensor([[.2, .4, 1.8, 1.9], [.2, .1, .05, .01]])
    total = torch.zeros(2)
    for tick in range(3):
        terminal = torch.full((2,), tick == 2)
        total += gamma ** tick * potential_reward(paths[:, tick], paths[:, tick + 1], terminal, gamma, gain)
    torch.testing.assert_close(total, torch.full((2,), -gain * .2), atol=1e-6, rtol=1e-5)


def test_nonterminal_time_limit_keeps_bootstrap_potential():
    gamma, gain = .998, 5.
    actual = potential_reward(torch.tensor([1.]), torch.tensor([1.5]), torch.tensor([False]), gamma, gain)
    torch.testing.assert_close(actual, torch.tensor([gain * (gamma * 1.5 - 1.)]))

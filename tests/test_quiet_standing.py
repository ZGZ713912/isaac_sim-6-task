import math

import torch

from wheeled_tasks.chassis.quiet_standing import QuietStanding


def test_slow_sway_inside_five_cm_has_no_position_cost():
    state = QuietStanding(1, "cpu", .005)
    active = torch.ones(1, dtype=torch.bool)
    zeros = torch.zeros(1, 6)
    state.update(active, torch.zeros(1, 2), zeros, zeros)
    cost, _, _, radius = state.update(active, torch.tensor([[.04, 0.]]), zeros, zeros)
    assert cost.item() == 0.
    torch.testing.assert_close(radius, torch.tensor([.04]))
    cost, _, _, _ = state.update(active, torch.tensor([[.06, 0.]]), zeros, zeros)
    torch.testing.assert_close(cost, torch.tensor([.02]))


def test_equal_amplitude_fast_targets_cost_more_than_slow_targets():
    state = QuietStanding(2, "cpu", .005)
    active = torch.ones(2, dtype=torch.bool)
    power = []
    for tick in range(800):
        target = torch.zeros(2, 6)
        target[:, 2] = torch.sin(torch.tensor([.3, 20.]) * (2 * math.pi * tick * .005))
        _, p, _, _ = state.update(active, torch.zeros(2, 2), target, torch.zeros_like(target))
        if tick > 400:
            power.append(p)
    average = torch.stack(power).mean(0)
    assert average[1] > 30 * average[0]


def test_shared_target_winding_does_not_create_a_vibration_cost():
    state = QuietStanding(1, "cpu", .005)
    active = torch.ones(1, dtype=torch.bool)
    target = torch.zeros(1, 6)
    target[:, :2] = math.pi - .001
    state.update(active, torch.zeros(1, 2), target, torch.zeros_like(target))
    target[:, :2] = -math.pi + .001
    _, power, _, _ = state.update(active, torch.zeros(1, 2), target, torch.zeros_like(target))
    assert power.item() < .2


def test_reset_reanchors_only_selected_environment():
    state = QuietStanding(2, "cpu", .005)
    active = torch.ones(2, dtype=torch.bool)
    zeros = torch.zeros(2, 6)
    state.update(active, torch.zeros(2, 2), zeros, zeros)
    state.reset(torch.tensor([1]))
    state.update(active, torch.ones(2, 2) * .04, zeros, zeros)
    torch.testing.assert_close(state.displacement(torch.ones(2, 2) * .04), torch.tensor([[.04, .04], [0., 0.]]))

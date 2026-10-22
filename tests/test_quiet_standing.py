import math

import torch

from wheeled_tasks.chassis.quiet_standing import QuietStanding, standing_request_mask, standing_support_cost, standing_position_cost


def test_slow_sway_inside_five_cm_has_no_position_cost():
    state = QuietStanding(1, "cpu", .005)
    active = torch.ones(1, dtype=torch.bool)
    zeros = torch.zeros(1, 6)
    state.update(active, torch.zeros(1, 2), zeros, zeros)
    cost, _, _, radius = state.update(active, torch.tensor([[.04, 0.]]), zeros, zeros)
    assert cost.item() == 0.
    torch.testing.assert_close(radius, torch.tensor([.04]))
    cost, _, _, _ = state.update(active, torch.tensor([[.06, 0.]]), zeros, zeros)
    torch.testing.assert_close(cost, torch.tensor([.5 * math.log1p(.04)]))


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


def test_standing_anchor_survives_disturbance_but_excludes_initial_fallen_birth():
    commands = torch.zeros(5, 3)
    commands[4, 0] = .5
    passive = torch.tensor([False, False, True, False, False])
    recovery_birth = torch.tensor([False, True, True, True, False])
    recovered = torch.tensor([False, False, False, True, False])
    mask = standing_request_mask(commands, passive, recovery_birth, recovered)
    assert mask.tolist() == [True, False, False, True, False]
    state = QuietStanding(5, "cpu", .005, position_weight=2.)
    zeros = torch.zeros(5, 6)
    state.update(mask, torch.zeros(5, 2), zeros, zeros)
    # The request stays active even when the body has tilted; the 8 cm error
    # must remain visible instead of being erased by a new standing anchor.
    state.update(mask, torch.full((5, 2), .04), zeros, zeros)
    cost, _, _, radius = state.update(mask, torch.tensor([[.08, 0.]]).expand(5, -1), zeros, zeros)
    assert radius[0] > .079
    assert cost[0] > 0.


def test_support_cost_distinguishes_loaded_wheels_unloading_and_geometric_flight():
    clearance = torch.tensor([[0., 0.], [.001, .001], [.04, .04]])
    support = torch.tensor([[1., 1.], [.5, .5], [0., 0.]])
    cost = standing_support_cost(clearance, support)
    assert cost[0] == 0.
    assert 0. < cost[1] < cost[2]
    assert cost[2] > 5.


def test_position_envelope_has_no_price_within_five_cm_and_penalizes_both_directions():
    xy = torch.tensor([[.04, 0.], [.05, 0.], [.068, 0.], [-.068, 0.]])
    cost = standing_position_cost(xy.norm(dim=-1))
    assert cost[:2].tolist() == [0., 0.]
    assert cost[2] == cost[3] and cost[2] > 0.

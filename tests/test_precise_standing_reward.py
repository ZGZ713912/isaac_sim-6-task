import math
from types import SimpleNamespace
import pytest
import torch

from wheeled_tasks.chassis.precise_standing_reward import scores


def state():
    height = torch.tensor([.305])
    return height, torch.tensor([[0., 0., -1.]]), torch.zeros(1, 3), torch.zeros(1, 3), height.clone()


def test_exact_stationary_target_has_maximum_density():
    assert scores(*state()).item() == 12.


def test_small_pose_and_motion_errors_are_visible_before_a_fall():
    h, g, v, w, target = state()
    assert scores(h + .01, g, v, w, target) < 11.
    g = torch.tensor([[math.sin(math.radians(5)), 0., -math.cos(math.radians(5))]])
    assert scores(h, g, v, w, target) < 9.5
    h, g, v, w, target = state()
    v[:, 0] = .03
    assert scores(h, g, v, w, target) < 9.5


def test_upside_down_stationary_floor_is_not_a_standing_reward():
    h, g, v, w, target = state()
    g[:, 2] = 1.
    assert scores(h, g, v, w, target).item() == 0.


def test_quiet_but_tilted_pose_cannot_collect_separate_velocity_reward():
    h, g, v, w, target = state()
    angle = math.radians(35.)
    g = torch.tensor([[math.sin(angle), 0., -math.cos(angle)]])
    assert scores(h, g, v, w, target).item() < .001


def test_bad_state_fails_closed():
    h, g, v, w, target = state()
    g[:, 0] = float("nan")
    with pytest.raises(ValueError):
        scores(h, g, v, w, target)


def test_terminal_cost_uses_pre_reset_diagnostics():
    from wheeled_tasks.chassis.precise_standing_reward import install
    diagnostic = dict(height=torch.tensor([.1]), gravity=torch.tensor([[1., 0., 0.]]),
        velocity=torch.zeros(1, 3), omega=torch.zeros(1, 3),
        commands=torch.tensor([[0., 0., .305]]), requested_command=torch.tensor([[0., 0., .305]]),
        terminated=torch.tensor([True]))
    observation = {"policy": torch.ones(1, 35)}  # Native step already returned the reset frame.
    env = SimpleNamespace(cfg={}, policy_dt=.005, physical_standing_reward={},
        physical_standing_last=dict(paid_speed=torch.zeros(1), paid_contact=torch.zeros(1), paid_slew=torch.zeros(1)),
        step=lambda action: (observation, torch.tensor([99.]), torch.tensor([True]),
            dict(diagnostics=diagnostic, policy_learning_mask=torch.tensor([True]), log={})))
    install(env)
    obs, reward, done, _ = env.step(torch.zeros(1, 6))
    assert obs is observation and done.item()
    assert reward.item() < -2.99

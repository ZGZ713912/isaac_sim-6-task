"""Contract signs, preservation and boundedness of the zero-PPO speed probe."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from flat_speed_prior import correct, wheel_error, reachable_reference


def test_forward_backward_and_yaw_have_correct_opposite_encoder_axes():
    obs = torch.zeros(4, 35)
    obs[:, 0] = torch.tensor([3., -3., 0., 0.])
    obs[:, 2] = torch.tensor([0., 0., 4., -4.])
    expected = torch.tensor([[5., -5.], [-5., 5.], [-.4373 * 2 / .6, -.4373 * 2 / .6],
                             [.4373 * 2 / .6, .4373 * 2 / .6]])
    torch.testing.assert_close(wheel_error(obs), expected)


def test_low_speed_parent_function_is_exact_for_arbitrary_sensor_states():
    generator = torch.Generator().manual_seed(91)
    obs = torch.randn(512, 35, generator=generator) * 10
    obs[:, 0] = torch.linspace(-.75, .75, len(obs))
    obs[:, 2] = torch.linspace(-1.5, 1.5, len(obs))
    parent = torch.randn(512, 6, generator=generator)
    assert torch.equal(correct(obs, parent, 1.), parent)


def test_high_speed_changes_wheels_only_and_keeps_the_declared_bound():
    obs = torch.zeros(2, 35)
    obs[:, 0] = torch.tensor([3., -3.])
    obs[:, 20:22] = torch.tensor([[100., -100.], [-100., 100.]])
    parent = torch.randn(2, 6)
    result = correct(obs, parent, 1.)
    assert torch.equal(result[:, :4], parent[:, :4])
    assert torch.all((result - parent).abs() <= 6. + 1e-6)
    assert torch.equal(correct(obs, parent, 0.), parent)


def test_matched_wheel_encoder_speed_has_zero_correction():
    obs = torch.zeros(2, 35)
    obs[:, 0] = torch.tensor([3., -3.])
    obs[:, 20:22] = torch.tensor([[5., -5.], [-5., 5.]])
    parent = torch.randn(2, 6)
    assert torch.equal(correct(obs, parent, .5), parent)


def test_reachable_reference_preserves_low_speed_zero_margin_and_other_sensors():
    obs = torch.randn(512, 35)
    obs[:, 0] = torch.linspace(-.75, .75, len(obs))
    obs[:, 2] = torch.linspace(-1.5, 1.5, len(obs))
    assert torch.equal(reachable_reference(obs, torch.full((512, 1), .3)), obs)
    obs[:, 0] = 3.
    before = obs.clone()
    assert torch.equal(reachable_reference(obs, torch.zeros(512, 1)), obs)
    result = reachable_reference(obs, torch.full((512, 1), .3))
    assert torch.equal(obs, before)
    assert torch.equal(result[:, 3:], obs[:, 3:])
    assert torch.equal(result[:, 1], obs[:, 1])


def test_reachable_reference_advances_both_directions_and_uses_imu_yaw():
    obs = torch.zeros(4, 35)
    obs[:, 0] = torch.tensor([3., -3., 0., 0.])
    obs[:, 2] = torch.tensor([0., 0., 4., -4.])
    obs[:, 6] = torch.tensor([0., 0., .5, -.5])
    result = reachable_reference(obs, torch.full((4, 1), .3))
    torch.testing.assert_close(result[:, 0], torch.tensor([.3, -.3, 0., 0.]))
    torch.testing.assert_close(result[:, 2], torch.tensor([0., 0., 1.+.3/(.4373/2), -1.-.3/(.4373/2)]))


def test_pure_rotation_keeps_zero_translation_target_despite_encoder_drift():
    obs = torch.zeros(2,35)
    obs[0,2] = 4.; obs[0,20:22] = torch.tensor([1.,-1.])
    obs[1,0] = 3.; obs[1,2] = .3; obs[1,6] = 1.
    result = reachable_reference(obs,torch.full((2,1),.05))
    assert result[0,0] == 0.
    assert result[1,2] == obs[1,2]

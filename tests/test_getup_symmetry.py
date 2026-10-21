import torch
from torch import nn

from wheeled_algo.getup_symmetry import reflect_actions, reflect_observation, canonical_recovery_logits


def test_body_and_motor_reflection_is_an_involution_with_correct_vector_parity():
    obs = torch.arange(70, dtype=torch.float32).reshape(2, 35)
    reflected = reflect_observation(obs)
    torch.testing.assert_close(reflect_observation(reflected), obs)
    torch.testing.assert_close(reflected[:, [0, 3, 5, 7, 9]], obs[:, [0, 3, 5, 7, 9]])
    torch.testing.assert_close(reflected[:, [1, 2, 4, 6, 8]], -obs[:, [1, 2, 4, 6, 8]])
    torch.testing.assert_close(reflected[:, 28:], obs[:, 28:])


def test_canonical_recovery_shares_left_right_gradients_and_has_no_central_jump():
    torch.manual_seed(13)
    head = nn.Sequential(nn.Linear(35, 64), nn.ELU(), nn.Linear(64, 6))
    scale = torch.ones(35)
    obs = torch.randn(24, 35)
    actual = canonical_recovery_logits(head, reflect_observation(obs), scale)
    expected = reflect_actions(canonical_recovery_logits(head, obs, scale))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    seam = obs[:1].repeat(2, 1)
    seam[:, 8] = torch.tensor([-.000001, .000001])
    value = canonical_recovery_logits(head, seam, scale)
    assert (value[0] - value[1]).abs().max() < .0001
    value.sum().backward()
    assert head[-1].weight.grad.abs().max() > 0

import math

import torch

from wheeled_tasks.chassis.paired_feedback import paired_position_error
from wheeled_tasks.chassis.paired_leg_targets import paired_leg_targets


def test_pd_error_is_invariant_to_independent_encoder_turns():
    measured = torch.tensor([[.2, 1.4, -.2, -1.4]], dtype=torch.float64)
    target = measured + torch.tensor([[.1, -.1, -.2, .2]], dtype=torch.float64)
    turns = torch.tensor([[2., -2., -1., 3.]], dtype=torch.float64) * math.tau
    expected = target - measured
    torch.testing.assert_close(paired_position_error(target, measured + turns), expected, atol=1e-14, rtol=0)
    torch.testing.assert_close(paired_position_error(target + math.tau, measured + turns), expected, atol=1e-14, rtol=0)


def test_ordinary_small_angle_pd_is_unchanged():
    measured = torch.tensor([[-.42, .137, .42, -.137]], dtype=torch.float64)
    target = measured + torch.tensor([[.02, .03, -.04, -.01]], dtype=torch.float64)
    torch.testing.assert_close(paired_position_error(target, measured), target - measured, atol=1e-15, rtol=0)


def test_reporting_seam_does_not_reverse_the_requested_knee_motion():
    # A real native trajectory: left hip and auxiliary straddle a 4pi seam.
    measured = torch.tensor([[5.72, -5.48, .42, -.137]], dtype=torch.float64)
    coherent = measured.clone()
    coherent[:, 1] += 2 * math.tau
    target = coherent + torch.tensor([[.1, -.1, 0., 0.]], dtype=torch.float64)
    target[:, :2] -= math.tau
    error = paired_position_error(target, measured)
    torch.testing.assert_close(error[:, :2], torch.tensor([[.1, -.1]], dtype=torch.float64), atol=1e-14, rtol=0)
    assert (target - measured)[0, 0] < -6.
    assert (target - measured)[0, 1] > 6.


def test_wrapped_feedback_matches_continuous_encoder_paired_pd():
    generator = torch.Generator().manual_seed(712)
    common = 40. * torch.rand(200, 2, generator=generator, dtype=torch.float64) - 20.
    delta = 1.4 * torch.rand(200, 2, generator=generator, dtype=torch.float64) - .1
    sign = torch.tensor([1., -1.], dtype=torch.float64)
    continuous = torch.stack((common - .5 * sign * delta, common + .5 * sign * delta), -1).flatten(-2)
    desired = continuous + .4 * torch.randn(200, 4, generator=generator, dtype=torch.float64)
    turns = torch.randint(-4, 5, (200, 4), generator=generator).to(torch.float64)
    independently_wrapped = continuous + math.tau * turns
    reference = paired_leg_targets(desired, continuous) - continuous
    target = paired_leg_targets(desired, independently_wrapped)
    torch.testing.assert_close(paired_position_error(target, independently_wrapped), reference, atol=4e-14, rtol=0)

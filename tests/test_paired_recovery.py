"""A target representation must allow full orbits without changing the knee."""
import math

import torch

from wheeled_algo.paired_recovery import paired_offsets, wrap
from wheeled_tasks.chassis.paired_leg_targets import paired_leg_targets


NOMINAL = torch.tensor([-.42, .13742282595395358, .42, -.1374155762580851], dtype=torch.float64)


def test_common_wrap_preserves_decoded_physical_targets_and_knees():
    generator = torch.Generator().manual_seed(872)
    offsets = torch.randn(512, 2, 1, generator=generator, dtype=torch.float64) * 15.
    offsets = offsets.expand(-1, -1, 2).clone().flatten(1)
    offsets[:, 1] += torch.linspace(-.65, .75, len(offsets))
    offsets[:, 3] -= torch.linspace(-.65, .75, len(offsets))
    current = offsets + NOMINAL + .01
    before = paired_leg_targets(offsets + NOMINAL, current)
    after = paired_leg_targets(paired_offsets(offsets) + NOMINAL, current)
    torch.testing.assert_close(after, before, rtol=0, atol=2e-13)
    assert (paired_offsets(offsets) / .25).abs().max() < 16.


def test_repeated_full_orbits_keep_moving_across_both_seams():
    measured = torch.zeros(1, 4, dtype=torch.float64)
    previous = measured.clone()
    for _ in range(160):
        requested = previous + .2
        target = paired_offsets(requested, wrap(measured))
        physical = paired_leg_targets(target + NOMINAL, measured + NOMINAL) - NOMINAL
        torch.testing.assert_close(physical - measured, torch.full_like(measured, .2), rtol=0, atol=1e-12)
        measured, previous = physical, target
    assert measured.min() > 5 * math.tau


def test_straddling_encoder_seam_does_not_invent_a_large_common_angle_error():
    measured = torch.tensor([[math.pi - .1, math.pi + .1, -math.pi - .1, -math.pi + .1]], dtype=torch.float64)
    target = paired_offsets(measured + .1, wrap(measured))
    physical = paired_leg_targets(target + NOMINAL, measured + NOMINAL) - NOMINAL
    torch.testing.assert_close(physical - measured, torch.full_like(measured, .1), rtol=0, atol=1e-12)


def test_target_cannot_lap_measured_joint_even_with_accumulated_action_history():
    measured = torch.zeros(1, 4, dtype=torch.float64)
    target = paired_offsets(torch.full_like(measured, 2.), wrap(measured))
    torch.testing.assert_close(target, torch.full_like(measured, .35))

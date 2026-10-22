import math

import torch

from wheeled_algo.paired_recovery import DELTA_BOUNDS, NOMINAL_DELTA, wrap
from wheeled_algo.recovery_coordinates import smooth_recovery_legs
from wheeled_tasks.chassis.paired_leg_targets import paired_leg_targets


NOMINAL = torch.tensor([-.42, .13742282595395358, .42, -.1374155762580851], dtype=torch.float64)
CENTER = (.924544, .924537)


def offsets_for_delta(delta):
    sign = delta.new_tensor([1., -1.])
    difference = sign * delta - delta.new_tensor(NOMINAL_DELTA)
    return torch.stack((-.5 * difference, .5 * difference), -1).flatten(-2)


def test_knee_gradient_remains_available_at_the_old_integrator_stop():
    delta = torch.tensor([[row[0] for row in DELTA_BOUNDS]], dtype=torch.float64)
    measured = offsets_for_delta(delta)
    raw = torch.zeros_like(measured, requires_grad=True)
    target = smooth_recovery_legs(raw, measured / .25, measured, .3, CENTER) * .25
    pair = target.unflatten(-1, (2, 2))
    knee = target.new_tensor([1., -1.]) * (pair[..., 1] - pair[..., 0] + target.new_tensor(NOMINAL_DELTA))
    knee.sum().backward()
    # A unit raw change must still move the knee measurably at the stop.
    # The smooth measured-relative lead intentionally reduces this derivative.
    assert raw.grad.abs().min() > 5e-3
    assert ((knee - delta) > 0.).all()
    assert ((knee - delta) < .15).all()


def test_knee_request_converges_without_a_jump_or_accumulated_clamp():
    measured = offsets_for_delta(torch.tensor([[row[0] for row in DELTA_BOUNDS]], dtype=torch.float64))
    previous = measured / .25
    for _ in range(80):
        target = smooth_recovery_legs(torch.zeros_like(measured), previous, wrap(measured), .3, CENTER)
        physical = paired_leg_targets(target * .25 + NOMINAL, measured + NOMINAL) - NOMINAL
        assert (physical - measured).abs().max() < .076
        measured, previous = physical, target
    pair = measured.unflatten(-1, (2, 2))
    knee = measured.new_tensor([1., -1.]) * (pair[..., 1] - pair[..., 0] + measured.new_tensor(NOMINAL_DELTA))
    torch.testing.assert_close(knee, measured.new_tensor([CENTER]), atol=1e-7, rtol=0)


def test_full_orbits_preserve_knee_and_common_rotation_direction():
    measured = offsets_for_delta(torch.tensor([CENTER], dtype=torch.float64))
    previous = measured / .25
    raw = measured.new_tensor([[1., 1., -1., -1.]])
    for _ in range(240):
        target = smooth_recovery_legs(raw, previous, wrap(measured), .3, CENTER)
        physical = paired_leg_targets(target * .25 + NOMINAL, measured + NOMINAL) - NOMINAL
        common_change = (physical - measured).unflatten(-1, (2, 2)).mean(-1)
        torch.testing.assert_close(common_change, measured.new_tensor([[1., -1.]]) * (.075 * math.tanh(1.)), rtol=0, atol=2e-13)
        measured, previous = physical, target
    assert measured[0, 0] > 2 * math.tau
    assert measured[0, 2] < -2 * math.tau

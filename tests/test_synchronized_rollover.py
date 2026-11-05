import math
import pytest

from wheeled_tasks.chassis.synchronized_rollover import Sweep


def test_mirrored_axes_preserve_pair_and_physical_direction():
    sweep = Sweep(1)
    sweep.angle = 2.
    initial = [-.42, 1., .42, -1.]
    delta = [a-b for a, b in zip(sweep.targets(initial), initial)]
    assert delta == [2., 2., -2., -2.]
    reverse = Sweep(-1)
    reverse.angle = 2.
    assert reverse.targets(initial) == pytest.approx([q-d for q, d in zip(initial, delta)])


def test_profile_never_jumps_position_and_brakes_before_capture():
    direct, braked = Sweep(1), Sweep(1, profile="braked")
    for _ in range(80):
        direct.step(.01, 180.)
        braked.step(.01, 180.)
    assert direct.angle == braked.angle
    old_velocity, old_angle = braked.velocity, braked.angle
    braked.step(.01, 70.)
    assert braked.braking and abs(braked.velocity-old_velocity) <= .2+1e-12
    assert 0. < braked.angle-old_angle <= .04
    for _ in range(40):
        braked.step(.01, 10.)
    assert braked.captured and braked.velocity == 0.
    stopped = braked.angle
    braked.step(.01, 40.)
    assert braked.angle == stopped


def test_preparation_and_stroke_are_bounded():
    sweep = Sweep(1)
    for _ in range(50):
        assert sweep.step(.01, 180.) == 0.
    for _ in range(300):
        sweep.step(.01, 180.)
    assert sweep.angle == 2.*math.pi and sweep.velocity == 0.
    with pytest.raises(ValueError):
        Sweep(1, float("nan"))
    with pytest.raises(ValueError):
        sweep.step(0., 180.)

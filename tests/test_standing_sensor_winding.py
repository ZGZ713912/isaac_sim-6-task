import math

import pytest
import torch

from wheeled_tasks.chassis.standing_geometry107 import extend_settings as extend_geometry
from wheeled_tasks.chassis.standing_sensor_winding import (
    coordinate_from_settings, extend_settings, observer_from_settings,
    validate_observation_contract, winding_from_settings,
)
from test_standing_geometry107 import settings


def test_many_actual_physx_wheel_seams_preserve_the_measured_distance():
    task = extend_settings(extend_geometry(settings()))
    coordinate = coordinate_from_settings(task)
    winding = winding_from_settings(task, 2, "cpu", coordinate)
    initial = torch.tensor([[-.42, .13742283, .42, -.13741558, 0., 0.]]).repeat(2, 1)
    gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    start = coordinate(winding.update(initial), gravity)[0]
    for tick in range(1, 1001):
        # 20 revolutions, with independent wraps across the 4pi PhysX seam.
        angle = tick * .125
        exact = initial + torch.tensor([[0., 0., 0., 0., angle, -angle]])
        raw = torch.remainder(exact + math.tau, 2 * math.tau) - math.tau
        continuous = winding.update(raw)
        actual, valid = coordinate(continuous, gravity)
        assert valid.all()
        torch.testing.assert_close(actual - start, torch.full((2, 2), .06 * angle), atol=3e-6, rtol=0.)
        assert torch.equal(winding.update(raw), continuous)
    saved = winding.encoders.q[1].clone()
    winding.reset(torch.tensor([0]))
    next_raw = raw.clone()
    next_raw[0] = initial[0]
    winding.update(next_raw)
    assert winding.encoders.initialized.all()
    # The other environment remains initialized and retains its turn count.
    assert winding.encoders.winding[1, 2] != 0
    assert torch.equal(winding.encoders.q[1], saved)


def test_no_seam_sensor_frames_are_bitwise_unchanged():
    task = extend_settings(extend_geometry(settings()))
    coordinate = coordinate_from_settings(task)
    winding = winding_from_settings(task, 1, "cpu", coordinate)
    for tick in range(100):
        q = torch.tensor([[-.42, .13742283, .42, -.13741558, tick * .01, -tick * .01]])
        assert torch.equal(winding.update(q), q)


def test_disabled_and_legacy_observations_keep_the_old_lifecycle():
    assert not validate_observation_contract(dict(standing_position_observation=None))
    assert observer_from_settings(dict(standing_position_observation=None), 1, "cpu") is None
    assert validate_observation_contract(settings())
    old = extend_geometry(settings())
    assert winding_from_settings(old, 1, "cpu", coordinate_from_settings(old)) is None


def test_winding_contract_cannot_be_silently_changed():
    task = extend_settings(extend_geometry(settings()))
    task["standing_position_observation"]["encoder_winding"]["reset"] = "every_frame"
    with pytest.raises(ValueError):
        validate_observation_contract(task)

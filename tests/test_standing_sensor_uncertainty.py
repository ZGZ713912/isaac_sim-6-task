import math
from copy import deepcopy

import pytest
import torch

from test_standing_geometry107 import settings
from wheeled_tasks.chassis.standing_geometry107 import extend_settings as extend_geometry
from wheeled_tasks.chassis.standing_sensor_winding import extend_settings as extend_winding
from wheeled_tasks.chassis.standing_sensor_uncertainty import (
    coordinate_from_settings, extend_settings, validate_observation_contract,
    winding_from_settings,
)


def task():
    return extend_settings(extend_winding(extend_geometry(settings())))


def test_in_domain_sensor_frames_are_bitwise_unchanged():
    old_settings = extend_winding(extend_geometry(settings()))
    old, new = coordinate_from_settings(old_settings), coordinate_from_settings(task())
    torch.manual_seed(7)
    q = torch.randn(512, 6)
    q[:, 1] = q[:, 0] + torch.linspace(-.13, 1.36, 512)
    q[:, 3] = q[:, 2] - torch.linspace(-.13, 1.36, 512)
    original = q.clone()
    gravity = torch.tensor([[0., 0., -1.]]).expand(len(q), -1)
    a, va = old(q, gravity)
    b, vb = new(q, gravity)
    assert va.all() and torch.equal(a, b) and torch.equal(va, vb)
    assert torch.equal(q, original)


@pytest.mark.parametrize("side,endpoint", [(0, 0), (0, -1), (1, 0), (1, -1)])
@pytest.mark.parametrize("common", [-math.tau, 0., math.tau])
def test_small_encoder_error_near_endpoints_is_projected_but_large_error_rejected(side, endpoint, common):
    coordinate = coordinate_from_settings(task())
    q = torch.tensor([[-.42, .13742283, .42, -.13741558, 2., -2.]])
    limits = getattr(coordinate.geometry, f"side{side}_delta_rad")
    i = side * 2
    direction = -1. if endpoint == 0 else 1.
    q[:, i] = common
    q[:, i + 1] = common + limits[endpoint] + direction * .008
    gravity = torch.tensor([[0., 0., -1.]])
    result, valid = coordinate(q, gravity)
    assert valid.all() and torch.isfinite(result).all() and coordinate.last_projected.all()
    # Original table and physical bounds remain exactly the same. Measurements
    # beyond the declared error bound and inverted gravity remain invalid.
    q[:, i + 1] += direction * .008
    result, valid = coordinate(q, gravity)
    assert not valid.any() and not result.any() and coordinate.last_rejected.all()
    assert not coordinate(q, -gravity)[1].any()


def test_nonfinite_and_continuous_wheel_distance_guard():
    config = task()
    coordinate = coordinate_from_settings(config)
    winding = winding_from_settings(config, 1, "cpu", coordinate)
    q = torch.tensor([[-.42, .13742283, .42, -.13741558, 0., 0.]])
    # Simulate a small sensor error below the physical low-knee endpoint.
    q[:, 1] = q[:, 0] + coordinate.geometry.side0_delta_rad[0] - .008
    gravity = torch.tensor([[0., 0., -1.]])
    initial = coordinate(winding.update(q), gravity)[0]
    for tick in range(1, 301):
        exact = q + torch.tensor([[0., 0., 0., 0., tick * .125, -tick * .125]])
        raw = torch.remainder(exact + math.tau, 2. * math.tau) - math.tau
        measured, valid = coordinate(winding.update(raw), gravity)
        assert valid.all()
        torch.testing.assert_close(measured - initial, torch.full((1, 2), .06 * tick * .125), rtol=0., atol=3e-6)
    q[:, 1] = float("nan")
    result, valid = coordinate(q, gravity)
    assert not valid.any() and not result.any()


def test_uncertainty_and_geometry_contract_cannot_be_silently_widened():
    for mutate in (lambda s: s["standing_position_observation"]["sensor_projection"].update(relative_error_bound_rad=.02),
                   lambda s: s["standing_position_observation"].update(passive_knee_domain_deg=[39., 108.]),
                   lambda s: s.update(encoder_bias_rad=.01)):
        changed = deepcopy(task())
        mutate(changed)
        with pytest.raises(ValueError):
            validate_observation_contract(changed)

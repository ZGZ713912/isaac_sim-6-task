"""Sensor geometry and observation-reference lifecycle, without simulated truth."""
import math

import pytest
import torch

from wheeled_algo.standing_displacement import WheelAngleCoordinate, StandingDisplacement


def nominal(count=3):
    model = WheelAngleCoordinate()
    q = torch.cat((model.geometry.nominal, torch.zeros(2)))[None].repeat(count, 1)
    g = torch.tensor([[0., 0., -1.]]).repeat(count, 1)
    return model, q, g


def test_continuous_wheel_angle_has_physical_radius_and_opposite_signs():
    model, q, g = nominal()
    before, valid = model(q, g)
    assert valid.all()
    q[:, 4] += torch.tensor([.5, 2.*math.pi, 30.])
    q[:, 5] -= torch.tensor([.5, 2.*math.pi, 30.])
    after, _ = model(q, g)
    expected = .06*torch.tensor([.5, 2.*math.pi, 30.])[:, None].expand(-1, 2)
    torch.testing.assert_close(after-before, expected, atol=3e-7, rtol=1e-6)
    assert not list(model.parameters())
    model.geometry.verify_frozen_geometry()


@pytest.mark.parametrize("bad", ["inverted", "side", "nonfinite_q", "nonfinite_g", "invalid_knee"])
def test_invalid_geometry_fails_closed_without_nonfinite_observation(bad):
    model, q, g = nominal()
    if bad == "inverted":
        g[1, 2] = 1.
    elif bad == "side":
        g[1] = torch.tensor([0., 1., 0.])
    elif bad == "nonfinite_q":
        q[1, 4] = float("nan")
    elif bad == "nonfinite_g":
        g[1, 0] = float("inf")
    else:
        q[1, 1] = q[1, 0] + 2.
    result, valid = model(q, g)
    assert valid[:, 0].tolist() == [True, False, True]
    assert torch.isfinite(result).all() and torch.equal(result[1], torch.zeros(2))


def test_reference_is_captured_once_and_does_not_follow_drift_or_invalid_samples():
    observer = StandingDisplacement(2, "cpu")
    p = torch.tensor([[10., 20.], [-3., -4.]])
    yes = torch.ones(2, dtype=torch.bool)
    result, active = observer.update(p, yes, yes, yes)
    assert active.all() and result.count_nonzero() == 0
    p += .04
    result, _ = observer.update(p, yes, yes, torch.zeros_like(yes))
    torch.testing.assert_close(result, torch.full((2, 2), .04/11.), atol=1e-7, rtol=1e-5)
    anchor = observer.anchor.clone()
    filtered = observer.filtered.clone()
    p[0] = float("nan")
    result, active = observer.update(p, yes, yes, yes)
    assert active.tolist() == [False, True] and torch.isfinite(result).all()
    assert torch.equal(observer.anchor, anchor)
    assert torch.equal(observer.filtered[0], filtered[0])
    p[0] = anchor[0] + .06
    result, _ = observer.update(p, yes, yes, yes)
    assert observer.active.all() and torch.equal(observer.anchor, anchor)
    assert (result[0] > filtered[0]).all()


def test_only_ended_request_or_explicit_reset_reanchors_selected_rows():
    observer = StandingDisplacement(3, "cpu")
    p = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    yes = torch.ones(3, dtype=torch.bool)
    observer.update(p, yes, yes, yes)
    initial_anchor = observer.anchor.clone()
    observer.update(p+.02, yes, torch.tensor([True, False, True]), yes)
    assert observer.active.tolist() == [True, False, True]
    observer.update(p+.03, yes, yes, yes)
    assert torch.equal(observer.anchor[[0, 2]], initial_anchor[[0, 2]])
    torch.testing.assert_close(observer.anchor[1], p[1]+.03)
    observer.reset(torch.tensor([0]))
    assert observer.active.tolist() == [False, True, True]
    assert torch.equal(observer.anchor[2], initial_anchor[2])


def test_repeated_observation_in_one_policy_tick_cannot_advance_filter_twice():
    observer = StandingDisplacement(2, "cpu")
    yes = torch.ones(2, dtype=torch.bool)
    q = torch.zeros(2, 2)
    observer.update(q, yes, yes, yes, sample_tick=10)
    first, _ = observer.update(q+.04, yes, yes, yes, sample_tick=11)
    repeated, _ = observer.update(q+.04, yes, yes, yes, sample_tick=11)
    assert torch.equal(first, repeated)
    observer.reset(torch.tensor([0]))
    resumed, _ = observer.update(q+.04, yes, yes, yes, sample_tick=11)
    assert resumed[0].count_nonzero() == 0
    assert torch.equal(resumed[1], first[1])
    advanced, _ = observer.update(q+.05, yes, yes, yes, sample_tick=12)
    assert (advanced > resumed).all()


@pytest.mark.parametrize("kwargs", [{"dt": 0.}, {"filter_seconds": float("nan")}, {"limit_m": -1.}])
def test_invalid_time_or_observation_bounds_rejected(kwargs):
    with pytest.raises(ValueError, match="finite and positive"):
        StandingDisplacement(1, "cpu", **kwargs)

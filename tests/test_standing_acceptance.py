import numpy as np
import pytest

from wheeled_algo.standing_acceptance import excursion_limit, high_recentering


SPEC = dict(tail_seconds=10., outward_radial_slope_max_m_s=.001,
            near_reference_radius_m=.05, minimum_peak_return_fraction=.2)


def trace(x):
    time = np.linspace(0., 30., 6001)
    return time, np.column_stack((x(time), np.zeros_like(time)))


def test_legacy_uniform_limit_preserved():
    assert excursion_limit({"stand_xy_excursion_max_m": .05}, "stand_high") == .05


def test_only_user_high_bound_changes():
    protocol = {"stand_xy_excursion_max_m": .05, "stand_xy_excursion_max_m_by_case":
                {"stand_low": .05, "stand": .05, "stand_high": .10}}
    assert [excursion_limit(protocol, c) for c in ("stand_low", "stand", "stand_high")] == [.05, .05, .10]


@pytest.mark.parametrize("bad", [{"stand_high": .1}, {"stand_low": .05, "stand": .05, "stand_high": float("nan")}])
def test_incomplete_or_nonfinite_limits_rejected(bad):
    with pytest.raises(ValueError, match="finite positive"):
        excursion_limit({"stand_xy_excursion_max_m_by_case": bad}, "stand_high")


def test_startup_six_cm_then_slow_recenter_allowed():
    time, xy = trace(lambda t: np.where(t < 1., .06 * t, .06 * np.exp(-(t - 1.) / 12.)))
    row = high_recentering(time, xy, SPEC)
    assert row["returned"] and row["no_sustained_outward_drift"]
    assert row["peak_excursion_m"] > .05


def test_bounded_nonzero_offset_does_not_require_zero_drift():
    time, xy = trace(lambda t: .035 * (1. - np.exp(-t)) + .004 * np.sin(t))
    row = high_recentering(time, xy, SPEC)
    assert row["returned"] and row["no_sustained_outward_drift"]


def test_below_ten_cm_but_continuous_outward_drift_rejected():
    time, xy = trace(lambda t: .003 * t)
    row = high_recentering(time, xy, SPEC)
    assert row["peak_excursion_m"] < .10
    assert not row["returned"] and not row["no_sustained_outward_drift"]


def test_inward_motion_is_not_rejected_as_speed():
    time, xy = trace(lambda t: np.where(t < 1., .09 * t, .09 - .002 * (t - 1.)))
    row = high_recentering(time, xy, SPEC)
    assert row["tail_outward_radial_slope_m_s"] < -.001
    assert row["returned"] and row["no_sustained_outward_drift"]


def test_short_tail_cannot_qualify():
    with pytest.raises(ValueError, match="tail window"):
        high_recentering([0., .5, 1.], np.zeros((3, 2)), SPEC)


def test_missing_long_evidence_cannot_qualify_new_protocol():
    from export_robust_getup import assess_extended_standing
    row = assess_extended_standing([], "unused", {"long_standing_confirmation": {"required_seeds": [1, 2]}})
    assert not row["adequate"]
    assert "long standing" in row["failures"][0]


def test_extended_review_allows_high_sway_but_preserves_other_gates():
    from copy import deepcopy
    from wheeled_algo.standing_acceptance import review_extended_standing
    cases = ("stand_low", "stand", "stand_high")
    hf = dict(leg_motion_fast_rms_rad_3_95=.001, wheel_motion_fast_rms_rad_s_3_95=.1)
    protocol = dict(closure_gap_max_m=.003, stand_xy_excursion_max_m_by_case=dict(zip(cases, [.05, .05, .10])),
        long_standing_confirmation=dict(minimum_seconds=30., replicas_per_case=1,
            wheel_clearance_after_02s_max_m=.003, high_recentering=SPEC,
            actual_hf_baseline=dict(by_case={case: hf for case in cases})))
    report = dict(lanes=[dict(lane=i, name=name) for i, name in enumerate(cases)])
    analysis = dict(passive_motor_peak_nm=0., max_gap_m=.001,
        cases={case: dict(survived=1, quiet=dict(hf)) for case in cases})
    time, high = trace(lambda t: np.where(t < 1., .08 * t, .08 * np.exp(-(t - 1.) / 12.)))
    xy = np.zeros((len(time), 3, 3))
    xy[:, 2, :2] = high
    data = dict(time=np.repeat(time[:, None], 3, axis=1), local_position=xy,
        alive=np.ones((len(time), 3), dtype=bool), passive=np.zeros((len(time), 3), dtype=bool),
        wheel_clearance_peak=np.zeros((len(time), 3)))
    assert review_extended_standing(report, analysis, data, protocol)["adequate"]
    bad = deepcopy(analysis)
    bad["cases"]["stand_high"]["quiet"]["wheel_motion_fast_rms_rad_s_3_95"] = .2
    assert not review_extended_standing(report, bad, data, protocol)["adequate"]
    lifted = dict(data, wheel_clearance_peak=np.full((len(time), 3), .004))
    assert not review_extended_standing(report, analysis, lifted, protocol)["adequate"]
    low_shift = xy.copy()
    low_shift[:, 0, 0] = high[:, 0]
    assert not review_extended_standing(report, analysis, dict(data, local_position=low_shift), protocol)["adequate"]

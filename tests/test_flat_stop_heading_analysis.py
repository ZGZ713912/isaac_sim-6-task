import numpy as np

from analyze_flat_stop_heading import complete_window_integrals


def test_body_gyro_integrates_in_seconds_and_excludes_window_edges():
    dt = .005
    ticks = np.tile(np.arange(1, 3601)[:, None], (1, 2))
    rate = np.full_like(ticks, .1, dtype=float)
    rate[(ticks-1)*dt < 8.] = 99.
    rate[(ticks-1)*dt >= 16.] = 99.
    valid = np.ones_like(ticks, dtype=bool)
    actual = complete_window_integrals(rate, valid, ticks, dt, 8., 16.)
    np.testing.assert_allclose(actual, [.8, .8])


def test_partial_stop_cannot_look_like_small_yaw_excursion():
    dt = .005
    ticks = np.tile(np.arange(1, 3601)[:, None], (1, 2))
    rate = np.full_like(ticks, .1, dtype=float)
    valid = np.ones_like(ticks, dtype=bool)
    valid[3000:, 1] = False
    actual = complete_window_integrals(rate, valid, ticks, dt, 8., 16.)
    assert abs(actual[0]-.8) < 1e-12
    assert actual[1] is None

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from flat_transient_metrics import (RETENTION, STEADY, TRANSIENT, response_cases,
                                    first_dwell, lane_metrics, condition_summary, assess_response)


def test_profiles_are_prospective_complete_cycles_and_source_unchanged():
    source = json.loads((ROOT/'contracts/v6_task_campaign_speed_worker_v18.json').read_text())['evaluation']['cases']
    before = json.dumps(source, sort_keys=True)
    cases = response_cases(source)
    assert [c['name'] for c in cases] == list(RETENTION + STEADY + TRANSIENT)
    assert json.dumps(source, sort_keys=True) == before
    by_name = {c['name']: c for c in cases}
    assert by_name['backward_2p5']['skill']['command'][0] == -2.5
    from analyze_flat_raw_requests import requested_planar
    ticks = np.array([1, 801, 1601, 2401])
    forward = requested_planar(ticks, by_name[TRANSIENT[0]], .005)
    reverse = requested_planar(ticks, by_name[TRANSIENT[1]], .005)
    np.testing.assert_allclose(forward[:, 0], [3., 0., -2.5, 0.])
    np.testing.assert_allclose(reverse[:, 0], [-2.5, 0., 3., 0.])


def test_dwell_requires_contiguous_samples_and_handles_censoring():
    assert first_dwell([False, True, True, True], .1, .3) == .1
    assert first_dwell([True, False, True, True], .1, .3) is None
    assert first_dwell([], .1) is None


def trace(stop_speed=0., incomplete=False):
    dt, count = .005, 3200
    ticks = np.arange(1, count+1)[:, None]
    time = (ticks[:, 0]-1)*dt
    phase = (time//4).astype(int)
    actual = np.where(phase == 0, 3., np.where(phase == 2, -2.5, stop_speed))
    velocity = np.zeros((count, 1, 3))
    velocity[:, 0, 0] = actual
    gravity = np.zeros_like(velocity)
    gravity[:, :, 2] = -1.
    position = np.zeros_like(velocity)
    position[:, 0, 0] = np.cumsum(actual)*dt
    active = np.ones((count, 1), dtype=bool)
    done = np.zeros_like(active)
    done[-1] = True
    if incomplete:
        active[1000:] = False
        done[999] = True
    dq = np.zeros((count, 1, 6))
    dq[:, 0, 2], dq[:, 0, 5] = actual/.06, -actual/.06
    return dict(episode_ticks=ticks, active=active, done=done, velocity=velocity,
        omega=np.zeros_like(velocity), position=position, gravity=gravity,
        gap=np.zeros((count, 1)), motor_velocity=dq, motor_position=np.zeros_like(dq))


def case():
    return dict(original_case=TRANSIENT[0], command=[3., 0., .305],
        skill=dict(kind='start_stop', segment_seconds=4., opposite_speed_ratio=5./6.))


def test_stop_metrics_have_explicit_zero_and_censored_outcomes():
    complete = lane_metrics(trace(), 0, case(), .005, 16.)
    assert complete['complete']
    assert complete['phases'][1]['settle_time_s'] == 0.
    assert complete['phases'][1]['travel_distance_m'] == 0.
    assert complete['slip_mean_m_s'] == pytest.approx(0.)
    assert condition_summary([complete])['final_planar_m_s'] == 0.
    slipping = lane_metrics(trace(stop_speed=.1), 0, case(), .005, 16.)
    assert slipping['phases'][1]['settle_time_s'] is None
    assert slipping['phases'][1]['travel_distance_m'] == pytest.approx(.4)
    assert condition_summary([slipping])['censored_stops'] == 2
    failed = lane_metrics(trace(incomplete=True), 0, case(), .005, 16.)
    summary = condition_summary([failed])
    assert summary['complete'] == 0
    assert summary['final_planar_m_s'] is None
    assert summary['settle_time_s'] is None


def test_no_evidence_never_admits_long_training():
    native = dict(checkpoint_comparison={'0': dict(role='parent'), '2': dict(role='trained')},
                  result=dict(cases={}))
    result = assess_response(native, dict(parent={}, trained={}))
    assert not result['admitted']

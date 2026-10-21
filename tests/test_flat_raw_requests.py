"""A lowered internal reference cannot count as tracking the human request."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_flat_raw_requests import motion_stats, requested_planar
from wheeled_tasks.chassis.skill_commands import profile_command


@pytest.mark.parametrize('skill,command', [
    ({'kind': 'forward'}, [3., 0., .305]),
    ({'kind': 'backward'}, [-3., 0., .42]),
    ({'kind': 'start_stop', 'segment_seconds': 8.}, [3., 0., .305]),
    ({'kind': 'start_stop', 'segment_seconds': 3.}, [-3., 0., .305]),
    ({'kind': 'rotate', 'yaw_start_stop_segment_seconds': 8.}, [0., -4., .305]),
])
def test_reconstructed_request_matches_frozen_native_profile(skill, command):
    ticks = np.array([[1, 200], [600, 601], [1600, 1601], [3200, 3201], [6400, 6401]])
    actual = requested_planar(ticks, dict(command=command, skill=skill), .005)
    time = torch.from_numpy((ticks-1).astype(np.float32).reshape(-1)) * .005
    expected = profile_command(time, command, skill)[:, :2].numpy().reshape(actual.shape)
    np.testing.assert_array_equal(actual, expected)


def test_issued_reference_error_does_not_hide_raw_request_error():
    mask = np.ones((2, 3), dtype=bool)
    actual = np.full(mask.shape, 1.5)
    stats = motion_stats(actual, actual.copy(), np.full(mask.shape, 3.), mask)
    assert stats['issued_mae'] == 0.
    assert stats['raw_mae'] == 1.5
    assert stats['changed_reference_fraction'] == 1.


def test_missing_frames_are_not_scored_as_zero_error():
    value = np.ones((2, 3))
    stats = motion_stats(value, value, value, np.zeros_like(value, dtype=bool))
    assert stats['frames'] == 0
    assert all(v is None for k, v in stats.items() if k != 'frames')

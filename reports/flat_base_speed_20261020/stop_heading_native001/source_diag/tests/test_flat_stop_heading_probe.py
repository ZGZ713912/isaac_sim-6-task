from copy import deepcopy

import pytest
import torch

from flat_stop_heading_probe import StopHeadingMemory
from wheeled_tasks.chassis.flat_command_reference import RmcsFlatCommandReference


@pytest.fixture
def references():
    cfg = dict(nominal_height_m=.305, heading_hold_enabled=True, heading_hold_gain=5.,
               heading_hold_yaw_rate_max_rad_s=4., yaw_rate_max_rad_s=4., vx_max_m_s=3.,
               wheel_track_m=.4373, wheel_speed_limit_rad_s=90., wheel_radius_m=.06,
               lateral_acceleration_m_s2=2.06, height_reference_rate_max_m_s=2/3, policy_hz=200)
    base = RmcsFlatCommandReference(2, 'cpu', cfg)
    candidate = RmcsFlatCommandReference(2, 'cpu', deepcopy(cfg))
    probe = StopHeadingMemory(candidate, torch.tensor([True, False]))
    return base, candidate, probe


def tick(reference, probe, vx, yaw_command, yaw, step):
    commands = torch.tensor([[vx, yaw_command, .42]] * 2)
    return probe.update(commands, torch.full((2,), yaw), torch.full((2,), step))


def test_stop_corrects_heading_and_reverse_keeps_original_capture(references):
    base, candidate, probe = references
    for ref in (base, candidate):
        ref.update(torch.tensor([[3., 0., .42]] * 2), torch.zeros(2), torch.zeros(2, dtype=torch.long))
    stopped = tick(candidate, probe, 0., 0., .2, 1)
    assert stopped[:, 0].count_nonzero() == 0
    assert stopped[:, 1].tolist() == pytest.approx([-1., 0.])
    assert candidate.heading_valid.tolist() == [True, False]
    reverse = tick(candidate, probe, -3., 0., .3, 2)
    assert reverse[0, 1].item() == pytest.approx(-1.5)
    assert reverse[1, 1].item() == 0.
    assert candidate.heading.tolist() == pytest.approx([0., .3])


def test_unselected_lanes_and_pre_stop_commands_are_bitwise_original(references):
    base, candidate, probe = references
    for step, (vx, wz, yaw) in enumerate([(0., 0., 0.), (3., 0., .1), (0., 0., .3),
                                         (0., 0., .4), (-3., 0., .5), (0., 1., .5)]):
        expected = tick(base, base, vx, wz, yaw, step)
        actual = tick(candidate, probe, vx, wz, yaw, step)
        assert torch.equal(expected[1], actual[1])
        if step < 2:
            assert torch.equal(expected, actual)


def test_explicit_turn_reset_and_repeated_tick_do_not_retain_stale_heading(references):
    _, candidate, probe = references
    tick(candidate, probe, 3., 0., 0., 0)
    first = tick(candidate, probe, 0., 0., .2, 1).clone()
    assert torch.equal(tick(candidate, probe, 0., 0., 1., 1), first)
    tick(candidate, probe, 0., 1., .2, 2)
    assert not candidate.heading_valid.any()
    assert tick(candidate, probe, 0., 0., .6, 3)[:, 1].count_nonzero() == 0
    tick(candidate, probe, 3., 0., .6, 4)
    candidate.reset(torch.arange(2))
    assert tick(candidate, probe, 0., 0., 1., 0)[:, 1].count_nonzero() == 0


def test_wrap_and_caps_use_original_limits(references):
    _, candidate, probe = references
    tick(candidate, probe, 3., 0., 3.1, 0)
    assert tick(candidate, probe, 0., 0., -3.1, 1)[0, 1].item() == pytest.approx(-.415927, abs=1e-5)
    assert tick(candidate, probe, 0., 0., 0., 2)[0, 1].item() == 4.
    assert candidate.reference[0, 2].item() <= .305 + 3 * (2/3/200) + 1e-6


def test_rejects_invalid_lane_mask(references):
    _, candidate, _ = references
    with pytest.raises(ValueError, match='boolean'):
        StopHeadingMemory(candidate, torch.ones(2))

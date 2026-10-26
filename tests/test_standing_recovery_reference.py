from copy import deepcopy

import pytest
import torch

from test_standing_sensor_uncertainty import task as sensor_task
from wheeled_algo.robust_getup import tilt_gate
from wheeled_tasks.chassis.standing_recovery_reference import (
    extend_settings, observer_from_settings, recovery_capture_gate, validate_observation_contract,
)


def task():
    return extend_settings(dict(sensor_task(), stance_aware_recovery=True, recovery_shape_aware=True,
                                gap_entry_handoff_version="orbit-v2", upright_entry_recovery=True))


def frame():
    obs = torch.zeros(1, 35)
    obs[:, 9] = -1.
    obs[:, 3] = .305 * 5.
    return obs


def test_legacy_capture_gate_retains_the_exact_old_value():
    obs = torch.randn(64, 35)
    old = dict(sensor_task(), stance_aware_recovery=True, recovery_shape_aware=True)
    assert torch.equal(recovery_capture_gate(old, obs), tilt_gate(obs, True, True))


def test_upright_nonstanding_leg_orbit_does_not_capture_until_entry_releases():
    settings = task()
    obs = frame()
    obs[:, 10:14] = torch.tensor([1., 1., -1., -1.])
    assert not tilt_gate(obs, True, True).any()
    assert (recovery_capture_gate(settings, obs) == 1.).all()
    observer = observer_from_settings(settings, 1, "cpu")
    valid = requested = torch.ones(1, dtype=torch.bool)
    eligible = recovery_capture_gate(settings, obs)[:, 0] <= .05
    observer.update(torch.zeros(1, 2), valid, requested, eligible, sample_tick=1)
    assert not observer.active.any()
    obs[:, 10:14] = 0.
    eligible = recovery_capture_gate(settings, obs)[:, 0] <= .05
    value, used = observer.update(torch.full((1, 2), .12), valid, requested, eligible, sample_tick=2)
    assert used.all() and observer.active.all() and not value.any()
    assert torch.equal(observer.anchor, torch.full((1, 2), .12))


def test_fallen_and_ordinary_balancing_keep_expected_gate_endpoints():
    settings = task()
    obs = frame()
    assert not recovery_capture_gate(settings, obs).any()
    obs[:, 9] = 1.
    assert (recovery_capture_gate(settings, obs) == 1.).all()


def test_gate_contract_and_handoff_cannot_be_silently_changed():
    for mutate in (lambda s: s["standing_position_observation"]["recovery_reference_gate"].update(threshold=.2),
                   lambda s: s.update(gap_entry_handoff_version=None),
                   lambda s: s.update(upright_entry_recovery=False)):
        modified = deepcopy(task())
        mutate(modified)
        with pytest.raises(ValueError):
            validate_observation_contract(modified)

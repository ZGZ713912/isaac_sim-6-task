"""Reference revisions cannot erase drift on noise or interpret air-spin as position."""
from copy import deepcopy

import pytest
import torch

from wheeled_algo.standing_displacement_contract import observation_contract as filter_contract
from wheeled_algo.standing_displacement_contract import observer_from_settings as old_factory
from wheeled_algo.standing_reference_lifecycle import (
    observation_contract, observer_from_settings, reference_settings, validate_observation_contract,
)


def settings(new=True, severe=False, seconds=.15):
    task = dict(quiet_velocity_feedback=True, quiet_velocity_feedback_mode="standing_displacement",
                standing_position_observation=filter_contract(seconds))
    if seconds is not None:
        task["standing_position_filter_seconds"] = seconds
    if new:
        task["standing_position_reference"] = reference_settings(.05, severe)
        task["standing_position_observation"] = observation_contract(seconds, .05, severe)
    return task


@pytest.mark.parametrize("seconds", [None, .15])
def test_v1_and_v2_legacy_reference_replay_remains_bitwise(seconds):
    task = settings(False, seconds=seconds)
    old, current = old_factory(task, 3, "cpu"), observer_from_settings(task, 3, "cpu")
    assert current.capture_delay_seconds == .2 and not current.reset_on_severe_tilt
    generator = torch.Generator().manual_seed(622)
    for tick in range(300):
        p = torch.randn(3, 2, generator=generator) * .03 + tick * .0001
        masks = [torch.rand(3, generator=generator) > threshold for threshold in (.05, .02, .2)]
        if tick % 17 == 0:
            old.reset(torch.tensor([1])); current.reset(torch.tensor([1]))
        assert current.disarm_severe_tilt(torch.tensor([[0., 0., 1.]]).repeat(3, 1)) is None
        a, aa = old.update(p, *masks, tick)
        b, ba = current.update(p, *masks, tick)
        assert torch.equal(a, b) and torch.equal(aa, ba)
        assert torch.equal(old.anchor, current.anchor)


def test_capture_delay_is_declared_without_changing_filter_or_control_outputs():
    model = observer_from_settings(settings(), 1, "cpu")
    assert model.capture_delay_seconds == .05 and model.alpha == .005 / .155
    assert not hasattr(model, "actions")


def test_severe_tilt_discards_reference_once_but_noise_and_moderate_lean_preserve_it():
    model = observer_from_settings(settings(severe=True), 3, "cpu")
    yes = torch.ones(3, dtype=torch.bool)
    zero = torch.zeros(3, 2)
    model.update(zero, yes, yes, yes, 0)
    model.update(zero + .04, yes, yes, yes, 1)
    before = model.filtered.clone()
    g = torch.tensor([[0., .76604444, -.64278761], [0., 0., float("nan")], [0., 0., -1.]])
    assert not model.disarm_severe_tilt(g).any()
    assert model.active.all() and torch.equal(model.filtered, before)
    g[1] = torch.tensor([0., 0., 1.])
    for _ in range(3):
        severe = model.disarm_severe_tilt(g)
        assert severe.tolist() == [False, True, False]
    assert model.active.tolist() == [True, False, True]
    assert model.severe_reset_count.tolist() == [0, 1, 0]
    assert torch.equal(model.filtered[[0, 2]], before[[0, 2]])
    spin = zero.clone(); spin[1] = 100.
    model.update(spin, torch.tensor([True, False, True]), yes, yes, 2)
    assert not model.active[1]
    model.disarm_severe_tilt(torch.tensor([[0., 0., -1.]]).repeat(3, 1))
    model.update(spin, yes, yes, torch.tensor([True, False, True]), 3)
    assert not model.active[1]
    model.update(spin, yes, yes, yes, 4)
    assert torch.equal(model.anchor[1], spin[1]) and model.filtered[1].count_nonzero() == 0
    model.reset(torch.tensor([1]))
    assert model.severe_reset_count.tolist() == [0, 0, 0]


@pytest.mark.parametrize("change", ["delay_type", "delay_bound", "reset_type", "extra_field", "missing_field", "wrong_contract"])
def test_malformed_reference_settings_or_contract_rejected(change):
    task = deepcopy(settings())
    reference = task["standing_position_reference"]
    if change == "delay_type": reference["capture_delay_seconds"] = True
    elif change == "delay_bound": reference["capture_delay_seconds"] = -.01
    elif change == "reset_type": reference["reset_on_severe_tilt"] = "false"
    elif change == "extra_field": reference["unknown"] = 1
    elif change == "missing_field": del reference["capture_delay_seconds"]
    else: task["standing_position_observation"]["capture_delay_seconds"] = .2
    with pytest.raises(ValueError):
        validate_observation_contract(task)

"""Observation revisions preserve old replay and reject undeclared timing changes."""
from copy import deepcopy

import pytest
import torch

from wheeled_algo.standing_displacement import StandingDisplacement, observation_contract as v1_contract
from wheeled_algo.standing_displacement_contract import (
    observation_contract, observer_from_settings, validate_observation_contract,
)


def settings(seconds=None):
    result = dict(quiet_velocity_feedback=True, quiet_velocity_feedback_mode="standing_displacement",
                  standing_position_observation=observation_contract(seconds))
    if seconds is not None:
        result["standing_position_filter_seconds"] = seconds
    return result


def test_legacy_default_replays_same_sequence_bitwise_with_partial_resets():
    assert observation_contract() == v1_contract()
    old, current = StandingDisplacement(3, "cpu"), observer_from_settings(settings(), 3, "cpu")
    rng = torch.Generator().manual_seed(612)
    for tick in range(300):
        value = torch.randn(3, 2, generator=rng) * .025 + tick * .0001
        masks = [torch.rand(3, generator=rng) > threshold for threshold in (.05, .02, .2)]
        if tick % 17 == 0:
            old.reset(torch.tensor([1]))
            current.reset(torch.tensor([1]))
        a, a_active = old.update(value, *masks, tick)
        b, b_active = current.update(value, *masks, tick)
        assert torch.equal(a, b) and torch.equal(a_active, b_active)
        assert torch.equal(old.anchor, current.anchor)
    assert observer_from_settings({}, 3, "cpu") is None


def test_declared_150ms_filter_has_expected_step_response_and_no_extra_tick_updates():
    fast = observer_from_settings(settings(), 1, "cpu")
    slow = observer_from_settings(settings(.15), 1, "cpu")
    yes = torch.ones(1, dtype=torch.bool)
    zero = torch.zeros(1, 2)
    for model in (fast, slow):
        model.update(zero, yes, yes, yes, 0)
    for tick in range(1, 101):
        a, _ = fast.update(zero + .04, yes, yes, yes, tick)
        b, _ = slow.update(zero + .04, yes, yes, yes, tick)
        repeat, _ = slow.update(zero + .04, yes, yes, yes, tick)
        assert torch.equal(b, repeat)
        expected = .04 * (1. - (1. - .005 / .155) ** tick)
        torch.testing.assert_close(b, torch.full((1, 2), expected), atol=2e-8, rtol=1e-6)
        assert (b <= a).all()
    assert b.min() > .038


@pytest.mark.parametrize("seconds", [None, True, "0.15", float("nan"), float("inf"), .049, .201])
def test_bad_explicit_filter_setting_fails_before_environment_or_actor(seconds):
    task = settings(.15)
    task["standing_position_filter_seconds"] = seconds
    with pytest.raises(ValueError, match="filter"):
        observer_from_settings(task, 1, "cpu")


@pytest.mark.parametrize("change", ["missing_setting", "wrong_contract_time", "wrong_schema", "extra_field", "wrong_mode"])
def test_timing_and_contract_must_be_migrated_together(change):
    task = settings(.15)
    if change == "missing_setting":
        del task["standing_position_filter_seconds"]
    elif change == "wrong_contract_time":
        task["standing_position_observation"]["filter_seconds"] = .05
    elif change == "wrong_schema":
        task["standing_position_observation"]["schema"] = "wheel-leg-standing-displacement-v1"
    elif change == "extra_field":
        task["standing_position_observation"]["unverified"] = True
    else:
        task["quiet_velocity_feedback_mode"] = "rolling_velocity"
    with pytest.raises(ValueError):
        validate_observation_contract(task)
    correct = deepcopy(settings(.15))
    assert validate_observation_contract(correct)

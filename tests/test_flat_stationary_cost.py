"""Position cost distinguishes drift without a cheap contact/failure escape."""
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.rewards import StationaryAnchor
from wheeled_tasks.chassis.flat_base_continuation import read_plan


def test_square_cost_keeps_precision_and_distinguishes_large_drift():
    d = torch.tensor([.03, .1, .2, .3, 1.])
    active = torch.ones(5, dtype=torch.bool)
    model = StationaryAnchor(5, "cpu", 1., .1, cost="capped_square", cap=25.)
    model.reward(torch.zeros(5, 2), active, active)
    positions = torch.stack((d, torch.zeros(5)), -1)
    torch.testing.assert_close(model.reward(positions, active, active), torch.tensor([-.09, -1., -4., -9., -25.]))
    # Losing support cannot turn a latched displacement cost into zero.
    torch.testing.assert_close(model.reward(positions, active, ~active), torch.tensor([-.09, -1., -4., -9., -25.]))
    model.reset(torch.tensor([0, 1]))
    assert model.reward(positions, active, active)[:2].eq(0).all()


def test_static_failure_is_one_event_and_moving_lanes_are_unchanged():
    model = StationaryAnchor(4, "cpu", 1., .1, cost="capped_square", failure_cost=50.)
    dead = torch.tensor([True, False, True, True])
    stationary = torch.tensor([True, True, False, True])
    assert model.failure_penalty(dead, stationary).tolist() == [-50., 0., 0., -50.]
    legacy = StationaryAnchor(4, "cpu", 1., .1)
    assert not legacy.failure_penalty(dead, stationary).count_nonzero()


def test_plan_rejects_a_cheaper_early_failure(tmp_path):
    root = Path(__file__).resolve().parents[1]
    plan = read_plan(root, "contracts/flat_base_finetune_v8.json")
    plan["stationary_tracking"]["failure_event_cost"] = 1.
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="cheaper"):
        read_plan(root, path)


@pytest.mark.parametrize("kwargs", [dict(cost="typo"), dict(cap=0.), dict(failure_cost=float("nan"))])
def test_bad_cost_contract_rejects(kwargs):
    with pytest.raises(ValueError):
        StationaryAnchor(1, "cpu", 1., .1, **kwargs)

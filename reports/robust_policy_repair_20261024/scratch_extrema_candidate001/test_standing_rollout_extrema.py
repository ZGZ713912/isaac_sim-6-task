"""Tests of the unapplied next-version observer, without importing live code."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("candidate_standing_rollout_extrema", Path(__file__).with_name("standing_rollout_extrema.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Writer:
    def __init__(self):
        self.rows = {}

    def add_scalar(self, name, value, step):
        self.rows[name] = (value, step)


def test_first_substep50_rad_s_survives128_step_window_without_rng_or_input_changes():
    observer = module.StandingRolloutExtrema(2, "cpu")
    rng = torch.get_rng_state().clone()
    for step in range(128):
        dq = torch.zeros(5, 2, 4)
        if step == 0:
            dq[0, 0, 3] = -50.
        original = dq.clone()
        observer.observe(dq)
        assert torch.equal(dq, original)
    assert torch.equal(torch.get_rng_state(), rng)
    writer = Writer()
    values = observer.write(writer, completed_update=1, expected_calls=128)
    assert values["peak_leg_speed_rad_s"] == values["peak_leg3_speed_rad_s"] == 50.
    assert values["observed_leg_motor_samples"] == 128 * 5 * 2 * 4
    assert all(values[f"leg_motor_samples_at_or_above_{threshold}_rad_s"] == 1 for threshold in (3, 10, 50))
    assert writer.rows["ScratchStandingExtrema/peak_leg_speed_rad_s"] == (50., 0)
    assert observer.policy_calls == 0
    observer.observe(torch.zeros(5, 2, 4))
    assert observer.snapshot(1)["peak_leg_speed_rad_s"] == 0.


def test_each_leg_max_and_threshold_counts_include_all_physical_samples():
    observer = module.StandingRolloutExtrema(1, "cpu")
    dq = torch.tensor([3., -10., 49., -51.]).expand(5, 1, 4)
    observer.observe(dq)
    values = observer.snapshot(1)
    assert [values[f"peak_leg{i}_speed_rad_s"] for i in range(4)] == [3., 10., 49., 51.]
    assert [values[f"leg_motor_samples_at_or_above_{t}_rad_s"] for t in (3, 10, 50)] == [20, 15, 5]


@pytest.mark.parametrize("shape", [(1, 2, 4), (5, 1, 4), (5, 2, 6)])
def test_missing_physics_or_wrong_population_is_rejected(shape):
    observer = module.StandingRolloutExtrema(2, "cpu")
    with pytest.raises(ValueError, match="all five"):
        observer.observe(torch.zeros(shape))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_extrema_cannot_be_logged_as_a_valid_zero(value):
    observer = module.StandingRolloutExtrema(1, "cpu")
    dq = torch.zeros(5, 1, 4)
    dq[0, 0, 0] = value
    observer.observe(dq)
    with pytest.raises(ValueError, match="Nonfinite"):
        observer.snapshot(1)


def test_incomplete_window_or_missing_writer_cannot_reset_the_accumulator():
    observer = module.StandingRolloutExtrema(1, "cpu")
    observer.observe(torch.ones(5, 1, 4))
    with pytest.raises(ValueError, match="entire completed"):
        observer.write(Writer(), 1, 2)
    with pytest.raises(ValueError, match="actual RSL writer"):
        observer.write(None, 1, 1)
    assert observer.policy_calls == 1

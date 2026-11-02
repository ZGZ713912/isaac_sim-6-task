"""Optional geometry diagnostics must not erase finite-state or inertia validity."""
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wheeled_tasks.wheelbipe_port.training_contract import constraint_limit_m,check_constraint_gap,leg_armature_kg_m2,projection_penalty


def test_projection_objective_has_explicit_historical_zero_control():
    assert projection_penalty(1.)==1.
    assert projection_penalty(0.)==0.
    for value in (-1.,math.inf,math.nan):
        with pytest.raises(ValueError):
            projection_penalty(value)


def test_user_removed_gap_gate_but_keeps_finite_diagnostics():
    assert constraint_limit_m(None) is None
    check_constraint_gap(.032897,None,"diagnostic only")
    with pytest.raises(ValueError):
        check_constraint_gap(math.nan,None,"diagnostic only")


def test_per_axis_armature_is_in_output_shaft_units():
    assert leg_armature_kg_m2(.015795)==.015795
    assert leg_armature_kg_m2(0.)==0.


@pytest.mark.parametrize("invalid",[-1.,math.inf,math.nan])
def test_invalid_armature_is_rejected(invalid):
    with pytest.raises(ValueError):
        leg_armature_kg_m2(invalid)


def test_five_millimetre_boundary_and_historical_limit_are_distinct():
    revised = constraint_limit_m(5.)
    assert revised==.005
    check_constraint_gap(.004,revised,"PPO")
    check_constraint_gap(.005,revised,"PPO")
    with pytest.raises(ValueError):
        check_constraint_gap(math.nextafter(.005,math.inf),revised,"PPO")
    with pytest.raises(ValueError):
        check_constraint_gap(.004,constraint_limit_m(3.),"historical PPO")


@pytest.mark.parametrize("invalid",[0.,-5.,math.inf,math.nan])
def test_invalid_admission_limit_cannot_disable_the_check(invalid):
    with pytest.raises(ValueError):
        constraint_limit_m(invalid)


@pytest.mark.parametrize("gap",[math.inf,-math.inf,math.nan])
def test_nonfinite_measured_gap_is_rejected(gap):
    with pytest.raises(ValueError):
        check_constraint_gap(gap,.005,"PPO")

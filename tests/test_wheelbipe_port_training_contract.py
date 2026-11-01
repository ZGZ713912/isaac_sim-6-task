"""The revised millimetre limit must not hide nonfinite or excessive gaps."""
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wheeled_tasks.wheelbipe_port.training_contract import constraint_limit_m,check_constraint_gap


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

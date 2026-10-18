import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("local_actuation_probe",
    Path(__file__).parents[1] / "scripts/diagnose_local_actuation.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_bounded_pulse_preserves_other_axes_and_common_prefix():
    for axis in range(6):
        for sign in (-1, 0, 1):
            values = [probe.intervention(tick, axis, sign) for tick in range(600)]
            assert all(row == [0.] * 6 for row in values[:200] + values[420:])
            assert values[200][axis] == pytest.approx(.005 * sign)
            assert values[219][axis] == values[399][axis] == pytest.approx(.1 * sign)
            assert values[419][axis] == pytest.approx(.005 * sign)
            assert all(abs(row[axis]) <= .1 and
                all(value == 0. for j, value in enumerate(row) if j != axis) for row in values)
            assert max(abs(values[t + 1][axis] - values[t][axis]) for t in range(599)) <= .00500000000001


@pytest.mark.parametrize("tick,axis,sign", [(-1, 0, 1), (0, 6, 1), (0, 0, 2)])
def test_invalid_intervention_fails_before_native_start(tick, axis, sign):
    with pytest.raises(ValueError):
        probe.intervention(tick, axis, sign)

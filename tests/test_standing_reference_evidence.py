"""Qualification rejects stale anchors, invalid recapture and bad counters."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[1] / "reports/unified_robust_20261021/iteration065_analyze.py"
spec = importlib.util.spec_from_file_location("standing_reference_evidence065", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def trace():
    obs = np.zeros((4, 35), dtype=np.float32)
    obs[:, 3] = .305 * 5.
    obs[:, 9] = [-1., 1., 1., -1.]
    return dict(obs=obs, active=np.array([True, False, False, True]), count=np.array([0, 1, 1, 1]),
                age=np.array([1., 0., 0., .05]), filtered=np.zeros((4, 2)),
                anchor=np.array([[1., 1.], [0., 0.], [0., 0.], [100., 100.]]),
                alive=np.ones(4, dtype=bool), task=dict(stance_aware_recovery=True, recovery_shape_aware=True))


def test_valid_reference_trace_allows_fresh_anchor_after_air_spin():
    assert module.reference_trace_violations(**trace()) == []


@pytest.mark.parametrize("bad", ["active_severe", "stale_severe_anchor", "stale_filter", "early_recapture",
                                 "nan_recapture", "moving_recapture", "counter_jump"])
def test_invalid_native_evidence_cannot_pass_reference_qualification(bad):
    data = trace()
    if bad == "active_severe": data["active"][1] = True
    elif bad == "stale_severe_anchor": data["anchor"][1] = 5.
    elif bad == "stale_filter": data["filtered"][-1] = .01
    elif bad == "early_recapture": data["age"][-1] = .045
    elif bad == "nan_recapture": data["obs"][-1, 9] = float("nan")
    elif bad == "moving_recapture": data["obs"][-1, 0] = .5
    else: data["count"][1:] = 2
    assert module.reference_trace_violations(**data)

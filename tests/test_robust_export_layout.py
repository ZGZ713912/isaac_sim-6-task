"""Reject misleading comparisons caused by different native placements."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("robust_export_layout", ROOT / "scripts/export_robust_getup.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report():
    return {"lanes": [dict(name="arc", pose_index=29, command=[1., 1., .305],
                            recovery=False, world_origin_m=[-216., 194., 0.])]}


def test_same_cases_at_same_recorded_native_origins_pass():
    a = report()
    assert MODULE.verify_native_pairing(a, deepcopy(a))["matched_native_origins"]


def test_large_native_jobs_are_refused_before_simulation():
    from wheeled_tasks.chassis.getup_ground import require_bounded_world, world_layout_bounds
    layout = require_bounded_world(280)
    assert layout["max_abs_origin_xy_m"] == [810., 824.5]
    assert max(layout["max_abs_support_xy_m"]) < 1024.
    assert max(world_layout_bounds(560)["max_abs_origin_xy_m"]) > 1024.
    for count in (560, 1024):
        with pytest.raises(ValueError, match="world-coordinate"):
            require_bounded_world(count)


@pytest.mark.parametrize("change", ["origin", "missing", "command", "pose", "nonfinite", "large"])
def test_wrong_native_placement_or_case_is_refused(change):
    a, b = report(), report()
    if change == "origin":
        b["lanes"][0]["world_origin_m"][0] += 108.
    elif change == "missing":
        del b["lanes"][0]["world_origin_m"]
    elif change == "command":
        b["lanes"][0]["command"][1] = -1.
    elif change == "pose":
        b["lanes"][0]["pose_index"] = 30
    elif change == "nonfinite":
        a["lanes"][0]["world_origin_m"][0] = float("nan")
    else:
        a["lanes"][0]["world_origin_m"][0] = 1080.
        b = deepcopy(a)
    with pytest.raises(ValueError):
        MODULE.verify_native_pairing(a, b)

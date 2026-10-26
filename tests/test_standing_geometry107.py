"""The extension preserves old observations and guards new physical limits."""
from copy import deepcopy
import json

import pytest
import torch

from wheeled_algo.standing_reference_lifecycle import observation_contract
from wheeled_algo.rolling_velocity import frozen_table_path
from wheeled_tasks.chassis.standing_geometry107 import (
    coordinate_from_settings, extend_settings, table_path, validate_observation_contract,
)


def settings():
    return dict(quiet_velocity_feedback=True, quiet_velocity_feedback_mode="standing_displacement",
                standing_position_reference=dict(capture_delay_seconds=.05, reset_on_severe_tilt=True),
                standing_position_filter_seconds=.15,
                standing_position_observation=observation_contract(.15, .05, True))


def test_every_original_knot_retained():
    old, new = [json.loads(p.read_text()) for p in (frozen_table_path(), table_path())]
    for side in ("left", "right"):
        for key in ("delta_rad", "wheel_center_b_m", "passive_inner_deg"):
            values = new["sides"][side][key]
            assert (values[:-16] if side == "left" else values[16:]) == old["sides"][side][key]
        assert new["sides"][side]["validation"]["max_error_mm"] < .1


def test_old_domain_bitwise_equal_and_new_domain_guarded():
    a, b = coordinate_from_settings(settings()), coordinate_from_settings(extend_settings(settings()))
    torch.manual_seed(9)
    q = torch.randn(512, 6)
    q[:, 1] = q[:, 0] + torch.linspace(0., 1.32, 512)
    q[:, 3] = q[:, 2] - torch.linspace(0., 1.32, 512)
    gravity = torch.tensor([0., 0., -1.]).expand(512, -1)
    ca, va = a(q, gravity)
    cb, vb = b(q, gravity)
    assert va.all() and torch.equal(va, vb) and torch.equal(ca, cb)
    q[:, 1] = q[:, 0] + 1.35
    q[:, 3] = q[:, 2] - 1.35
    assert not a(q, gravity)[1].any()
    assert b(q, gravity)[1].all()
    for shift in (1.371, -.142):
        q[:, 1] = q[:, 0] + shift
        q[:, 3] = q[:, 2] - shift
        result, valid = b(q, gravity)
        assert not valid.any() and not result.any()
    assert not b(q, -gravity)[1].any()


@pytest.mark.parametrize("key,value", [("geometry_table_sha256", "wrong"),
                                      ("passive_knee_domain_deg", [40., 108.]),
                                      ("original_knots_unchanged", False)])
def test_contract_tampering_rejected(key, value):
    task = deepcopy(extend_settings(settings()))
    task["standing_position_observation"][key] = value
    with pytest.raises(ValueError):
        validate_observation_contract(task)

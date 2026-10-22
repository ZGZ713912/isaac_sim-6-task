"""Model identity, basic rolling units and invalid-geometry fallback."""
from pathlib import Path
import json

import pytest
import torch

from wheeled_algo.rolling_velocity import RollingBaseVelocity

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "reports/flat_base_continuation_20261019/low_friction_candidate001/height_lookup.json"


def test_straight_wheel_rotation_and_common_velocity_units():
    model = RollingBaseVelocity(json.loads(TABLE.read_text()))
    x = torch.zeros(5, 35)
    x[:, 9] = -1.
    x[:, 20] = 1.
    x[:, 21] = -1.
    velocity, valid = model(x)
    assert valid.all()
    torch.testing.assert_close(velocity, torch.full((5, 1), .6), rtol=0., atol=1e-5)
    x[:, 20:22] = 0.
    assert model(x)[0].count_nonzero() == 0


def test_invalid_knee_domain_and_degenerate_gravity_disable_estimate():
    model = RollingBaseVelocity(json.loads(TABLE.read_text()))
    x = torch.zeros(2, 35)
    x[:, 9] = -1.
    x[0, 11] = 3.
    x[1, 7:10] = 0.
    velocity, valid = model(x)
    assert not valid.any() and velocity.count_nonzero() == 0


def test_different_asset_and_nonfinite_geometry_rejected():
    table = json.loads(TABLE.read_text())
    table["model_spec_sha256"] = "other"
    with pytest.raises(ValueError, match="authenticated107"):
        RollingBaseVelocity(table)
    table = json.loads(TABLE.read_text())
    table["sides"]["left"]["hip_axis_b"][0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        RollingBaseVelocity(table)

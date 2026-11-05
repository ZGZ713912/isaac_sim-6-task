import numpy as np
import pytest
import json
from pathlib import Path
from scipy.spatial import ConvexHull

from wheeled_tasks.chassis.collision_geometry import (authenticate, partition, restore_envelope,
    selected_asset, separated_support_parts, support_directions, vertices_inside)


def prism():
    return np.array([[x, y, z] for y in [-.3, .3]
                     for x, z in [(-.1, -.1), (.1, -.1), (-.2, .1), (.2, .1)]])


def test_restores_slope_instead_of_original_axis_envelope():
    raw = prism()
    box = np.array([[x, y, z] for x in [-.2, .2] for y in [-.3, .3] for z in [-.1, .1]])
    extremes, parts = restore_envelope(raw, [box], limit=8)
    directions = np.random.default_rng(42).normal(size=(4096, 3))
    expected = (raw @ directions.T).max(0)
    actual = (np.concatenate(parts) @ directions.T).max(0)
    assert np.allclose(actual, expected, atol=1e-10)
    assert ((box @ directions.T).max(0) - expected).max() > .05
    assert max(len(p) for p in parts) <= 8
    assert vertices_inside(np.concatenate(parts), ConvexHull(raw).equations).max() < 1e-10


def test_refuses_source_extreme_missing_from_parent_pieces():
    raw = prism()
    with pytest.raises(ValueError, match="do not cover"):
        restore_envelope(raw, [raw * .5])


def test_partition_preserves_many_source_extremes_with_small_cooking_budget():
    rng = np.random.default_rng(25)
    raw = rng.normal(size=(80, 3))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    box = np.array([[x, y, z] for x in [-1., 1.] for y in [-1., 1.] for z in [-1., 1.]])
    extremes, parts = restore_envelope(raw, [box], limit=20)
    assert len(parts) > 1 and max(len(p) for p in parts) <= 20
    covered = np.stack([vertices_inside(extremes, ConvexHull(p).equations) for p in parts]).min(0)
    assert covered.max() < 1e-8
    weights = rng.exponential(size=(256, len(raw)))
    interior = weights @ raw / weights.sum(1, keepdims=True)
    assert np.stack([vertices_inside(interior, ConvexHull(p).equations) for p in parts]).min(0).max() < 1e-8


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "contracts/wheel_leg_rollover_chassis_slopes_v3.json"


def test_asset_selection_is_scoped_and_rejects_external_override(monkeypatch):
    config = json.loads(PLAN.read_text())["chassis_collision"]
    monkeypatch.delenv("WHEELED_ASSET_USD", raising=False)
    import os
    with pytest.raises(RuntimeError):
        with selected_asset(ROOT, config):
            assert os.environ["WHEELED_ASSET_USD"] == str(ROOT/config["usd"]["file"])
            raise RuntimeError("construction failed")
    assert "WHEELED_ASSET_USD" not in os.environ
    monkeypatch.setenv("WHEELED_ASSET_USD", "unbound.usda")
    with pytest.raises(ValueError, match="external asset"):
        with selected_asset(ROOT, config):
            pass
    assert os.environ["WHEELED_ASSET_USD"] == "unbound.usda"


def test_corrected_contract_changes_only_asset_selection():
    original = json.loads((ROOT/"contracts/wheel_leg_rollover_learning_v1.json").read_text())
    corrected = json.loads(PLAN.read_text())
    config = corrected.pop("chassis_collision")
    assert original == corrected
    geometry = authenticate(ROOT, config)
    source = np.array(geometry["source_extremes"])
    points = np.concatenate([np.array(p["vertices"]) for p in geometry["pieces"]])
    dirs = support_directions(source)
    assert np.abs((source@dirs.T).max(0)-(points@dirs.T).max(0)).max() < 1e-6
    assert all(len(p["vertices"]) <= 64 for p in geometry["pieces"])
    assert len(geometry["disabled_parent_paths"]) == 8


def test_support_parts_keep_all_extremes_with_thick_interior_and_no_false_extent():
    theta = np.linspace(0., 2.*np.pi, 400, endpoint=False)
    raw = np.array([[.3*np.cos(t), .3*np.sin(t), z] for t in theta for z in (-.1,.1)])
    source, parts = separated_support_parts(raw)
    directions = support_directions(source)
    assert np.abs((np.concatenate(parts)@directions.T).max(0)-(source@directions.T).max(0)).max() < 1e-9
    planes = ConvexHull(raw).equations
    assert max(len(p) for p in parts) <= 64
    for points in parts:
        assert vertices_inside(points, planes).max() < 1e-10
        assert ConvexHull(points).volume > 1e-5


def test_actual_usd_only_changes_chassis_collisions():
    from pxr import Usd, UsdGeom, UsdPhysics
    config = json.loads(PLAN.read_text())["chassis_collision"]
    geometry = authenticate(ROOT, config)
    old = Usd.Stage.Open(str(ROOT/geometry["parent_usd"]))
    new = Usd.Stage.Open(str(ROOT/config["usd"]["file"]))
    disabled = set(geometry["disabled_parent_paths"])
    for prim in old.Traverse():
        path = str(prim.GetPath())
        actual = new.GetPrimAtPath(path)
        assert actual.GetTypeName() == prim.GetTypeName()
        for attr in prim.GetAttributes():
            if path in disabled and attr.GetName() == "physics:collisionEnabled":
                assert actual.GetAttribute(attr.GetName()).Get() is False
            else:
                assert actual.GetAttribute(attr.GetName()).Get() == attr.Get(), (path, attr.GetName())
    for item in geometry["pieces"]:
        prim = new.GetPrimAtPath(item["path"])
        assert prim.GetParent().GetPath() == "/Robot/base_link"
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        assert np.array_equal(np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get()), np.array(item["vertices"]))
        assert abs(prim.GetAttribute("physxConvexHullCollision:minThickness").Get()-1e-5) < 1e-10
    extras = {str(p.GetPath()) for p in new.Traverse()}-{str(p.GetPath()) for p in old.Traverse()}
    assert extras == {p["path"] for p in geometry["pieces"]}

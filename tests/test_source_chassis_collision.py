"""Shared CAD support, all export variants, and stale-override rejection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull
import trimesh

from wheeled_tasks.chassis.source_chassis_collision import load, selected_usd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "model/wheel_leg_v2.0.2"


def test_hulls_retain_the_original_slopes_and_all_support_extremes():
    data = load()
    source = ROOT / data["source_stl"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == data["source_stl_sha256"]
    raw = trimesh.load(source, force="mesh").vertices
    # Canonical chassis coordinates re-express the CAD visual by +90 degrees Z.
    raw = np.column_stack((-raw[:, 1], raw[:, 0], raw[:, 2]))
    hull = ConvexHull(raw)
    directions = np.vstack((hull.equations[:, :3], np.eye(3), -np.eye(3)))
    points = np.concatenate([p["vertices"] for p in data["pieces"]])
    np.testing.assert_allclose((points @ directions.T).max(0),
                               (raw[hull.vertices] @ directions.T).max(0), atol=1e-7, rtol=0)
    assert max(len(p["vertices"]) for p in data["pieces"]) <= 64
    assert max((points @ hull.equations[:, :3].T + hull.equations[:, 3]).max(1)) < 1e-7


@pytest.mark.parametrize("variant", ["isaac", "training", "friction_v1/isaac", "friction_v1/training"])
def test_every_default_export_uses_only_source_support_hulls(variant, monkeypatch):
    directory = BASE / variant
    if not directory.exists():
        pytest.skip("This historical branch predates the training derivative")
    manifest = json.loads((directory / "manifest.json").read_text())
    monkeypatch.delenv("WHEELED_ASSET_USD", raising=False)
    path = selected_usd(ROOT, dict(asset_directory=str(directory.relative_to(ROOT))), manifest)
    stage = Usd.Stage.Open(path)
    body = stage.GetPrimAtPath("/Robot/base_link")
    children = list(body.GetChildren())
    assert len([p for p in children if p.HasAPI(UsdPhysics.CollisionAPI)]) == 42
    assert not any(p.GetName().startswith("Collision_") for p in children)
    spec = json.loads((directory / "model_spec.json").read_text())
    chassis = next(b for b in spec["bodies"] if b["name"] == "base_link")
    assert len(chassis["collisions"]) == 42
    for collision, piece in zip(chassis["collisions"], load()["pieces"]):
        obj = trimesh.load(directory / collision["file"], force="mesh", process=False)
        directions = ConvexHull(np.asarray(piece["vertices"])).equations[:, :3]
        np.testing.assert_allclose((obj.vertices @ directions.T).max(0),
            (np.asarray(piece["vertices"]) @ directions.T).max(0), atol=1e-12, rtol=0)
        np.testing.assert_array_equal(collision["origin"], np.eye(4))
    assert not list(directory.glob("collisions/base_link_0*.obj"))


def test_explicit_box_override_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "obsolete.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/Robot/base_link")
    cube = UsdGeom.Cube.Define(stage, "/Robot/base_link/Box")
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
    monkeypatch.setenv("WHEELED_ASSET_USD", str(path))
    with pytest.raises(ValueError, match="requires source-support-v3"):
        selected_usd(ROOT, {}, dict(model_kind="full_robot_v202_gas_spring_closedchain"))


def test_no_retired_primitive_or_slope_variant_remains():
    assert not list(BASE.rglob("robot_prim*.usda"))
    assert not list(BASE.glob("chassis_slopes_*"))

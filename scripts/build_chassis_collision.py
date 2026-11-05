#!/usr/bin/env python3
"""Build a source-slope-preserving base_link overlay; never edit its parent."""
from pathlib import Path
import argparse
import json
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
from scipy.spatial import ConvexHull

from wheeled_tasks.chassis.collision_geometry import SCHEMA, VERTEX_LIMIT, digest, restore_envelope


ROOT = Path(__file__).resolve().parents[1]
PARENT = "model/wheel_leg_v2.0.2/friction_v1/training/robot.usda"
OUTPUT = "model/wheel_leg_v2.0.2/chassis_slopes_v2"
MIN_THICKNESS_M = .00001


def mesh_points(stage, prim, body):
    matrix = np.array(UsdGeom.XformCache().ComputeRelativeTransform(prim, body)[0])
    points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
    return np.column_stack((points, np.ones(len(points)))) @ matrix[:, :3]


def build(output):
    if output.exists():
        raise FileExistsError("Never overwrite an existing generated asset")
    stage = Usd.Stage.Open(str(ROOT / PARENT))
    body = stage.GetPrimAtPath("/Robot/base_link")
    visual = stage.GetPrimAtPath("/Robot/base_link/Visual")
    source = np.unique(mesh_points(stage, visual, body), axis=0)
    old = [p for p in stage.Traverse() if p.GetParent() == body and p.HasAPI(UsdPhysics.CollisionAPI)]
    if len(old) != 8 or any(not p.IsA(UsdGeom.Mesh) for p in old):
        raise ValueError("Unexpected original chassis partition")
    extremes, parts = restore_envelope(source, [mesh_points(stage, p, body) for p in old])
    output.mkdir(parents=True)
    asset = output / "robot.usda"
    overlay = Usd.Stage.CreateNew(str(asset))
    overlay.GetRootLayer().subLayerPaths = [os.path.relpath(ROOT / PARENT, output)]
    overlay.SetDefaultPrim(overlay.GetPrimAtPath("/Robot"))
    for prim in old:
        UsdPhysics.CollisionAPI(overlay.OverridePrim(prim.GetPath())).CreateCollisionEnabledAttr(False)
    rows = []
    for index, points in enumerate(parts):
        points = points.astype(np.float32).astype(float)
        hull = ConvexHull(points)
        triangles = hull.simplices.copy()
        # Qhull simplex ordering is not an outward winding guarantee.
        for face, equation in zip(triangles, hull.equations):
            a, b, c = points[face]
            if np.dot(np.cross(b-a, c-a), equation[:3]) < 0.:
                face[1], face[2] = face[2], face[1]
        mesh = UsdGeom.Mesh.Define(overlay, f"/Robot/base_link/SourceCollision_{index:03d}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
        mesh.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        mesh.GetPrim().AddAppliedSchema("PhysxConvexHullCollisionAPI")
        mesh.GetPrim().CreateAttribute("physxConvexHullCollision:hullVertexLimit", Sdf.ValueTypeNames.Int).Set(VERTEX_LIMIT)
        mesh.GetPrim().CreateAttribute("physxConvexHullCollision:minThickness", Sdf.ValueTypeNames.Float).Set(MIN_THICKNESS_M)
        rows.append(dict(path=str(mesh.GetPath()), vertices=points.tolist(), faces=triangles.tolist()))
    overlay.GetRootLayer().Save()
    parents = [Path(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath]
    for extra in ["model/wheel_leg_v2.0.2/friction_v1/isaac/meshes/base_link.stl",
                  "model/wheel_leg_v2.0.2/friction_v1/training/model_spec.json"]:
        parents.append(ROOT / extra)
    geometry = dict(schema=SCHEMA, scope="exact base_link supporting envelope on an infinite flat ground plane; concavity/terrain contacts unqualified",
        method="intersect each old piece with original visual convex hull, then group boundary-to-interior tetrahedra",
        parent_usd=PARENT, body="base_link", disabled_parent_paths=[str(p.GetPath()) for p in old],
        source_unique_points=len(source), source_extremes=extremes.tolist(),
        cooking_vertex_limit=VERTEX_LIMIT, cooking_min_thickness_m=MIN_THICKNESS_M, pieces=rows,
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(parents))})
    (output / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), pieces=len(rows), max_vertices=max(len(p) for p in parts),
        usd_sha256=digest(asset), geometry_sha256=digest(output / "geometry.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    build(parser.parse_args().output)

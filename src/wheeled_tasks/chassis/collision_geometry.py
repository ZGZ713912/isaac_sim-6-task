"""Convex contact envelopes derived from the original chassis surface.

Flat-ground support is the minimum of a linear functional over vertices.
Intersecting the old pieces with the source convex hull removes false exterior
contacts while retaining the old interior partition. Every source extreme must
remain covered. This does not certify contacts inside concavities on terrain.
"""
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection


SCHEMA = "wheel-leg-source-chassis-contact-envelope-v1"
TOLERANCE_M = 1e-6
VERTEX_LIMIT = 64


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vertices_inside(points, planes):
    return (np.asarray(points) @ planes[:, :3].T + planes[:, 3]).max(axis=1)


def intersect(planes):
    planes = np.unique(np.round(np.asarray(planes, dtype=float), 12), axis=0)
    result = linprog([0., 0., 0., -1.],
        A_ub=np.column_stack((planes[:, :3], np.ones(len(planes)))),
        b_ub=-planes[:, 3], bounds=[(None, None)] * 3 + [(0., None)], method="highs")
    if not result.success or result.x[3] <= 1e-9:
        raise ValueError("Collision partition has no volumetric strict interior")
    points = HalfspaceIntersection(planes, result.x[:3]).intersections
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    return vertices[np.lexsort(vertices.T[::-1])]


def partition(points, limit=VERTEX_LIMIT):
    hull = ConvexHull(points)
    points = np.asarray(points)[hull.vertices]
    if len(points) <= limit:
        return [points]
    hull = ConvexHull(points)
    center = points.mean(0)
    remaining = set(range(len(hull.simplices)))
    result = []
    # A convex body is the union of the tetrahedra joining each boundary
    # triangle to an interior point. Grouping adjacent tetrahedra into convex
    # pieces stays inside the body and cannot lose any of those tetrahedra.
    while remaining:
        vertices, frontier = set(), {min(remaining)}
        while frontier:
            candidates = [i for i in frontier if len(vertices | set(hull.simplices[i])) < limit]
            if not candidates:
                break
            face = min(candidates, key=lambda i: (len(set(hull.simplices[i]) - vertices), i))
            vertices.update(hull.simplices[face])
            remaining.remove(face)
            frontier.remove(face)
            frontier.update(int(i) for i in hull.neighbors[face] if i in remaining)
        if not vertices:
            raise ValueError("Collision cooking budget cannot fit a tetrahedron")
        result.append(np.vstack((points[sorted(vertices)], center)))
    return result


def restore_envelope(source_points, old_pieces, limit=VERTEX_LIMIT):
    source_hull = ConvexHull(source_points)
    extremes = np.asarray(source_points)[source_hull.vertices]
    old_planes = [ConvexHull(points).equations for points in old_pieces]
    covered = np.stack([vertices_inside(extremes, plane) for plane in old_planes]).min(axis=0)
    if covered.max() > 1e-7:
        raise ValueError("Old pieces do not cover all original chassis extremes")
    parts = []
    for old in old_planes:
        clipped = intersect(np.vstack((source_hull.equations, old)))
        parts.extend(partition(clipped, limit))
    restored = np.concatenate(parts)
    if vertices_inside(restored, source_hull.equations).max() > 1e-7:
        raise ValueError("Restored collision protrudes outside original surface hull")
    covered = np.stack([vertices_inside(extremes, ConvexHull(part).equations)
                        for part in parts]).min(axis=0)
    if covered.max() > 1e-7:
        raise ValueError("Restored pieces lost an original supporting extreme")
    return extremes, parts


def authenticate(root, config):
    if config is None:
        return None
    if set(config) != {"schema", "usd", "geometry"} or config["schema"] != SCHEMA:
        raise ValueError("Invalid source chassis collision contract")
    root = Path(root).resolve()
    for item in (config["usd"], config["geometry"]):
        path = root / item["file"]
        if root not in path.resolve().parents or digest(path) != item["sha256"]:
            raise ValueError("Source chassis collision input changed")
    geometry = json.loads((root / config["geometry"]["file"]).read_text())
    if geometry["schema"] != SCHEMA or geometry["cooking_vertex_limit"] != VERTEX_LIMIT:
        raise ValueError("Collision cooking contract changed")
    for name, expected in geometry["input_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Original collision source changed: " + name)
    return geometry


@contextmanager
def selected_asset(root, config):
    """Use the existing constructor hook only for a hash-authenticated asset."""
    if config is None:
        yield
        return
    authenticate(root, config)
    if os.environ.get("WHEELED_ASSET_USD"):
        raise ValueError("An external asset override cannot replace authenticated chassis slopes")
    os.environ["WHEELED_ASSET_USD"] = str(Path(root) / config["usd"]["file"])
    try:
        yield
    finally:
        os.environ.pop("WHEELED_ASSET_USD", None)


def support_directions(extremes):
    """Source facet normals plus a deterministic sphere, including all axes."""
    count = 4096
    z = 1.-2.*(np.arange(count)+.5)/count
    angle = np.arange(count)*np.pi*(3.-np.sqrt(5.))
    radius = np.sqrt(1.-z*z)
    sphere = np.column_stack((radius*np.cos(angle), radius*np.sin(angle), z))
    return np.vstack((np.eye(3), -np.eye(3), ConvexHull(extremes).equations[:, :3], sphere))


def native_audit(env, config):
    """Read the active USD and PhysX's cooked hulls; do not edit the scene."""
    from pxr import PhysicsSchemaTools, UsdGeom, UsdPhysics, UsdUtils
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult

    geometry = authenticate(env.asset_root, config)
    stage = env.sim.stage
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    cooking = get_physx_cooking_interface()
    checks = dict(old_chassis_disabled=True, all_authored_pieces_exact=True,
                  all_clones_have_exact_active_chassis_set=True, all_cooks_valid=True)
    expected_names = {Path(p["path"]).name for p in geometry["pieces"]}
    cooked, rows = [], []
    for origin in env.env_paths:
        body = stage.GetPrimAtPath(origin+"/Robot/base_link")
        active = {p.GetName() for p in body.GetChildren() if p.HasAPI(UsdPhysics.CollisionAPI)
                  and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()}
        checks["all_clones_have_exact_active_chassis_set"] &= active == expected_names
        for path in geometry["disabled_parent_paths"]:
            prim = stage.GetPrimAtPath(origin+path)
            checks["old_chassis_disabled"] &= not bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
    for item in geometry["pieces"]:
        path = env.env_paths[0]+item["path"]
        prim = stage.GetPrimAtPath(path)
        authored = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        checks["all_authored_pieces_exact"] &= np.array_equal(authored, np.array(item["vertices"]))
        checks["all_authored_pieces_exact"] &= prim.GetAttribute("physxConvexHullCollision:hullVertexLimit").Get() == VERTEX_LIMIT
        received = []
        def on_result(result, convexes):
            if result == PhysxCollisionRepresentationResult.RESULT_VALID:
                received.extend(np.array([[v.x, v.y, v.z] for v in mesh.vertices]) for mesh in convexes)
        cooking.request_convex_collision_representation(stage_id=stage_id,
            collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path),
            run_asynchronously=False, on_result=on_result)
        checks["all_cooks_valid"] &= len(received) == 1 and all(np.isfinite(p).all() for p in received)
        cooked.extend(received)
        rows.append(dict(path=item["path"], authored_vertices=len(authored),
                         cooked_vertices=[p.tolist() for p in received]))
    source = np.array(geometry["source_extremes"])
    directions = support_directions(source)
    error = np.full(len(directions), np.inf)
    if cooked:
        points = np.concatenate(cooked)
        # Check both inward loss and outward false contact, in metres.
        error = (points @ directions.T).max(0)-(source @ directions.T).max(0)
    checks["cooked_flat_ground_support_error_below_0_1mm"] = bool(np.abs(error).max() <= .0001)
    return dict(schema=SCHEMA, all_checks_pass=all(checks.values()), checks=checks,
        selection=config, source_support_scope=geometry["scope"],
        chassis_collision_pieces=len(rows), direction_count=len(directions),
        cooked_support_error_max_m=float(np.abs(error).max()),
        cooked_outward_support_error_m=float(error.max()),
        cooked_inward_support_error_m=float(-error.min()), cooked_pieces=rows,
        reset_bank="original joint/root states unchanged; old clearance labels are not new-asset admission",
        hardware_qualified=False, behavior_qualified=False)

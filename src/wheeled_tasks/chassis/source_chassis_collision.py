"""The v2.0.2 CAD supporting surface, shared by every exported vehicle.

These are the well-conditioned v3 hulls. Each source extreme remains present;
the hulls certify support on flat ground, not the CAD's interior concavities.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GEOMETRY = Path("model/wheel_leg_v2.0.2/chassis_collision/geometry.json")


def load(root=ROOT):
    data = json.loads((Path(root) / GEOMETRY).read_text())
    if (data["schema"] != "wheel-leg-source-chassis-contact-envelope-v1"
            or len(data["pieces"]) != 42 or data["cooking_vertex_limit"] != 64
            or data["cooking_min_thickness_m"] != .00001):
        raise ValueError("Unexpected source chassis collision geometry")
    return data


def mechanical_identity(spec):
    value = deepcopy(spec)
    for body in value["bodies"]:
        if body["name"] == "base_link":
            body.pop("collisions", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def collisions(data):
    identity = [[float(i == j) for j in range(4)] for i in range(4)]
    return [dict(type="mesh", file=f"collisions/base_link_source_{i:04d}.obj", origin=identity)
            for i in range(len(data["pieces"]))]


def write_meshes(bundle, data):
    bundle = Path(bundle)
    (bundle / "collisions").mkdir(exist_ok=True)
    for collision, piece in zip(collisions(data), data["pieces"]):
        rows = ["# v2.0.2 source chassis support; body coordinates"]
        rows += ["v " + " ".join(format(v, ".17g") for v in point) for point in piece["vertices"]]
        rows += ["f " + " ".join(str(v + 1) for v in face) for face in piece["faces"]]
        (bundle / collision["file"]).write_text("\n".join(rows) + "\n")


def author(stage, data):
    """Remove the superseded geometry itself; do not leave disabled hulls."""
    from pxr import Sdf, UsdGeom, UsdPhysics
    body = stage.GetPrimAtPath("/Robot/base_link")
    if not body:
        raise ValueError("Missing v2.0.2 chassis body")
    for child in list(body.GetChildren()):
        if child.HasAPI(UsdPhysics.CollisionAPI) or child.GetName().startswith(("Collision_", "SourceCollision_")):
            stage.RemovePrim(child.GetPath())
    for piece in data["pieces"]:
        mesh = UsdGeom.Mesh.Define(stage, piece["path"])
        mesh.CreatePointsAttr(piece["vertices"])
        mesh.CreateFaceVertexCountsAttr([3] * len(piece["faces"]))
        mesh.CreateFaceVertexIndicesAttr([v for face in piece["faces"] for v in face])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateVisibilityAttr("invisible")
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        prim.AddAppliedSchema("PhysxConvexHullCollisionAPI")
        prim.CreateAttribute("physxConvexHullCollision:hullVertexLimit", Sdf.ValueTypeNames.Int).Set(64)
        prim.CreateAttribute("physxConvexHullCollision:minThickness", Sdf.ValueTypeNames.Float).Set(.00001)


def selected_usd(root, config, manifest):
    """Refuse stale explicit overrides as well as stale default exports."""
    path = Path(os.environ.get("WHEELED_ASSET_USD")
                or Path(root) / config["asset_directory"] / manifest["usd"])
    if manifest.get("model_kind") != "full_robot_v202_gas_spring_closedchain":
        return str(path)
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(path))
    data = load(root)
    body = stage.GetPrimAtPath("/Robot/base_link")
    active = {str(p.GetPath()): p for p in body.GetChildren()
              if p.HasAPI(UsdPhysics.CollisionAPI)
              and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False}
    if set(active) != {p["path"] for p in data["pieces"]}:
        raise ValueError("v2.0.2 requires source-support-v3 chassis collisions: " + str(path))
    for piece in data["pieces"]:
        prim = active[piece["path"]]
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if (not prim.IsA(UsdGeom.Mesh) or [list(p) for p in points] != piece["vertices"]
                or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "convexHull"
                or prim.GetAttribute("physxConvexHullCollision:hullVertexLimit").Get() != 64
                or abs(prim.GetAttribute("physxConvexHullCollision:minThickness").Get() - .00001) > 1e-10
                or UsdGeom.Xformable(prim).GetOrderedXformOps()):
            raise ValueError("Changed source-support-v3 collision: " + str(prim.GetPath()))
    return str(path)

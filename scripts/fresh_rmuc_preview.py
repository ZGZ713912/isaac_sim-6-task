"""Offline RMUC geometry adapter for the single-robot fresh policy viewer.

No simulation is started here. Native playback replaces its single flat static
mesh before PhysX starts, retaining that mesh's material and contact filter.
The existing RMUC visuals are referenced separately with collision disabled.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

EXAMPLE = Path('/home/noir/Documents/workspace/example/wheeled-legged_RL')
DEFAULT_FIELD = EXAMPLE / 'assets/terrains/RMUC2026_V2_gray_fence_revision_03'


def prepare_field(field_root, output, helper=None):
    """Prepare the RMUC floor outside the Kit process.

    Importing pxr here loads pip usd-core before Kit boots. Kit then dlopens its
    own USD libraries and aborts while registering Tf enum names, so no window
    is ever created. The parse runs in a child interpreter and only its files
    come back here.
    """
    output = Path(output)
    helper = Path(helper or EXAMPLE / 'src/agent_tasks/direct/deformable_suspension/field_surface.py').resolve()
    subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--prepare',
                    str(Path(field_root)), str(output), str(helper)], check=True)
    return json.loads((output / 'rmuc_geometry.json').read_text())


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _prepare(field_root, output, helper):
    from pxr import Usd, UsdGeom

    output, helper = Path(output), Path(helper).resolve()
    spec = importlib.util.spec_from_file_location('fresh_rmuc_surface', helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    root, entry, package_sha = module.resolve_field_asset(field_root)
    stage = Usd.Stage.Open(str(entry))
    if UsdGeom.GetStageUpAxis(stage) != 'Z' or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
        raise ValueError('RMUC playback needs metre, Z-up geometry')
    surface = module.FieldSurface.from_stage(stage, str(stage.GetDefaultPrim().GetPath()), 'cpu')
    # This only selects a clear patch of main floor. Robot reset poses and their
    # exact closed-chain geometry still come from the authenticated pose bank.
    spawn = module.select_field_spawn(surface, [[0., .2, -.245], [0., -.2, -.245]], .06)
    origin, rotation = np.asarray(spawn['origin']), np.asarray(spawn['rotation'])
    # Express this GUI map in the native task's level-floor coordinates. This
    # is an explicit preview-only rigid transform, never an acceptance result.
    vertices = ((surface.vertices.astype(np.float64)-origin) @ rotation).astype(np.float32)
    triangles = surface.triangles.copy()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[3, :3] = -origin @ rotation
    np.savez_compressed(output/'rmuc_collision.npz', vertices=vertices, triangles=triangles)
    report = dict(schema='fresh-rmuc-preview-geometry-v1', field_root=str(root), entry=str(entry),
        package_sha256=package_sha, entry_sha256=_digest(entry), helper=str(helper), helper_sha256=_digest(helper),
        collision_meshes_in_source=len(surface.records), merged_collision_triangles=len(triangles),
        merged_vertices=len(vertices), collision_file='rmuc_collision.npz', collision_sha256=_digest(output/'rmuc_collision.npz'),
        frame_transform_row_major=transform.tolist(), selected_main_floor=spawn,
        main_floor_alignment_deg=float(np.degrees(np.arccos(np.clip(spawn['normal'][2],-1.,1.)))),
        original_asset_written=False, reference_collisions=False, merged_collision=True,
        material='Native policy task ground material; source collider partition merged into one static triangle mesh',
        height_reference='Native level-floor reference near spawn; terrain capability metrics are not evaluated here',
        physics_steps=0, native_playback_verified=False, new_PPO=0, behavior_qualified=False)
    _write_json(output/'rmuc_geometry.json', report)
    return report


def install_field(stage, prepared, output):
    """Author before simulation reset; do not edit the authenticated task code."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    mesh_path = '/World/envs/env_0/Terrain/surface_0/geometry/mesh'
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
    if not mesh or not mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError('RMUC preview requires the native single flat mesh/contact filter')
    data = np.load(Path(output)/prepared['collision_file'], allow_pickle=False)
    points, faces = data['vertices'], data['triangles']
    mesh.GetPointsAttr().Set(points.tolist())
    mesh.GetFaceVertexCountsAttr().Set([3]*len(faces))
    mesh.GetFaceVertexIndicesAttr().Set(faces.reshape(-1).tolist())
    mesh.CreateOrientationAttr('rightHanded')
    mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    mesh.CreateExtentAttr([points.min(0).tolist(), points.max(0).tolist()])
    frame = UsdGeom.Xform.Define(stage, '/World/envs/env_0/Terrain/RMUCFrame')
    frame.AddTransformOp().Set(Gf.Matrix4d(*np.asarray(prepared['frame_transform_row_major']).reshape(-1).tolist()))
    reference = UsdGeom.Xform.Define(stage, str(frame.GetPath())+'/Map').GetPrim()
    if not reference.GetReferences().AddReference(prepared['entry']):
        raise RuntimeError('RMUC visual reference failed')
    for prim in list(Usd.PrimRange(reference)):
        if prim.IsA(UsdPhysics.Scene):
            prim.SetActive(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
    if (not np.array_equal(np.asarray(mesh.GetPointsAttr().Get()), points)
            or list(mesh.GetFaceVertexIndicesAttr().Get()) != faces.reshape(-1).tolist()
            or sum(p.IsA(UsdPhysics.Scene) for p in stage.Traverse()) != 1):
        raise RuntimeError('RMUC geometry readback or single PhysicsScene failed')
    return dict(physical_mesh=mesh_path, reference=str(reference.GetPath()),
        collision_triangles=len(faces), authored_before_physics=True, duplicate_flat_floor=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Isolated RMUC floor preparation')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('field_root')
    parser.add_argument('output')
    parser.add_argument('helper')
    arguments = parser.parse_args()
    if not arguments.prepare:
        parser.error('this module only exposes --prepare for the preview GUI')
    _prepare(arguments.field_root, arguments.output, arguments.helper)


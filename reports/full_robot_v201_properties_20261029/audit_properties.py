"""Read-only mass, COM, inertia, mesh and collision comparison with chassis v7."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FULL = ROOT / 'model/wheel_leg_v2.0.1/轮腿整车URDFv2.0.1/urdf/轮腿整车URDFv2.0.1.urdf'
OLD = ROOT / 'model/纯底盘_v7_flat_base107/urdf/robot.urdf'
ALIGN = Rotation.from_euler('z', 90, degrees=True).as_matrix()
helper_path = ROOT / 'reports/full_robot_axis_inertia_20261029/update_axes.py'
helper_spec = importlib.util.spec_from_file_location('offline_fk', helper_path)
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vec(s):
    return np.fromstring(s or '0 0 0', sep=' ')


def transform(origin):
    result = np.eye(4)
    if origin is not None:
        result[:3, :3] = Rotation.from_euler('xyz', vec(origin.get('rpy'))).as_matrix()
        result[:3, 3] = vec(origin.get('xyz'))
    return result


def props(link):
    element = link.find('inertial')
    origin = transform(element.find('origin'))
    a = {k: float(v) for k, v in element.find('inertia').attrib.items()}
    matrix = np.array([[a['ixx'], a['ixy'], a['ixz']],
                       [a['ixy'], a['iyy'], a['iyz']],
                       [a['ixz'], a['iyz'], a['izz']]])
    return float(element.find('mass').get('value')), origin[:3, 3], origin[:3, :3] @ matrix @ origin[:3, :3].T


def validity(matrix):
    e = np.linalg.eigvalsh(matrix)
    return dict(principal_moments_kg_m2=e.tolist(), positive_definite=bool(e[0] > 0),
                principal_triangle_valid=bool(e[0] + e[1] >= e[2] - 1e-12))


def point_inertia(matrix, mass, com):
    return matrix + mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))


def resolve(urdf, uri):
    return urdf.parent.parent / uri.split('/', 3)[3] if uri.startswith('package://') else urdf.parent / uri


def aggregate(model, names, frames, alignment):
    records = []
    for name in names:
        mass, c, I = props(model[name])
        frame = frames[name]
        R = alignment @ frame[:3, :3]
        records.append((mass, R @ c + alignment @ frame[:3, 3], R @ I @ R.T))
    mass = sum(e[0] for e in records)
    com = sum(m * c for m, c, _ in records) / mass
    I = sum(point_inertia(tensor, m, c - com) for m, c, tensor in records)
    return dict(names=names, mass_kg=mass, com_in_chassis_root_m=com.tolist(),
                com_tensor_in_chassis_root_kg_m2=I.tolist(), validation=validity(I),
                pose='All URDF tree coordinates zero; not a standing-pose qualification')


def main():
    paths = {'full': FULL, 'chassis': OLD}
    inputs = {str(p): digest(p) for p in paths.values()}
    robots = {k: ET.parse(p).getroot() for k, p in paths.items()}
    links = {k: {l.get('name'): l for l in robot.findall('link')} for k, robot in robots.items()}
    frames = {k: helper.fk(robot) for k, robot in robots.items()}
    full_joints = {j.find('child').get('link'): j for j in robots['full'].findall('joint')}
    provenance = json.loads((OLD.parent / 'inertial_sources.json').read_text())
    comparisons = []
    for name, link in links['chassis'].items():
        om, oc, oi = props(link)
        nm, nc, ni = props(links['full'][name])
        old_frame, new_frame = frames['chassis'][name], frames['full'][name]
        new_aligned_R = ALIGN @ new_frame[:3, :3]
        Q = new_aligned_R.T @ old_frame[:3, :3]
        offset = new_aligned_R.T @ (old_frame[:3, 3] - ALIGN @ new_frame[:3, 3])
        old_I, old_com = Q @ oi @ Q.T, Q @ oc + offset
        diagonal_ratio = np.diag(ni) / np.diag(old_I)
        e = dict(name=name, old_mass_kg=om, new_mass_kg=nm, mass_delta_g=(nm - om) * 1000,
                 old_method=provenance[name]['method'], old_hardware_identified=provenance[name]['hardware_identified'],
                 old_com_in_full_link_m=old_com.tolist(), new_com_in_full_link_m=nc.tolist(),
                 com_delta_mm=((nc - old_com) * 1000).tolist(),
                 old_tensor_in_full_link_kg_m2=old_I.tolist(), new_tensor_in_full_link_kg_m2=ni.tolist(),
                 diagonal_change_percent=((diagonal_ratio - 1) * 100).tolist(),
                 old_validation=validity(old_I), new_validation=validity(ni))
        # The user reports positive-product CAD notation for old exports. Keep
        # this conditional interpretation separate from the actual URDF matrix.
        if provenance[name]['method'] == 'source_CAD_preserved':
            physical_old = 2 * np.diag(np.diag(oi)) - oi
            e['conditional_old_positive_products_converted_tensor'] = (Q @ physical_old @ Q.T).tolist()
        if name != 'base_link':
            axis = vec(full_joints[name].find('axis').get('xyz'))
            axis /= np.linalg.norm(axis)
            old_J, new_J = float(axis @ old_I @ axis), float(axis @ ni @ axis)
            e.update(old_hinge_parallel_com_inertia_kg_m2=old_J,
                     new_hinge_parallel_com_inertia_kg_m2=new_J,
                     hinge_parallel_com_inertia_change_percent=(new_J / old_J - 1) * 100,
                     old_about_new_joint_origin_inertia_kg_m2=float(axis @ point_inertia(old_I, om, old_com) @ axis),
                     new_about_joint_origin_inertia_kg_m2=float(axis @ point_inertia(ni, nm, nc) @ axis))
        comparisons.append(e)

    mesh_cache, collision_records, visual_records = {}, {}, {}
    def load(path):
        path = path.resolve()
        if path not in mesh_cache:
            mesh_cache[path] = trimesh.load_mesh(path, process=True)
        return mesh_cache[path]

    for model, urdf in paths.items():
        collision_records[model], visual_records[model] = [], []
        for name, link in links[model].items():
            for kind, destination in (('collision', collision_records[model]), ('visual', visual_records[model])):
                for el in link.findall(kind):
                    geometry = list(el.find('geometry'))[0]
                    rec = dict(name=name, type=geometry.tag, origin=el.find('origin').attrib if el.find('origin') is not None else {})
                    if geometry.tag == 'mesh':
                        path = resolve(urdf, geometry.get('filename'))
                        mesh = load(path)
                        scaled = np.asarray(mesh.vertices) * np.fromstring(geometry.get('scale', '1 1 1'), sep=' ')
                        rec.update(path=str(path.resolve()), sha256=digest(path), faces=len(mesh.faces),
                                   vertices=len(mesh.vertices), watertight=bool(mesh.is_watertight),
                                   winding_consistent=bool(mesh.is_winding_consistent),
                                   local_bounds_m=np.stack((scaled.min(axis=0), scaled.max(axis=0))).tolist(),
                                   source_mesh_is_convex=bool(mesh.is_convex),
                                   unconstrained_convex_hull_vertices=len(mesh.convex_hull.vertices))
                    else:
                        rec['parameters'] = {k: float(v) for k, v in geometry.attrib.items()}
                    destination.append(rec)
    aggregate_data = {
        'full_upper_assembly': aggregate(links['full'], ['base_link', 'M_link1', 'M_link2'], frames['full'], ALIGN),
        'full_zero_pose': aggregate(links['full'], list(links['full']), frames['full'], ALIGN),
        'chassis_zero_pose': aggregate(links['chassis'], list(links['chassis']), frames['chassis'], np.eye(3))}

    # Read the actual authored chassis USD, not just its source URDF.
    usd_path = OLD.parent / 'robot.usda'
    stage = Usd.Stage.Open(str(usd_path))
    authored_colliders, authored_self_collisions = [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            authored_colliders.append(dict(
                path=str(prim.GetPath()), type=str(prim.GetTypeName()),
                enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                    if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None,
                mesh_faces=len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or []) if prim.IsA(UsdGeom.Mesh) else 0))
        attr = prim.GetAttribute('physxArticulation:enabledSelfCollisions')
        if attr:
            authored_self_collisions.append(dict(path=str(prim.GetPath()), enabled=attr.Get()))

    base_tree = cKDTree(trimesh.transform_points(load(resolve(FULL, links['full']['base_link'].find('visual/geometry/mesh').get('filename'))).vertices,
                                               frames['full']['base_link']))
    duplicate_diagnostics = []
    for name in ('M_link1', 'M_link2'):
        mesh = load(resolve(FULL, links['full'][name].find('visual/geometry/mesh').get('filename')))
        distances = base_tree.query(trimesh.transform_points(mesh.vertices, frames['full'][name]))[0]
        duplicate_diagnostics.append(dict(name=name, fraction_vertices_within_10um_of_base=float(np.mean(distances < 1e-5)),
                                          minimum_distance_m=float(distances.min()),
                                          interpretation='No coincident whole-mesh copy detected; this does not verify CAD mass ownership'))

    lower_left_com = props(links['full']['LLL_link2'])[1]
    lower_right_com = props(links['full']['RRR_link2'])[1]
    spring_asymmetry = dict(
        left_com_m=lower_left_com.tolist(), right_com_m=lower_right_com.tolist(),
        along_rod_difference_mm=float(abs(lower_left_com[0] - lower_right_com[0]) * 1000),
        left_mass_kg=props(links['full']['LLL_link2'])[0],
        right_mass_kg=props(links['full']['RRR_link2'])[0],
        inertia_eigenvalues_identical=bool(np.allclose(np.linalg.eigvalsh(props(links['full']['LLL_link2'])[2]),
                                                     np.linalg.eigvalsh(props(links['full']['RRR_link2'])[2]), atol=1e-15)),
        right_com_matches_old_source_cad_along_rod=bool(np.isclose(lower_right_com[0], provenance['RRR_link2']['source_com_m'][1], atol=1e-12)),
        assessment='Review CAD component selection/materials or stale properties; insufficient evidence to replace automatically')

    wheels = []
    for name in ('L_link3', 'R_link3'):
        link = links['full'][name]
        col = link.find('collision')
        assert col.find('geometry/cylinder') is not None
        baseline_col = links['chassis'][name].find('collision')
        assert col.find('origin').attrib == baseline_col.find('origin').attrib
        assert col.find('geometry/cylinder').attrib == baseline_col.find('geometry/cylinder').attrib
        mesh = load(resolve(FULL, link.find('visual/geometry/mesh').get('filename')))
        hull = ConvexHull(mesh.vertices[:, :2])
        angles = np.linspace(0, 2 * np.pi, 3601)
        support = (mesh.vertices[hull.vertices, :2] @ np.array([np.cos(angles), np.sin(angles)])).max(axis=0)
        wheels.append(dict(name=name, collider='cylinder', radius_m=float(col.find('geometry/cylinder').get('radius')),
                           width_m=float(col.find('geometry/cylinder').get('length')),
                           former_stl_2d_hull_edges=len(hull.vertices), former_stl_support_ripple_mm=float(np.ptp(support) * 1000)))

    old_mass = sum(props(l)[0] for l in links['chassis'].values())
    new_mass = sum(props(l)[0] for l in links['full'].values())
    old_legs = sum(props(l)[0] for n, l in links['chassis'].items() if n != 'base_link')
    new_legs = sum(props(l)[0] for n, l in links['full'].items() if n != 'base_link' and not n.startswith('M_'))
    summary = dict(old_total_mass_kg=old_mass, new_total_mass_kg=new_mass,
                   total_mass_change_percent=(new_mass / old_mass - 1) * 100,
                   old_leg_mass_kg=old_legs, new_leg_mass_kg=new_legs, leg_mass_change_percent=(new_legs / old_legs - 1) * 100,
                   old_base_mass_kg=props(links['chassis']['base_link'])[0], new_base_mass_kg=props(links['full']['base_link'])[0],
                   new_gimbal_mass_kg=sum(props(links['full'][n])[0] for n in ('M_link1', 'M_link2')),
                   full_inertia_valid_count=sum(all(validity(props(l)[2])[k] for k in ('positive_definite', 'principal_triangle_valid'))
                                               for l in links['full'].values()),
                   old_mesh_prior_count=sum('research_prior' in d['method'] for d in provenance.values()),
                   leg_hinge_inertia_change_percent_range=[min(e['hinge_parallel_com_inertia_change_percent'] for e in comparisons if e['name'] != 'base_link'),
                                                           max(e['hinge_parallel_com_inertia_change_percent'] for e in comparisons if e['name'] != 'base_link')],
                   collision_source_types={k: dict(Counter(e['type'] for e in rows)) for k, rows in collision_records.items()},
                   collision_source_mesh_triangles={k: sum(e.get('faces', 0) for e in rows) for k, rows in collision_records.items()},
                   full_visual_triangles=sum(e.get('faces', 0) for e in visual_records['full']))
    result = dict(
        scope='Static source/model review plus user-requested wheel-collider edit; no dynamics, PPO or hardware test',
        inputs=inputs, summary=summary, comparison= comparisons, aggregate=aggregate_data,
        tensor_comparison_rule='Both tensors about own COM, rotated to the full link frame; hinge J is about an axis through COM parallel to that hinge, not whole-leg reflected inertia',
        sign_convention_rule='Primary numbers describe stored URDF matrices; conditional old positive-product CAD conversion stored separately; no blanket sign flip of mesh-derived priors',
        collision_source=collision_records, visual=visual_records,
        baseline_usd=dict(path=str(usd_path), sha256=digest(usd_path), colliders=authored_colliders,
                          authored_self_collisions=authored_self_collisions),
        full_import_note='Full model currently supplies URDF source shapes; actual PhysX approximation/material/self-collision settings require an imported USD and runtime readback',
        suspected_spring_com_asymmetry=spring_asymmetry, duplicate_mesh_diagnostics=duplicate_diagnostics,
        wheels=wheels,
        velocity_limit_review=[dict(name=j.get('name'), type=j.get('type'), limit=j.find('limit').attrib if j.find('limit') is not None else None)
                               for j in robots['full'].findall('joint')],
        unchanged_input_hashes_after_audit={str(p): digest(p) for p in paths.values()})
    assert result['unchanged_input_hashes_after_audit'] == inputs
    assert summary['full_inertia_valid_count'] == 21
    assert max(abs(e['hinge_parallel_com_inertia_change_percent']) for e in comparisons if e['name'] != 'base_link') < 4.1
    (OUT / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('spring_com_review', json.dumps(spring_asymmetry, ensure_ascii=False))
    print('base_inertia', json.dumps(next(e for e in comparisons if e['name'] == 'base_link'), ensure_ascii=False))


if __name__ == '__main__':
    main()

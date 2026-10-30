"""Reproducible v2.0.2 full-vehicle frames and closed-chain asset generation.

CAD masses, COMs and inertia tensors are re-expressed, never re-estimated.
The separate fixed-gimbal asset is a declared six-motor training fixture.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import HalfspaceIntersection
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model/wheel_leg_v2.0.1/轮腿整车URDFv2.0.1"
REFERENCE = ROOT / "model/纯底盘_v7_flat_base107/urdf"
TARGET = ROOT / "model/wheel_leg_v2.0.2"
SOURCE_SHA = "da8136d5654de4e2b718e8213ba436159720f0b9adc462df1a0756cd7854f037"
REFERENCE_SPEC_SHA = "0ac82045ab584f20091538aa782d93ea545382dc701faca1f8ae92aeb7cd1997"
REFERENCE_MANIFEST_SHA = "d3af5b544b874c8629860a9cc0c1529d215b0e3644126b3b9da7ad87b3256690"
ACTIVE = ["L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"]
GIMBAL = ["M_joint1", "M_joint2"]
SPRINGS = ["L_spring_slide", "R_spring_slide"]
ALIGN = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
MODEL_KIND = "full_robot_v202_gas_spring_closedchain"


def collision_groups(parts, maximum):
    """Spatially group assembly components, keeping every CAD vertex covered."""
    groups = [list(range(len(parts)))]
    centers = np.array([part.bounds.mean(axis=0) for part in parts])
    while len(groups) < maximum:
        candidates = [(float(np.ptp(centers[g], axis=0).max()), i)
                      for i, g in enumerate(groups) if len(g) > 1]
        if not candidates:
            break
        _, index = max(candidates)
        group = groups.pop(index)
        axis = int(np.ptp(centers[group], axis=0).argmax())
        group.sort(key=lambda k: (centers[k, axis], k))
        mid = len(group)//2
        groups.extend((group[:mid], group[mid:]))
    return groups


def bounded_convex_hull(vertices):
    """A conservative 26-plane envelope fits the GPU's small convex budget."""
    hull = trimesh.convex.convex_hull(vertices)
    if len(hull.vertices) <= 64:
        return hull, "convex_hull"
    directions = np.array([[x,y,z] for x in (-1,0,1) for y in (-1,0,1)
                           for z in (-1,0,1) if x or y or z],dtype=float)
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    support = np.max(vertices@directions.T,axis=0)
    halfspaces = np.column_stack((directions,-support))
    points = HalfspaceIntersection(halfspaces,vertices.mean(axis=0)).intersections
    result = trimesh.convex.convex_hull(points)
    if len(result.vertices)>64:
        raise ValueError("Convex collision exceeded the declared GPU vertex budget")
    return result, "conservative_26_plane_envelope"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def numbers(value):
    return " ".join(format(float(x), ".17g") for x in np.asarray(value).reshape(-1))


def transform(xyz=(0., 0., 0.), rpy=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def origin(node):
    if node is None:
        return np.eye(4)
    return transform(np.fromstring(node.get("xyz", "0 0 0"), sep=" "),
                     np.fromstring(node.get("rpy", "0 0 0"), sep=" "))


def set_origin(parent, matrix):
    node = parent.find("origin")
    if node is None:
        node = ET.SubElement(parent, "origin")
    node.set("xyz", numbers(matrix[:3, 3]))
    node.set("rpy", numbers(Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")))


def point(frame, value):
    return frame[:3, :3] @ value + frame[:3, 3]


def parse_joints(robot):
    return [dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                 child=j.find("child").get("link"), origin=origin(j.find("origin")).tolist(),
                 axis=np.fromstring(j.find("axis").get("xyz"), sep=" ").tolist(),
                 limit=dict(j.find("limit").attrib) if j.find("limit") is not None else {})
            for j in robot.findall("joint")]


def fk(spec, q, root=None):
    result = {"base_link": np.eye(4) if root is None else root}
    pending = list(spec["joints"])
    while pending:
        before = len(pending)
        for j in pending[:]:
            if j["parent"] not in result:
                continue
            motion = np.eye(4)
            if j["type"] == "prismatic":
                motion[:3, 3] = np.asarray(j["axis"]) * q.get(j["name"], 0.)
            elif j["type"] != "fixed":
                motion[:3, :3] = Rotation.from_rotvec(np.asarray(j["axis"]) * q.get(j["name"], 0.)).as_matrix()
            result[j["child"]] = result[j["parent"]] @ np.asarray(j["origin"]) @ motion
            pending.remove(j)
        if len(pending) == before:
            raise ValueError("Disconnected joint tree")
    return result


def closure_error(spec, q):
    frames = fk(spec, q)
    return np.concatenate([point(frames[c["body0"]], c["local_pos0_m"])
                           - point(frames[c["body1"]], c["local_pos1_m"]) for c in spec["constraints"]])


def solve_pose(spec, prescribed, initial=None):
    q = dict.fromkeys((j["name"] for j in spec["joints"]), 0.)
    q.update(initial or {})
    q.update(prescribed)
    for j in spec["joints"]:
        if j["type"] == "fixed":
            q[j["name"]] = 0.
    unknown = [j["name"] for j in spec["joints"] if j["name"] not in prescribed and j["type"] != "fixed"]

    def residual(values):
        return closure_error(spec, {**q, **dict(zip(unknown, values))})

    result = least_squares(residual, [q[n] for n in unknown], xtol=1e-12, ftol=1e-12,
                           gtol=1e-12, max_nfev=200, diff_step=1e-5)
    q.update(zip(unknown, result.x.tolist()))
    if np.abs(residual(result.x)).max() > 2e-7:
        raise ValueError(f"Full vehicle loop solve failed: {np.abs(residual(result.x)).max()} m")
    return q


def inertia(link):
    node = link.find("inertial")
    a = {k: float(v) for k, v in node.find("inertia").attrib.items()}
    matrix = np.array([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]],
                       [a["ixz"], a["iyz"], a["izz"]]])
    frame = origin(node.find("origin"))
    return float(node.find("mass").get("value")), frame[:3, 3], frame[:3, :3] @ matrix @ frame[:3, :3].T


def normalized_robot(raw, reference):
    """Keep every physical screw axis and zero pose; change frame coordinates."""
    result = deepcopy(raw)
    source_spec = dict(joints=parse_joints(raw))
    old_frames, source_frames = fk(reference, {}), fk(source_spec, {})
    align = np.eye(4)
    align[:3, :3] = ALIGN
    frames, changes = {"base_link": np.eye(4)}, {}
    for name, frame in source_frames.items():
        if name == "base_link":
            continue
        desired = np.eye(4)
        desired[:3, 3] = point(align, frame[:3, 3])
        desired[:3, :3] = old_frames[name][:3, :3] if name in old_frames else np.eye(3)
        frames[name] = desired
    for link in result.findall("link"):
        name = link.get("name")
        change = np.linalg.inv(frames[name]) @ align @ source_frames[name]
        changes[name] = change
        for node in [*link.findall("visual"), *link.findall("collision")]:
            set_origin(node, change @ origin(node.find("origin")))
        mass, com, tensor = inertia(link)
        com = point(change, com)
        tensor = change[:3, :3] @ tensor @ change[:3, :3].T
        node = link.find("inertial")
        set_origin(node, transform(com))
        node.find("inertia").attrib = {"i"+a+b: format(float(tensor[i, j]), ".17g")
            for i, a in enumerate("xyz") for j, b in enumerate("xyz") if i <= j}
    for j in result.findall("joint"):
        child, parent = j.find("child").get("link"), j.find("parent").get("link")
        set_origin(j, np.linalg.inv(frames[parent]) @ frames[child])
        raw_axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
        axis = changes[child][:3, :3] @ raw_axis
        j.find("axis").set("xyz", numbers(axis / np.linalg.norm(axis)))
    result.set("name", "轮腿整车URDFv2.0.2")
    for mesh in result.iter("mesh"):
        mesh.set("filename", mesh.get("filename").replace("轮腿整车URDFv2.0.1", "轮腿整车URDFv2.0.2"))
    return result, changes


def verify_reframing(raw, normalized):
    specs = [dict(joints=parse_joints(r)) for r in (raw, normalized)]
    links = [{l.get("name"): l for l in r.findall("link")} for r in (raw, normalized)]
    align = np.eye(4)
    align[:3, :3] = ALIGN
    peaks = dict(visual_transform=0., collision_transform=0., com_m=0., inertia_kg_m2=0., joint_axis=0.)
    rng = np.random.default_rng(261030)
    cases = [dict.fromkeys((j["name"] for j in specs[0]["joints"]), 0.)]
    cases += [dict(zip((j["name"] for j in specs[0]["joints"]), rng.uniform(-1.5, 1.5, 20))) for _ in range(64)]
    for q in cases:
        f0, f1 = (fk(s, q) for s in specs)
        for name in links[0]:
            for tag in ("visual", "collision"):
                for a, b in zip(links[0][name].findall(tag), links[1][name].findall(tag)):
                    error = np.abs(align @ f0[name] @ origin(a.find("origin")) - f1[name] @ origin(b.find("origin"))).max()
                    peaks[tag+"_transform"] = max(peaks[tag+"_transform"], float(error))
            m0, c0, i0 = inertia(links[0][name])
            m1, c1, i1 = inertia(links[1][name])
            if m0 != m1:
                raise ValueError("Mass changed during reframing")
            r0, r1 = (align @ f0[name])[:3, :3], f1[name][:3, :3]
            peaks["com_m"] = max(peaks["com_m"], float(np.abs(point(align @ f0[name], c0)-point(f1[name], c1)).max()))
            peaks["inertia_kg_m2"] = max(peaks["inertia_kg_m2"], float(np.abs(r0@i0@r0.T-r1@i1@r1.T).max()))
        for a, b in zip(specs[0]["joints"], specs[1]["joints"]):
            v0 = (align @ f0[a["child"]])[:3, :3] @ a["axis"]
            v1 = f1[b["child"]][:3, :3] @ b["axis"]
            peaks["joint_axis"] = max(peaks["joint_axis"], float(np.linalg.norm(v0-v1)))
    if max(peaks.values()) > 1e-10:
        raise ValueError(f"Reframing changed physical geometry: {peaks}")
    return dict(passed=True, poses=len(cases), maximum_errors=peaks, masses_exact=True,
                active_joint_signs=dict.fromkeys(ACTIVE, 1), encoder_zero_changes=0,
                gimbal_zero_link_frames="parallel to normalized base; actual CAD screw-axis tilt retained")


def export_csv(source, target, robot):
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    links = {l.get("name"): l for l in robot.findall("link")}
    joints = {j.find("child").get("link"): j for j in robot.findall("joint")}
    for row in rows:
        link = links[row["Link Name"]]
        for tag, prefix in (("inertial", "Center of Mass"), ("visual", "Visual"), ("collision", "Collision")):
            node = link.find(tag)
            t = origin(node.find("origin"))
            for suffix, value in zip(("X", "Y", "Z", "Roll", "Pitch", "Yaw"),
                                     [*t[:3, 3], *Rotation.from_matrix(t[:3, :3]).as_euler("xyz")]):
                row[prefix+" "+suffix] = format(float(value), ".17g")
        a = link.find("inertial/inertia").attrib
        for key, value in a.items():
            row["Moment I"+key[1:]] = value
        for key in ("Mesh Filename", "Collision Mesh Filename"):
            row[key] = row[key].replace("轮腿整车URDFv2.0.1", "轮腿整车URDFv2.0.2")
        j = joints.get(row["Link Name"])
        if j is not None:
            t = origin(j.find("origin"))
            for suffix, value in zip(("X", "Y", "Z", "Roll", "Pitch", "Yaw"),
                                     [*t[:3, 3], *Rotation.from_matrix(t[:3, :3]).as_euler("xyz")]):
                row["Joint Origin "+suffix] = format(float(value), ".17g")
            for axis, value in zip("XYZ", j.find("axis").get("xyz").split()):
                row["Joint Axis "+axis] = value
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def make_spec(robot, reference, bundle):
    joints = parse_joints(robot)
    frames = fk(dict(joints=joints), {})
    spec = dict(model_kind=MODEL_KIND, coordinate_schema="full_robot_v202_canonical_v1",
                control_frame="Xforward_Yleft_Zup", joints=joints, bodies=[], constraints=[],
                nominal_base_height_m=.32, spring_binding={}, passive_knee_zero_inner_rad={})
    collision_report = {}
    for link in robot.findall("link"):
        name = link.get("name")
        mass, com, tensor = inertia(link)
        eig = np.linalg.eigvalsh(tensor)
        if eig[0] <= 0 or eig[0]+eig[1] < eig[2]-1e-12:
            raise ValueError(f"Invalid CAD inertia for {name}")
        mesh_file = SOURCE / "meshes" / (name+".STL")
        shutil.copy2(mesh_file, bundle / "meshes" / (name+".stl"))
        collisions, methods = [], []
        source_collision = link.find("collision")
        col_origin = origin(source_collision.find("origin"))
        cylinder = source_collision.find("geometry/cylinder")
        if cylinder is not None:
            collisions.append(dict(type="cylinder", radius=float(cylinder.get("radius")),
                length=float(cylinder.get("length")), origin=col_origin.tolist()))
            methods.append(dict(method="preserved_v201_cylinder"))
        else:
            mesh = trimesh.load(mesh_file, force="mesh", process=True)
            raw_parts = list(mesh.split(only_watertight=False))
            parts = []
            for index, part in enumerate(raw_parts):
                if len(part.vertices)<4 or np.linalg.matrix_rank(part.vertices-part.vertices[0],tol=1e-9)<3:
                    methods.append(dict(source_part=index,method="skip_zero_volume_surface",faces=len(part.faces)))
                    continue
                parts.append(part)
            # CAD assemblies include hundreds of screws/internal parts. Their
            # individual hulls are unsuitable for thousands of RL instances.
            maximum = 8 if name=="base_link" else 4 if name.startswith("M_") else len(parts)
            groups = collision_groups(parts,min(maximum,len(parts)))
            for index, members in enumerate(groups):
                vertices = np.concatenate([parts[k].vertices for k in members])
                hull, method = bounded_convex_hull(vertices)
                path = f"collisions/{name}_{index:04d}.obj"
                hull.export(bundle/path)
                collisions.append(dict(type="mesh", file=path, origin=col_origin.tolist()))
                methods.append(dict(group=index,source_components=len(members),method=method,
                    vertices=len(hull.vertices),faces=len(hull.faces),conservative_envelope=True))
        if not collisions:
            raise ValueError(f"No collision volume for {name}")
        rgba = link.find("visual/material/color")
        color = np.fromstring(rgba.get("rgba"), sep=" ")[:3].tolist() if rgba is not None else [.65]*3
        spec["bodies"].append(dict(name=name, component=name, mass=mass, com=com.tolist(),
            inertia=tensor.tolist(), mesh=f"meshes/{name}.stl", mesh_origin=origin(link.find("visual/origin")).tolist(),
            color=color, collisions=collisions))
        collision_report[name] = methods
    for j in joints:
        if j["name"] in GIMBAL:
            j["type"] = "fixed"
            j["limit"] = {}
        else:
            cap = 40. if j["name"] in (ACTIVE[0], ACTIVE[1], ACTIVE[3], ACTIVE[4]) else 4.5 if j["name"] in (ACTIVE[2], ACTIVE[5]) else 100.
            j["limit"] = dict(effort=str(cap), velocity="100")
    for side, name in (("L", "L_joint2"), ("R", "R_jonit2")):
        knee = frames[side+"_link2"][:3, 3]
        upper = frames[side+"_link1"][:3, 3]-knee
        lower = frames[side+"_link3"][:3, 3]-knee
        j = next(j for j in joints if j["name"] == name)
        axis = frames[j["child"]][:3, :3] @ j["axis"]
        upper -= axis*(axis@upper)
        lower -= axis*(axis@lower)
        beta0 = float(np.arccos(np.clip(upper@lower/np.linalg.norm(upper)/np.linalg.norm(lower), -1., 1.)))
        spec["passive_knee_zero_inner_rad"][name] = beta0
        sign = 1. if side == "L" else -1.
        bounds = sorted((np.deg2rad(np.array([40., 107.]))-beta0)/sign)
        j["type"] = "revolute"
        j["limit"].update(lower=str(bounds[0]), upper=str(bounds[1]))
    for c in reference["constraints"]:
        if "spring_upper_mount" in c["name"]:
            continue
        c = deepcopy(c)
        side = c["name"][0]
        axis = frames[side+"_link1"][:3, :3] @ next(j["axis"] for j in joints if j["name"] == side+"_joint1")
        w0, w1 = (point(frames[c[f"body{i}"]], c[f"local_pos{i}_m"]) for i in (0, 1))
        shift = axis*((w0-w1)@axis)
        c["local_pos1_m"] = point(np.linalg.inv(frames[c["body1"]]), w1+shift).tolist()
        c["v202_axial_representative_shift_m"] = float(np.linalg.norm(shift))
        spec["constraints"].append(c)
    installation = json.loads((REFERENCE/"installation.json").read_text())
    full_length = installation["full_extension_pin_distance_m"]
    for side in ("L", "R"):
        upper, lower = side*3+"_link1", side*3+"_link2"
        j = next(j for j in joints if j["child"] == upper)
        parent, mount = j["parent"], np.asarray(j["origin"])[:3, 3]
        distance = float(np.linalg.norm(frames[upper][:3, 3]-frames[lower][:3, 3]))
        direction = (frames[upper][:3, 3]-frames[lower][:3, 3])/distance
        compression0 = full_length-distance
        name = side+"_spring_slide"
        j.update(name=name, parent=lower, type="prismatic",
            origin=(np.linalg.inv(frames[lower])@frames[upper]).tolist(),
            axis=(frames[upper][:3, :3].T@direction).tolist(),
            limit=dict(lower=str(compression0-installation["stroke_m"]), upper=str(compression0), effort="1500", velocity="100"))
        spec["constraints"].append(dict(name=side+"_spring_upper_mount", type="spherical", body0=parent,
            body1=upper, local_pos0_m=mount.tolist(), local_pos1_m=[0., 0., 0.],
            **{"physics:excludeFromArticulation": True}))
        spec["spring_binding"][name] = dict(joint_coordinate="extension_relative_to_CAD_zero_m",
            source_mount_distance_m=distance, stroke_m=installation["stroke_m"],
            full_extension_pin_distance_m=full_length, compression_at_q_zero_m=compression0,
            compression_expression="compression_at_q_zero_m - q_m", effort_sign=1,
            source="inherited232mm_pin_distance80mm_stroke; v201 mount geometry",
            hardware_stop_measurement_verified=False)
    ordered, reached, pending = [], {"base_link"}, list(joints)
    while pending:
        ready = [j for j in pending if j["parent"] in reached]
        if not ready:
            raise ValueError("Reparented spring tree is disconnected")
        for j in ready:
            ordered.append(j)
            reached.add(j["child"])
            pending.remove(j)
    spec["joints"] = ordered
    prescribed = dict.fromkeys(["L_joint1", "L_joint2", "L_joint3", "R_joint1", "R_jonit2", "R_joint3"], 0.)
    q = solve_pose(spec, prescribed)
    spec["source_closed_joint_pos"] = q
    for fraction in np.linspace(0., 1., 11)[1:]:
        prescribed.update(zip(prescribed, np.array([-.42, .4, 0., .42, -.4, 0.])*fraction))
        q = solve_pose(spec, prescribed, q)
    spec["nominal_joint_pos"] = q
    return spec, collision_report


def load_exporter():
    path = REFERENCE/"tools/mechanism_export_legacy.py"
    definition = importlib.util.spec_from_file_location("full_robot_v202_exporter", path)
    exporter = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(exporter)
    exporter.CONTROL, exporter.tree_fk = ACTIVE, fk
    return exporter


def export_assets(bundle, spec):
    from pxr import Gf, Sdf, Usd, UsdPhysics
    exporter = load_exporter()
    exporter.export_urdf(bundle, spec)
    exporter.export_usd(bundle, spec)
    stage = Usd.Stage.Open(str(bundle/"robot.usda"))

    def quat(matrix):
        v = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        return Gf.Quatf(float(v[3]), Gf.Vec3f(*v[:3]))

    for j in spec["joints"]:
        path = "/Robot/tree_joints/"+j["name"]
        stage.RemovePrim(path)
        kind = j["type"]
        cls = UsdPhysics.FixedJoint if kind == "fixed" else UsdPhysics.PrismaticJoint if kind == "prismatic" else UsdPhysics.RevoluteJoint
        joint = cls.Define(stage, path)
        joint.CreateBody0Rel().SetTargets(["/Robot/"+j["parent"]])
        joint.CreateBody1Rel().SetTargets(["/Robot/"+j["child"]])
        t, a = np.asarray(j["origin"]), np.eye(4)
        if kind != "fixed":
            basic = np.array([1., 0., 0.] if kind == "prismatic" else [0., 0., 1.])
            a[:3, :3] = Rotation.align_vectors([j["axis"]], [basic])[0].as_matrix()
            joint.CreateAxisAttr("X" if kind == "prismatic" else "Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*t[:3, 3]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot0Attr(quat(t@a))
        joint.CreateLocalRot1Attr(quat(a))
        joint.CreateExcludeFromArticulationAttr(False)
        if kind in ("prismatic", "revolute"):
            factor = 1. if kind == "prismatic" else 180./np.pi
            joint.CreateLowerLimitAttr(float(j["limit"]["lower"])*factor)
            joint.CreateUpperLimitAttr(float(j["limit"]["upper"])*factor)
    stage.GetDefaultPrim().SetCustomData(dict(model_kind=MODEL_KIND, gimbal_mode="fixed_zero",
        control_frame="Xforward_Yleft_Zup", hardware_deployment_ready=False))
    stage.GetRootLayer().Save()
    free = Usd.Stage.CreateNew(str(bundle/"robot_gimbal_free.usda"))
    free.GetRootLayer().subLayerPaths = ["robot.usda"]
    free.SetDefaultPrim(free.GetPrimAtPath("/Robot"))
    free.GetDefaultPrim().SetCustomData(dict(model_kind=MODEL_KIND, gimbal_mode="free_uncontrolled",
        control_frame="Xforward_Yleft_Zup", hardware_deployment_ready=False))
    for j in spec["joints"]:
        if j["name"] not in GIMBAL:
            continue
        joint = UsdPhysics.RevoluteJoint.Define(free, "/Robot/tree_joints/"+j["name"])
        a = np.eye(4)
        a[:3, :3] = Rotation.align_vectors([j["axis"]], [[0., 0., 1.]])[0].as_matrix()
        joint.CreateAxisAttr("Z")
        joint.CreateLocalRot0Attr(quat(np.asarray(j["origin"])@a))
        joint.CreateLocalRot1Attr(quat(a))
    free.GetRootLayer().Save()
    robot = ET.parse(bundle/"robot.urdf").getroot()
    robot.set("name", "wheel_leg_v2.0.2_gimbal_fixed_closedchain")
    for j in robot.findall("joint"):
        if j.get("type") == "fixed":
            j.remove(j.find("limit"))
            j.remove(j.find("axis"))
        else:
            ET.SubElement(j, "dynamics", damping="0" if j.get("name") in ACTIVE else "0.002", friction="0")
    exporter.write_xml(bundle/"robot.urdf", robot)


def build(output=TARGET):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new v2.0.2 output folder; existing versions are never overwritten")
    source_urdf = SOURCE/"urdf/轮腿整车URDFv2.0.1.urdf"
    if digest(source_urdf) != SOURCE_SHA or digest(REFERENCE/"model_spec.json") != REFERENCE_SPEC_SHA:
        raise ValueError("Reviewed v2.0.1 source or chassis geometry identity changed")
    if digest(REFERENCE/"manifest.json") != REFERENCE_MANIFEST_SHA:
        raise ValueError("Chassis reference manifest changed")
    reference_manifest = json.loads((REFERENCE/"manifest.json").read_text())
    for name, expected in reference_manifest["files_sha256"].items():
        if digest(REFERENCE/name) != expected:
            raise ValueError(f"Chassis reference dependency changed: {name}")
    source_models = {
        "model/gas_spring_compensation_20261028/model.json":"098fd7d5fbd876cd57d0b6e01e239eb16f6789690a5eadf7820ec20d974919a3",
        "model/wheel_identified_1562_20261028/model.json":"2c67e1113f0a460cd477e3ec6897e258619304a69879da3acabec148bb61823d"}
    for name, expected in source_models.items():
        if digest(ROOT/name) != expected:
            raise ValueError(f"Inherited measured model changed: {name}")
    source_hashes = {str(p.relative_to(SOURCE)): digest(p) for p in sorted(SOURCE.rglob("*")) if p.is_file()}
    raw = ET.parse(source_urdf).getroot()
    reference = json.loads((REFERENCE/"model_spec.json").read_text())
    normalized, changes = normalized_robot(raw, reference)
    frame_validation = verify_reframing(raw, normalized)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".full-v202-", dir=output.parent) as tmp:
        staging = Path(tmp)/"v202"
        staging.mkdir()
        package = staging/"轮腿整车URDFv2.0.2"
        shutil.copytree(SOURCE, package)
        for path in sorted(package.rglob("*"), reverse=True):
            if not path.is_file() or path.suffix == ".STL":
                continue
            if path.name != "export.log":
                content = path.read_text().replace("轮腿整车URDFv2.0.1", "轮腿整车URDFv2.0.2")
                path.write_text(content)
            if "v2.0.1" in path.name:
                path.rename(path.with_name(path.name.replace("v2.0.1", "v2.0.2")))
        dest = package/"urdf/轮腿整车URDFv2.0.2.urdf"
        ET.indent(normalized, space="  ")
        ET.ElementTree(normalized).write(dest, encoding="utf-8", xml_declaration=True)
        export_csv(source_urdf.with_suffix(".csv"), dest.with_suffix(".csv"), normalized)
        save(staging/"coordinate_mapping.json", dict(schema="full-robot-v202-frame-reexpression-v1",
            source_urdf_sha256=SOURCE_SHA, source_base_to_control_rotation=ALIGN.tolist(),
            body_local_source_to_v202={k:v.tolist() for k,v in changes.items()},
            physical_gimbal_tilt_preserved=True, active_joint_sign_from_v201=dict.fromkeys(ACTIVE, 1),
            encoder_zero_changes=0, measured_mass_or_inertia_changes=False))
        save(staging/"frame_validation.json", frame_validation)
        bundle = staging/"isaac"
        for name in ("meshes", "collisions"):
            (bundle/name).mkdir(parents=True)
        spec, collision_report = make_spec(normalized, reference, bundle)
        save(bundle/"model_spec.json", spec)
        save(bundle/"constraints.json", dict(constraints=spec["constraints"], urdf_is_tree_only=True,
            required_for_dynamics=True, expected_independent_constraint_rank=12))
        save(bundle/"gas_spring_binding.json", spec["spring_binding"])
        save(bundle/"collision_generation.json", collision_report)
        export_assets(bundle, spec)
        save(staging/"source_provenance.json", dict(source_directory=str(SOURCE.relative_to(ROOT)),
            source_files_sha256=source_hashes, chassis_reference_spec_sha256=REFERENCE_SPEC_SHA,
            chassis_reference_manifest_sha256=REFERENCE_MANIFEST_SHA,
            inherited_models_sha256=source_models,
            exporter_sha256=digest(REFERENCE/"tools/mechanism_export_legacy.py"),
            generator=str(Path(__file__).relative_to(ROOT)), generator_sha256=digest(__file__)))
        for name, expected in source_hashes.items():
            if digest(SOURCE/name) != expected:
                raise ValueError("v2.0.1 source changed during generation")
        # Remaining bindings and geometry tables are generated before publication.
        finish_bundle(staging, output, spec)
        shutil.copy2(ROOT/"docs/FULL_ROBOT_V202.md",staging/"README.md")
        shutil.move(str(staging), str(output))
    return dict(output=str(output), mass_kg=sum(b["mass"] for b in spec["bodies"]),
                source_unchanged=True, frame_validation=frame_validation)


def finish_bundle(staging, published, spec):
    """Attach the inherited measured curves and the new geometry identity."""
    bundle = staging/"isaac"
    model = deepcopy(json.loads((ROOT/"model/gas_spring_compensation_20261028/model.json").read_text()))
    model["model_spec_sha256"] = digest(bundle/"model_spec.json")
    model["source_geometry_model_sha256"] = digest(ROOT/"model/gas_spring_compensation_20261028/model.json")
    model["source_curve"] = "model/gas_spring_identified_symmetric_20261028/model.json"
    model["sides"] = {}
    names = [j["name"] for j in spec["joints"] if j["type"] != "fixed"]
    active = [names.index(n) for n in ACTIVE]
    passive = [i for i in range(len(names)) if i not in active]
    rows, q, max_gap = [], spec["nominal_joint_pos"], 0.
    print("Solving v2.0.2 closed-chain gas-spring and encoder geometry", flush=True)
    for beta in np.linspace(39.5, 108.5, 277):
        prescribed = {"L_joint1": .42, "R_joint1": -.42, "L_joint3": 0., "R_joint3": 0.,
            "L_joint2": np.deg2rad(beta)-spec["passive_knee_zero_inner_rad"]["L_joint2"],
            "R_jonit2": -(np.deg2rad(beta)-spec["passive_knee_zero_inner_rad"]["R_jonit2"])}
        q = solve_pose(spec, prescribed, q)
        columns = []
        for name in names:
            plus, minus = dict(q), dict(q)
            plus[name] += 1e-6
            minus[name] -= 1e-6
            columns.append((closure_error(spec, plus)-closure_error(spec, minus))/2e-6)
        jac = np.column_stack(columns)
        mapping = np.zeros((len(names), 6))
        mapping[active] = np.eye(6)
        mapping[passive] = np.linalg.lstsq(jac[:, passive], -jac[:, active], rcond=1e-8)[0]
        if np.abs(jac@mapping).max() > 1e-7:
            raise ValueError("Closed-chain velocity mapping residual exceeds inherited tolerance")
        gap = float(np.linalg.norm(closure_error(spec, q).reshape(-1, 3), axis=1).max())
        max_gap = max(max_gap, gap)
        rows.append((float(beta), dict(q), mapping, fk(spec, q)))
    table = dict(schema="full_robot_v202_encoder_geometry_v1", model_spec_sha256=model["model_spec_sha256"],
        control_joint_names=ACTIVE, passive_knee_domain_deg=[40., 107.], max_closure_gap_m=max_gap,
        nominal_motor_positions=[spec["nominal_joint_pos"][n] for n in (ACTIVE[0], ACTIVE[1], ACTIVE[3], ACTIVE[4])],
        hardware_calibrated=False, sides={})
    for word, side, columns in (("left", "L", slice(0, 2)), ("right", "R", slice(3, 5))):
        hip, aux, slide = side+"_joint1", side*2+"_joint1", side+"_spring_slide"
        delta = np.array([(r[1][aux]-r[1][hip])*(1. if side == "L" else -1.) for r in rows])
        if not np.all(np.diff(delta)>0):
            raise ValueError("Gas geometry is not a continuous monotone assembly branch")
        model["sides"][word] = dict(motor_names=[hip, aux], delta_rad=delta.tolist(),
            beta_deg=[r[0] for r in rows], compression_m=[spec["spring_binding"][slide]["compression_at_q_zero_m"]-r[1][slide] for r in rows],
            length_jacobian_m_per_rad=[r[2][names.index(slide), columns].tolist() for r in rows], closure_max_m=max_gap)
        subset = [r for r in rows if 40. <= r[0] <= 107.]
        subset.sort(key=lambda r:r[1][aux]-r[1][hip])
        hj = next(j for j in spec["joints"] if j["name"] == hip)
        bodies = {b["name"]:b for b in spec["bodies"]}
        wheel = bodies[side+"_link3"]["collisions"][0]
        wheel_frames = [r[3][side+"_link3"]@np.array(wheel["origin"]) for r in subset]
        table["sides"][word] = dict(hip_reference_rad=.42 if side=="L" else -.42,
            hip_origin_b_m=np.array(hj["origin"])[:3,3].tolist(),
            hip_axis_b=(np.array(hj["origin"])[:3,:3]@hj["axis"]).tolist(),
            wheel_axis_b=wheel_frames[0][:3,2].tolist(), wheel_radius_m=wheel["radius"],
            delta_rad=[r[1][aux]-r[1][hip] for r in subset], passive_inner_deg=[r[0] for r in subset],
            wheel_center_b_m=[f[:3,3].tolist() for f in wheel_frames])
    save(bundle/"gas_spring_model.json", model)
    save(bundle/"encoder_geometry.json", table)
    wheel_source = ROOT/"model/wheel_identified_1562_20261028/model.json"
    wheel_model = deepcopy(json.loads(wheel_source.read_text()))
    for name in ("L_joint3", "R_joint3"):
        body = next(b for b in spec["bodies"] if b["name"] == name.replace("joint", "link"))
        joint = next(j for j in spec["joints"] if j["name"] == name)
        axis = np.asarray(joint["axis"])
        cad = float(axis@np.array(body["inertia"])@axis)
        row = wheel_model["axes"][name]
        row["cad_inertia_kg_m2"] = cad
        row["additional_armature_kg_m2"] = row["effective_total_inertia_kg_m2"]-cad
        if row["additional_armature_kg_m2"] < 0:
            raise ValueError("New CAD inertia exceeds measured effective total wheel inertia")
    wheel_model["asset_rebinding"] = dict(source_model_sha256=digest(wheel_source),
        model_spec_sha256=model["model_spec_sha256"], rule="same measured total inertia minus new CAD axial inertia; no refit")
    save(bundle/"wheel_model.json", wheel_model)
    prior = json.loads((ROOT/"contracts/wheel_leg_prone_motion_control_100hz_v1.json").read_text())
    save(bundle/"control.json", prior)
    cfg = json.loads((ROOT/"contracts/scratch_standing_base_mechanism_v2.json").read_text())
    settings = cfg["v5_control"]
    settings["gas_spring_compensation"] = dict(enabled=True,
        model_file=str((published/"isaac/gas_spring_model.json").relative_to(ROOT)),
        model_sha256=digest(bundle/"gas_spring_model.json"), gain=1.,
        force_extrapolation="positive_c1_endpoint_trend_v1", physical_curve_outside_geometry="hold_geometry_endpoint",
        relative_soft_limits=dict(minimum=-.140638855, maximum=1.3889088544, margin=.03, projection="outward_relative_torque_v1"))
    settings["source_inertia_scope"] = "v201 CAD mass COM inertia preserved; coordinate re-expression only"
    settings["mass_asymmetry"] = "v201 CAD retained; LL_link2 and RR_link2 are both0.08357kg; no imposed offset"
    settings["spring_pressure_mpa"] = None
    settings["spring_reserve_model"] = "inherited measured effective load and explicit endpoint trend"
    wheel_cfg = dict(enabled=True, model_path=str((published/"isaac/wheel_model.json").relative_to(ROOT)),
        model_sha256=digest(bundle/"wheel_model.json"), apply_aggregate_delay=True)
    save(bundle/"runtime.json", dict(schema="full-robot-v202-six-motor-runtime-v1", gimbal_mode="fixed_zero",
        physics_dt=.001, policy_dt=.01, pc_control_dt=.001, solver_position_iterations=64, solver_velocity_iterations=32,
        enable_external_forces_every_iteration=True, control_joint_names=ACTIVE,
        v5_control=settings, identified_wheel_response=wheel_cfg,
        low_level_extra_delay_ms=0., wheel_delay_ms=[5.5, 3.5], policy_observation_age="existing v2 uncertainty, separate from calibrated wheel delay",
        control_file="control.json", encoder_geometry_file="encoder_geometry.json",
        randomization_contract="contracts/wheel_leg_randomization_study_v2.json",
        training_initialization="fresh actor critic and empty Adam; no checkpoint teacher donor",
        loaded_wheel_torque_curve_identified=False, physical_gas_pressure_identified=False,
        right_gas_spring_symmetry_assumed=True, behavior_qualified=False))
    save(bundle/"mechanical_limits.json", dict(knee_inner_limits_deg=[40.,107.],
        passive_knee_zero_inner_rad=spec["passive_knee_zero_inner_rad"],
        relative_soft_limits=settings["gas_spring_compensation"]["relative_soft_limits"],
        leg_total_limit_nm=40., wheel_motor_limit_nm=4.5,
        passive_joint_velocity_limit_rad_s=100., velocity_limit_scope="inherited solver guard, not newly identified motor speed"))
    manifest = dict(schema_version=1, model_kind=MODEL_KIND, asset_revision="full_robot_v202_gimbal_fixed_v1",
        coordinate_schema=spec["coordinate_schema"], control_frame=spec["control_frame"],
        urdf="robot.urdf", usd="robot.usda", gimbal_free_usd="robot_gimbal_free.usda",
        usd_prim_path="/Robot", usd_articulation_root_path="/Robot/base_link",
        rigid_body_names=[b["name"] for b in spec["bodies"]],
        tree_joint_names=[j["name"] for j in spec["joints"] if j["type"]!="fixed"],
        fixed_joint_names=GIMBAL, control_joint_names=ACTIVE, spring_joint_names=SPRINGS,
        nominal_joint_pos={k:v for k,v in spec["nominal_joint_pos"].items() if k not in GIMBAL},
        nominal_base_height_m=.32, total_mass_kg=sum(b["mass"] for b in spec["bodies"]),
        wheel_gear_ratio=15.62, knee_inner_limits_deg=[40.,107.], gimbal_mode="fixed_zero",
        self_collision_enabled=False, runtime_spring_force_required=True,
        inertia_rebuilt_count=0, source_urdf_sha256=SOURCE_SHA,
        static_validation_passed=True, physx_validation="pending; separate probe receipt required",
        training_ready=False, hardware_deployment_ready=False,
        pending_gates=["native_PhysX_component_probe", "new_full_vehicle_pose_bank_and_curriculum_admission", "full_behavior_acceptance"],
        files_sha256={str(p.relative_to(bundle)):digest(p) for p in sorted(bundle.rglob("*")) if p.is_file()})
    save(bundle/"manifest.json", manifest)
    save(staging/"build_receipt.json", dict(version="2.0.2", source_version="2.0.1", new_PPO=0,
        raw_source_unchanged=True, total_mass_kg=manifest["total_mass_kg"], bodies=21,
        moving_tree_joints=18, fixed_gimbal_joints=2, external_loop_constraints=6,
        geometry_max_closure_gap_m=max_gap, new_model_spec_sha256=model["model_spec_sha256"],
        asset_manifest_sha256=digest(bundle/"manifest.json")))


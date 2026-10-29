"""Build a full v2.0.1 plant without changing the supplied CAD source.

CAD inertias and geometry are preserved. Loop pins are transported from the
reviewed chassis datum, with the transport error and research limits recorded.
This module has no Isaac/Kit imports and can be checked on the CPU.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ACTIVE = ["L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"]
SPRINGS = ["L_spring_slide", "R_spring_slide"]
MODEL_KIND = "full_robot_v201_closedchain_research"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def transform(xyz=(0., 0., 0.), rpy=(0., 0., 0.)):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    out[:3, 3] = xyz
    return out


def origin(node):
    if node is None:
        return np.eye(4)
    return transform(np.fromstring(node.get("xyz", "0 0 0"), sep=" "),
                     np.fromstring(node.get("rpy", "0 0 0"), sep=" "))


def point(frame, value):
    return frame[:3, :3] @ value + frame[:3, 3]


def fk(spec, coordinates, root=None):
    frames = {"base_link": np.eye(4) if root is None else root}
    pending = list(spec["joints"])
    while pending:
        ready = [joint for joint in pending if joint["parent"] in frames]
        if not ready:
            raise ValueError("Disconnected or cyclic mechanism")
        for joint in ready:
            q = coordinates.get(joint["name"], 0.)
            motion = np.eye(4)
            if joint["type"] == "prismatic":
                motion[:3, 3] = np.asarray(joint["axis"]) * q
            elif joint["type"] != "fixed":
                motion[:3, :3] = Rotation.from_rotvec(np.asarray(joint["axis"]) * q).as_matrix()
            frames[joint["child"]] = frames[joint["parent"]] @ np.asarray(joint["origin"]) @ motion
            pending.remove(joint)
    return frames


def closure_residual(spec, coordinates):
    frames = fk(spec, coordinates)
    return np.concatenate([point(frames[c["body0"]], c["local_pos0_m"])
                           - point(frames[c["body1"]], c["local_pos1_m"])
                           for c in spec["constraints"]])


def solve_pose(spec, prescribed, initial=None, tolerance_m=2e-6):
    coordinates = dict.fromkeys((j["name"] for j in spec["joints"]), 0.)
    coordinates.update(initial or {})
    coordinates.update(prescribed)
    unknown = [j["name"] for j in spec["joints"] if j["name"] not in prescribed
               and j["name"] not in ("M_joint1", "M_joint2")]

    def residual(values):
        return closure_residual(spec, dict(coordinates, **dict(zip(unknown, values))))

    result = least_squares(residual, [coordinates[n] for n in unknown],
                           xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=180)
    coordinates.update(zip(unknown, result.x.tolist()))
    if np.max(np.abs(residual(result.x))) > tolerance_m:
        raise ValueError(f"Full-model closure solve failed: {result.message}")
    for joint in spec["joints"]:
        if joint["type"] in ("prismatic", "revolute"):
            q = coordinates[joint["name"]]
            if not float(joint["limit"]["lower"]) - 1e-7 <= q <= float(joint["limit"]["upper"]) + 1e-7:
                raise ValueError(f"Pose outside {joint['name']} limit")
    return coordinates


def read_cad(source):
    robot = ET.parse(source).getroot()
    root_rotation = transform(rpy=(0., 0., np.pi / 2))
    spec = {"joints": [], "bodies": [], "constraints": [],
            "control_frame": "Xforward_Yleft_Zup", "model_kind": MODEL_KIND}
    for node in robot.findall("joint"):
        frame = origin(node.find("origin"))
        parent = node.find("parent").get("link")
        if parent == "base_link":
            frame = root_rotation @ frame
        spec["joints"].append(dict(name=node.get("name"), parent=parent,
            child=node.find("child").get("link"), type=node.get("type"), origin=frame.tolist(),
            axis=np.fromstring(node.find("axis").get("xyz"), sep=" ").tolist(),
            limit=dict(node.find("limit").attrib) if node.find("limit") is not None else {}))
    for node in robot.findall("link"):
        inertia = node.find("inertial")
        inertial_frame = origin(inertia.find("origin"))
        tensor_node = inertia.find("inertia")
        tensor = np.array([[float(tensor_node.get("i" + a + b if a <= b else "i" + b + a))
                            for b in "xyz"] for a in "xyz"])
        tensor = inertial_frame[:3, :3] @ tensor @ inertial_frame[:3, :3].T
        frame = root_rotation if node.get("name") == "base_link" else np.eye(4)
        tensor = frame[:3, :3] @ tensor @ frame[:3, :3].T
        eigen = np.linalg.eigvalsh(tensor)
        if eigen[0] <= 0 or eigen[0] + eigen[1] < eigen[2] - 1e-12:
            raise ValueError(f"Invalid CAD inertia: {node.get('name')}")
        spec["bodies"].append(dict(name=node.get("name"), component=node.get("name"),
            mass=float(inertia.find("mass").get("value")),
            com=point(frame, inertial_frame[:3, 3]).tolist(), inertia=tensor.tolist(),
            mesh=f"meshes/{node.get('name')}.stl", mesh_origin=frame.tolist(),
            color=[.65, .68, .72], collisions=[]))
    return robot, spec


def build_spec(source, reference_spec):
    source = Path(source)
    robot, spec = read_cad(source)
    zero = fk(spec, {})
    reference_zero = fk(reference_spec, {})
    transports = {name: np.linalg.inv(zero[name]) @ reference_zero[name]
                  for name in reference_zero if name != "base_link"}
    max_anchor_shift = max(float(np.linalg.norm(zero[name][:3, 3] - reference_zero[name][:3, 3]))
                           for name in transports)
    if max_anchor_shift > 1e-4:
        raise ValueError("CAD anchor geometry changed; reviewed pin transport no longer applies")
    # All legs have the same pin geometry but different local mesh axes.
    for old in reference_spec["constraints"]:
        if "spring_upper_mount" in old["name"]:
            continue
        constraint = copy.deepcopy(old)
        for end in (0, 1):
            name = constraint[f"body{end}"]
            constraint[f"local_pos{end}_m"] = point(transports[name], old[f"local_pos{end}_m"]).tolist()
        spec["constraints"].append(constraint)
    spring_binding = {}
    for side in ("L", "R"):
        upper, lower = side * 3 + "_link1", side * 3 + "_link2"
        joint = next(j for j in spec["joints"] if j["child"] == upper)
        parent, mount = joint["parent"], np.asarray(joint["origin"])[:3, 3]
        delta = zero[upper][:3, 3] - zero[lower][:3, 3]
        distance = float(np.linalg.norm(delta))
        relative = np.linalg.inv(zero[lower]) @ zero[upper]
        extension, stroke = .232, .08
        compression = extension - distance
        name = side + "_spring_slide"
        joint.update(name=name, parent=lower, type="prismatic", origin=relative.tolist(),
                     axis=(zero[upper][:3, :3].T @ (delta / distance)).tolist(),
                     limit=dict(lower=str(compression - stroke), upper=str(compression),
                                effort="1000", velocity="100"))
        spec["constraints"].append(dict(name=side + "_spring_upper_mount", type="spherical",
            body0=parent, body1=upper, local_pos0_m=mount.tolist(), local_pos1_m=[0., 0., 0.],
            **{"physics:excludeFromArticulation": True}))
        spring_binding[name] = dict(source_mount_distance_m=distance, stroke_m=stroke,
            full_extension_pin_distance_m=extension, compression_at_q_zero_m=compression,
            effort_sign=1, source="inherited_installed_232mm_80mm_research_contract",
            hardware_verified_for_full_robot=False)
    for joint in spec["joints"]:
        # CAD's effort=100 and speed=1 are exporter placeholders. Plant actuator
        # capacities are explicitly configured separately; never train at 1rad/s.
        joint["limit"].update(effort="100", velocity="100")
        if joint["name"] in ("L_joint2", "R_jonit2"):
            zero_inner = np.pi - 2.3573
            limits = np.deg2rad([40., 107.]) - zero_inner
            if joint["name"] == "R_jonit2":
                limits = -limits[::-1]
            joint.update(type="revolute")
            joint["limit"].update(lower=str(limits[0]), upper=str(limits[1]))
    spec["spring_binding"] = spring_binding
    fixed = {"L_joint1": 0., "R_joint1": 0., "L_joint2": 0., "R_jonit2": 0.,
             "L_joint3": 0., "R_joint3": 0., "M_joint1": 0., "M_joint2": 0.}
    q = solve_pose(spec, fixed)
    spec["source_closed_joint_pos"] = q
    for fraction in np.linspace(0., 1., 11)[1:]:
        q = solve_pose(spec, dict(fixed, L_joint1=-.42*fraction, R_joint1=.42*fraction,
                                 L_joint2=.4*fraction, R_jonit2=-.4*fraction), q)
    spec["nominal_joint_pos"] = q
    spec["nominal_base_height_m"] = .32
    spec["coordinate_schema"] = "full_robot_v201_CAD_links_Rz90_root_v1"
    return robot, spec, dict(max_source_anchor_shift_m=max_anchor_shift,
        root_cad_to_control=transform(rpy=(0., 0., np.pi/2)).tolist(),
        inertia_policy="preserve_CAD_inertia_in_transformed_link_frame",
        loop_pin_source="reviewed_v7_pins_transported_by_zero_frame_registration",
        research_knee_inner_limits_deg=[40., 107.], full_robot_hardware_mapping_verified=False)


def export_bundle(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("Output must be a new directory")
    source = root / "model/wheel_leg_v2.0.1/轮腿整车URDFv2.0.1/urdf/轮腿整车URDFv2.0.1.urdf"
    reference_path = root / "model/纯底盘_v7_flat_base107/urdf/model_spec.json"
    robot, spec, provenance = build_spec(source, json.loads(reference_path.read_text()))
    output.mkdir(parents=True)
    (output / "meshes").mkdir()
    (output / "collisions").mkdir()
    import trimesh
    for body in spec["bodies"]:
        node = robot.find(f"link[@name='{body['name']}']")
        mesh_node = node.find("visual/geometry/mesh")
        name = Path(mesh_node.get("filename")).name
        mesh_source = source.parent.parent / "meshes" / name
        shutil.copyfile(mesh_source, output / body["mesh"])
        for index, collision in enumerate(node.findall("collision")):
            frame = origin(collision.find("origin"))
            if body["name"] == "base_link":
                frame = transform(rpy=(0., 0., np.pi/2)) @ frame
            cylinder = collision.find("geometry/cylinder")
            if cylinder is not None:
                body["collisions"].append(dict(type="cylinder", radius=float(cylinder.get("radius")),
                    length=float(cylinder.get("length")), origin=frame.tolist()))
            else:
                file = f"collisions/{body['name']}_{index}.obj"
                mesh = trimesh.load(mesh_source, force="mesh", process=False)
                mesh.convex_hull.export(output / file)
                body["collisions"].append(dict(type="mesh", file=file, origin=frame.tolist()))
    tools = root / "model/纯底盘_v7_flat_base107/urdf/tools"
    module_spec = importlib.util.spec_from_file_location("v201_usd_export", tools / "mechanism_export_legacy.py")
    exporter = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(exporter)
    exporter.tree_fk = fk
    exporter.export_usd(output, spec)
    # Replace exporter revolute boundaries with their genuine arbitrary-axis
    # definitions, including the two spring prismatic joints.
    from pxr import Gf, Usd, UsdPhysics
    stage = Usd.Stage.Open(str(output / "robot.usda"))
    for joint in spec["joints"]:
        path = "/Robot/tree_joints/" + joint["name"]
        stage.RemovePrim(path)
        prism = joint["type"] == "prismatic"
        usd = (UsdPhysics.PrismaticJoint if prism else UsdPhysics.RevoluteJoint).Define(stage, path)
        usd.CreateBody0Rel().SetTargets(["/Robot/" + joint["parent"]])
        usd.CreateBody1Rel().SetTargets(["/Robot/" + joint["child"]])
        axis = np.array([1., 0., 0.] if prism else [0., 0., 1.])
        alignment = np.eye(4)
        alignment[:3, :3] = Rotation.align_vectors([joint["axis"]], [axis])[0].as_matrix()
        t = np.asarray(joint["origin"])
        def quat(matrix):
            xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            return Gf.Quatf(float(xyzw[3]), Gf.Vec3f(*xyzw[:3]))
        usd.CreateAxisAttr("X" if prism else "Z")
        usd.CreateLocalPos0Attr(Gf.Vec3f(*t[:3, 3]))
        usd.CreateLocalPos1Attr(Gf.Vec3f(0))
        usd.CreateLocalRot0Attr(quat(t @ alignment))
        usd.CreateLocalRot1Attr(quat(alignment))
        usd.CreateExcludeFromArticulationAttr(False)
        if joint["type"] in ("prismatic", "revolute"):
            scale = 1. if prism else 180./np.pi
            usd.CreateLowerLimitAttr(scale*float(joint["limit"]["lower"]))
            usd.CreateUpperLimitAttr(scale*float(joint["limit"]["upper"]))
    stage.GetDefaultPrim().SetCustomData(dict(model_kind=MODEL_KIND, hardware_deployment_ready=False))
    stage.GetRootLayer().Save()
    provenance.update(source_urdf_sha256=digest(source), reference_spec_sha256=digest(reference_path),
        self_collision_policy="disabled_inherited_research_scope",
        other_collisions="per_link_convex_hull; wheel_CAD_cylinders_preserved",
        native_physics_validation="pending", behavior_validation="pending")
    write_json(output / "model_spec.json", spec)
    write_json(output / "provenance.json", provenance)
    manifest = dict(schema="full-robot-v201-plant/v1", model_kind=MODEL_KIND, usd="robot.usda",
        rigid_body_names=[b["name"] for b in spec["bodies"]],
        tree_joint_names=[j["name"] for j in spec["joints"]], control_joint_names=ACTIVE,
        spring_joint_names=SPRINGS, nominal_joint_pos=spec["nominal_joint_pos"],
        total_mass_kg=sum(b["mass"] for b in spec["bodies"]),
        control_frame="Xforward_Yleft_Zup", wheel_radius_m=.06, wheel_reduction=15.62,
        source_urdf_sha256=digest(source), hardware_deployment_ready=False,
        files_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()})
    write_json(output / "manifest.json", manifest)
    return manifest

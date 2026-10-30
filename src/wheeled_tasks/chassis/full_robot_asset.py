"""CPU geometry for the supplied canonical v2.0.2 closed-chain asset.

All input transforms are already X-forward, Y-left, Z-up.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ACTIVE = ["L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"]
SPRINGS = ["L_spring_slide", "R_spring_slide"]
MODEL_KIND = "full_robot_v202_gas_spring_closedchain"


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
    for joint in spec["joints"]:
        if joint["type"] == "fixed":
            coordinates[joint["name"]] = 0.
        elif joint["name"] in prescribed and joint["type"] in ("prismatic", "revolute"):
            value = prescribed[joint["name"]]
            if not float(joint["limit"]["lower"]) - 1e-7 <= value <= float(joint["limit"]["upper"]) + 1e-7:
                raise ValueError(f"Pose outside {joint['name']} limit")
    unknown = [j["name"] for j in spec["joints"] if j["name"] not in prescribed
               and j["type"] != "fixed"]

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

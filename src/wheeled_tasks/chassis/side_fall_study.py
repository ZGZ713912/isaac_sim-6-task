"""Feasible, paired side-fall starts; no policy or motor positioning during physics."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .fallen_pose_bank import family

SCHEMA = "wheel-leg-side-fall-paired-starts-v1"
KNEE_PAIRS = ((45., 45.), (75., 75.), (105., 105.), (45., 105.), (105., 45.))


def orbital_angles(spec, q):
    """Physical wheel-center angle: zero below hip, positive towards vehicle front."""
    from build_full_vehicle_training import geometry
    frames = geometry.fk(spec, q)
    joints = {j["name"]: j for j in spec["joints"]}
    result = []
    for side in ("L", "R"):
        v = frames[side+"_link3"][:3, 3]-np.asarray(joints[side+"_joint1"]["origin"])[:3, 3]
        result.append(math.degrees(math.atan2(v[0], -v[2])))
    return result


def leg_configuration(builder, knees, orbits):
    """Continue from nominal on the physical closed-chain branch, then rotate both arms."""
    from build_full_vehicle_training import geometry
    spec = builder.spec
    q = dict(spec["nominal_joint_pos"])
    zero = spec["passive_knee_zero_inner_rad"]
    initial = (math.degrees(q["L_joint2"]+zero["L_joint2"]),
        math.degrees(zero["R_jonit2"]-q["R_jonit2"]))
    for a in np.linspace(0., 1., 25)[1:]:
        q = builder.solve(dict(L_joint1=q["L_joint1"], R_joint1=q["R_joint1"], L_joint3=0., R_joint3=0.,
            L_joint2=math.radians(initial[0]+a*(knees[0]-initial[0]))-zero["L_joint2"],
            R_jonit2=zero["R_jonit2"]-math.radians(initial[1]+a*(knees[1]-initial[1]))), q)
    angles = orbital_angles(spec, q)
    delta = [math.radians(a-b)/builder.axis_sign(s) for a,b,s in zip(angles, orbits, ("L", "R"))]
    # Small continuation steps avoid jumping to the other four-bar branch.
    steps = max(1, math.ceil(max(abs(v) for v in delta)/.25))
    for _ in range(steps):
        q = builder.rotate(q, *(v/steps for v in delta))
    # Hip pivots in the CAD are slightly offset; common arm rotation is only
    # approximately a rotation of the wheel-center vector about this hip.
    for _ in range(8):
        actual = orbital_angles(spec,q)
        error = [(a-b+180.)%360.-180. for a,b in zip(actual,orbits)]
        if max(abs(v) for v in error) < 1e-6:
            break
        q = builder.rotate(q, *(math.radians(e)/builder.axis_sign(s) for e,s in zip(error,("L","R"))))
    gap = np.linalg.norm(geometry.closure_error(spec, q).reshape(-1, 3), axis=1).max()
    if gap > 1e-7:
        raise ValueError("Side start has an open linkage")
    for joint in spec["joints"]:
        if joint["type"] in ("prismatic", "revolute") and "lower" in joint["limit"]:
            lo, hi = float(joint["limit"]["lower"]), float(joint["limit"]["upper"])
            if not lo-1e-7 <= q[joint["name"]] <= hi+1e-7:
                raise ValueError("Side start exceeds a joint limit")
    for n, binding in spec["spring_binding"].items():
        compression = binding["compression_at_q_zero_m"]-q[n]
        if not -1e-8 <= compression <= binding["stroke_m"]+1e-8:
            raise ValueError("Side start exceeds gas spring travel")
    actual = orbital_angles(spec, q)
    if any(abs((a-b+180.)%360.-180.) > 1e-5 for a,b in zip(actual, orbits)):
        raise ValueError("Physical orbital-angle convention disagrees with FK")
    return q


def placement_fingerprint(start):
    data = {n: start[n] for n in ("root_position_m", "quaternion_xyzw", "full_joint_positions")}
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def build_side_starts(spec, bundle, *, rolls=(60., 75., 90., 105., 120.),
        orbits=(-90., 0., 90.), knee_pairs=KNEE_PAIRS, seed=261031, clearance_m=.002):
    from build_full_vehicle_training import ResetGeometry, geometry
    builder = ResetGeometry(spec, Path(bundle))
    if (not rolls or not orbits or any(not 0 < r < 180 for r in rolls)
            or any(not -180 <= r <= 180 for r in orbits)
            or any(not 40 <= k <= 110 for pair in knee_pairs for k in pair)):
        raise ValueError("Invalid physical side-study grid")
    rng = np.random.default_rng(seed)
    shapes = []
    for knees in knee_pairs:
        for left in orbits:
            for right in orbits:
                q = leg_configuration(builder, knees, (left, right))
                q["M_joint1"] = float(rng.uniform(-math.pi, math.pi))
                shapes.append((knees, (left, right), q))
    starts = []
    for side, sign in (("left", -1.), ("right", 1.)):
        for roll in rolls:
            rotation = Rotation.from_euler("x", sign*roll, degrees=True)
            for shape_id, (knees, angles, joints) in enumerate(shapes):
                q = deepcopy(joints)
                bounds = builder.bounds(q, rotation.as_matrix())
                height = float(-min(v[2] for _,v in bounds)+clearance_m)
                case = len(starts)
                start = dict(name=f"side_{case:04d}", pair_id=case, side=side,
                    roll_magnitude_deg=roll, signed_roll_deg=sign*roll, shape_id=shape_id,
                    knee_pair_deg=list(knees), orbital_pair_deg=list(angles),
                    actual_knee_pair_deg=[math.degrees(q["L_joint2"]+spec["passive_knee_zero_inner_rad"]["L_joint2"]),
                        math.degrees(spec["passive_knee_zero_inner_rad"]["R_jonit2"]-q["R_jonit2"])],
                    leg_shape=f"k{knees[0]:g}_{knees[1]:g}_o{angles[0]:g}_{angles[1]:g}",
                    sampling="closed_chain_side_grid", gimbal_yaw_rad=q["M_joint1"],
                    quaternion_xyzw=rotation.as_quat().tolist(),
                    spawn_family=family(rotation.inv().apply([0., 0., -1.])),
                    root_position_m=[0., 0., height], full_joint_positions=q,
                    min_collision_clearance_m=clearance_m,
                    initial_body_clearance_m={n: min(float(v[2]+height) for b,v in bounds if b == n)
                        for n in {b for b,_ in bounds}},
                    initial_loop_gap_m=float(np.linalg.norm(geometry.closure_error(spec,q).reshape(-1,3),axis=1).max()))
                start["placement_sha256"] = placement_fingerprint(start)
                starts.append(start)
    return starts


def paired(starts):
    result = []
    for start in starts:
        for enabled in (False, True):
            row = deepcopy(start)
            row.update(name=start["name"]+("_moving" if enabled else "_off"), leg_friction_enabled=enabled)
            result.append(row)
    return result


def validate_placements(payload, spec_sha, manifest_sha):
    if (payload.get("schema") != SCHEMA or payload.get("model_spec_sha256") != spec_sha
            or payload.get("asset_manifest_sha256") != manifest_sha or not payload.get("starts")):
        raise ValueError("Side placements need this authenticated asset identity")
    pairs = {}
    for start in payload["starts"]:
        if placement_fingerprint(start) != start["placement_sha256"]:
            raise ValueError("Side placement changed")
        if type(start["leg_friction_enabled"]) is not bool:
            raise ValueError("Missing explicit leg-loss arm")
        pairs.setdefault(start["pair_id"], []).append(start)
    for rows in pairs.values():
        if (len(rows) != 2 or {r["leg_friction_enabled"] for r in rows} != {False, True}
                or len({r["placement_sha256"] for r in rows}) != 1):
            raise ValueError("Need exactly matching off/moving starts for every pair")
    return payload["starts"]

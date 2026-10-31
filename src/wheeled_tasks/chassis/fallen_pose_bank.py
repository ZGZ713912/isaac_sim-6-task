"""Passive full-vehicle placement, admission and yaw-invariant rest-pose groups.

The bank contains measured PhysX endpoints, not geometrically guessed rests.
No policy, optimizer or scripted get-up trajectory is involved in collection.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = "wheel-leg-passive-fallen-bank-v1"
SHAPES = ("standard", "balanced", "folded", "extended", "asymmetric_left", "asymmetric_right", "split")
SETTLE = dict(min_seconds=1., max_seconds=20., hold_seconds=1.,
    body_linear_speed_m_s=.02, body_angular_speed_rad_s=.1,
    rotary_speed_rad_s=.15, slider_speed_m_s=.005,
    minimum_support_weight_fraction=.8, maximum_support_weight_fraction=1.2,
    loop_gap_m=.003, ground_penetration_m=.003)
WINDOW = dict(seconds=2., root_translation_range_m=.005, gravity_range_deg=1.,
    rotary_position_range_rad=.05, slider_position_range_m=.001,
    body_linear_speed_rms_m_s=.02, body_angular_speed_rms_rad_s=.1,
    rotary_speed_rms_rad_s=.15, slider_speed_rms_m_s=.005,
    mean_support_weight_fraction=[.8, 1.2])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family(gravity):
    """Use measured body-frame gravity, independent of the spawn label/yaw."""
    g = np.asarray(gravity, dtype=float)
    g = g / np.linalg.norm(g)
    if g[2] < -math.cos(math.radians(30)):
        return "upright"
    if g[2] > math.cos(math.radians(45)):
        return "inverted"
    if abs(abs(g[0])-abs(g[1])) < .15:
        return "compound"
    if abs(g[0]) > abs(g[1]):
        return "front" if g[0] > 0 else "back"
    # Canonical body frame: X forward,Y left,Z up. Gravity towards+Y
    # means the left side is down; positive roll instead puts the right down.
    return "left" if g[1] > 0 else "right"


def posture_class(pose):
    f = family(pose["gravity_body"])
    if f == "upright":
        support = set(pose["support_bodies"])
        return "wheel_supported" if support and support <= {"L_link3", "R_link3"} else "belly_down"
    return dict(front="front_down", back="rear_down", left="left_side", right="right_side",
        inverted="roof_down", compound="diagonal")[f]


def orientations(tilt_step_deg=45., azimuth_step_deg=45.):
    if (not 0 < tilt_step_deg <= 90 or not 0 < azimuth_step_deg <= 90
            or not math.isclose(180/tilt_step_deg, round(180/tilt_step_deg))
            or not math.isclose(360/azimuth_step_deg, round(360/azimuth_step_deg))):
        raise ValueError("Angle steps must divide180/360 degrees and be in(0,90]")
    result = []
    for tilt in np.linspace(0., 180., round(180/tilt_step_deg)+1):
        azimuths = [0.] if tilt in (0., 180.) else np.arange(0., 360., azimuth_step_deg)
        for azimuth in azimuths:
            # Horizontal rotation axis gives the full sphere of gravity
            # directions. Heading duplicates on a homogeneous floor are omitted.
            a = math.radians(azimuth)
            rot = Rotation.from_rotvec(np.array([math.sin(a), math.cos(a), 0.])*math.radians(tilt))
            g = rot.inv().apply([0., 0., -1.])
            result.append(dict(tilt_deg=float(tilt), azimuth_deg=float(azimuth),
                quaternion_xyzw=rot.as_quat().tolist(), spawn_family=family(g)))
    return result


def build_starts(spec, bundle, base_bank, shapes=SHAPES, *, tilt_step_deg=45.,
                 azimuth_step_deg=45., clearance_m=.002, random_count=0, seed=261031):
    from build_full_vehicle_training import ResetGeometry, geometry
    if not 0 < clearance_m <= .01 or not shapes or set(shapes)-set(SHAPES):
        raise ValueError("Need known leg shapes and a near-ground positive clearance <=1cm")
    builder = ResetGeometry(spec, Path(bundle))
    nominal = spec["nominal_joint_pos"]
    standard = next(p["full_joint_positions"] for p in base_bank["poses"] if p["name"].startswith("floor_knee103"))
    variants = {}
    for name in shapes:
        if name == "standard":
            q = deepcopy(standard)
        elif name == "balanced":
            q = builder.balanced(75.)[0]
        elif name in ("folded", "extended"):
            q = builder.knee(45. if name == "folded" else 103.)
        elif name == "split":
            q = builder.rotate(builder.knee(75.), .65, .65)
        else:
            left, right = (50., 100.) if name == "asymmetric_left" else (100., 50.)
            q = dict(nominal)
            zero = spec["passive_knee_zero_inner_rad"]
            b0 = [math.degrees(q["L_joint2"]+zero["L_joint2"]),
                  math.degrees(zero["R_jonit2"]-q["R_jonit2"])]
            for alpha in np.linspace(0., 1., 25)[1:]:
                q = builder.solve(dict(L_joint1=q["L_joint1"], R_joint1=q["R_joint1"], L_joint3=0., R_joint3=0.,
                    L_joint2=math.radians(b0[0]+alpha*(left-b0[0]))-zero["L_joint2"],
                    R_jonit2=zero["R_jonit2"]-math.radians(b0[1]+alpha*(right-b0[1]))), q)
        gap = float(np.linalg.norm(geometry.closure_error(spec, q).reshape(-1, 3), axis=1).max())
        if gap > 1e-7:
            raise ValueError(f"Unclosed start: {name}, {gap}")
        for n, binding in spec["spring_binding"].items():
            compression = binding["compression_at_q_zero_m"]-q[n]
            if not 0 <= compression <= binding["stroke_m"]:
                raise ValueError("Start exceeds gas-spring travel")
        variants[name] = q
    if type(random_count) is not int or random_count < 0:
        raise ValueError("Random placement count must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    placements = [(orientation, name) for orientation in orientations(tilt_step_deg, azimuth_step_deg) for name in variants]
    for _ in range(random_count):
        rotation = Rotation.random(random_state=rng)
        gravity = rotation.inv().apply([0., 0., -1.])
        placements.append((dict(tilt_deg=float(np.rad2deg(np.arccos(np.clip(-gravity[2], -1., 1.)))),
            azimuth_deg=None, quaternion_xyzw=rotation.as_quat().tolist(), spawn_family=family(gravity),
            sampling="uniform_SO3"), str(rng.choice(list(variants)))))
    starts = []
    for orientation, name in placements:
        rot = Rotation.from_quat(orientation["quaternion_xyzw"])
        q = deepcopy(variants[name])
        q["M_joint1"] = float(rng.uniform(-math.pi, math.pi))
        minimum = min(v[2] for _, v in builder.bounds(q, rot.as_matrix()))
        starts.append(dict(name=f"placement_{len(starts):05d}_{name}",
            leg_shape=name, gimbal_yaw_rad=q["M_joint1"], **orientation,
            root_position_m=[0., 0., float(-minimum+clearance_m)],
            full_joint_positions=q, min_collision_clearance_m=clearance_m,
            initial_loop_gap_m=float(np.linalg.norm(geometry.closure_error(spec, q).reshape(-1, 3), axis=1).max())))
    return starts


def rest_mask(body_velocity, joint_velocity, slider_ids, support_z, weight, cfg=None):
    """All bodies/joints must remain slow; a slow base alone is insufficient."""
    import torch
    cfg = cfg or SETTLE
    limits = torch.full_like(joint_velocity, cfg["rotary_speed_rad_s"])
    limits[:, slider_ids] = cfg["slider_speed_m_s"]
    return (torch.isfinite(body_velocity).all(dim=(-2, -1)) & torch.isfinite(joint_velocity).all(-1)
        & (body_velocity[..., :3].norm(dim=-1).amax(-1) <= cfg["body_linear_speed_m_s"])
        & (body_velocity[..., 3:].norm(dim=-1).amax(-1) <= cfg["body_angular_speed_rad_s"])
        & (joint_velocity.abs() <= limits).all(-1)
        & (support_z >= weight*cfg["minimum_support_weight_fraction"])
        & (support_z <= weight*cfg["maximum_support_weight_fraction"]))


def chassis_rest_mask(body_velocity, joint_velocity, body_names, joint_names, support_z, weight, cfg=None):
    """Separate diagnostic: chassis/legs at rest while the free gimbal may move.

    This never admits a fully static endpoint. Gimbal velocity and configuration
    remain in the raw record, including when its contact supports the chassis.
    """
    bodies = [i for i, n in enumerate(body_names) if n not in ("M_link1", "M_link2")]
    joints = [i for i, n in enumerate(joint_names) if n not in ("M_joint1", "M_joint2")]
    sliders = [i for i, j in enumerate(joints) if joint_names[j] in ("L_spring_slide", "R_spring_slide")]
    return rest_mask(body_velocity[:, bodies], joint_velocity[:, joints], sliders, support_z, weight, cfg)


def window_rest(trace, columns, joint_names, sample_dt=.01, *, chassis_only=False):
    """Kinematic window diagnosis handles contact-force solver chatter explicitly.

    Keep the stricter instantaneous hold separately; do not overwrite its result.
    """
    count = round(WINDOW["seconds"]/sample_dt)
    if len(trace) < count:
        return dict(passed=False, reason="short_window")
    a = np.asarray(trace[-count:], dtype=float)
    if not np.isfinite(a).all():
        return dict(passed=False, reason="nonfinite")
    take = lambda name: a[:, columns.index(name)]
    names = [n for n in joint_names if not chassis_only or n not in ("M_joint1", "M_joint2")]
    q = np.stack([take("q/"+n) for n in names], -1)
    v = np.stack([take("dq/"+n) for n in names], -1)
    sliders = np.array([n in ("L_spring_slide", "R_spring_slide") for n in names])
    qrange = np.ptp(q, axis=0)
    vrms = np.sqrt(np.mean(v*v, axis=0))
    root = np.stack([take("root_"+n) for n in "xyz"], -1)
    rotations = Rotation.from_quat(np.stack([take("quat_"+n) for n in "xyzw"], -1))
    gravity = rotations.inv().apply(np.tile([0., 0., -1.], (count, 1)))
    gravity_range = float(np.rad2deg(np.arccos(np.clip(gravity@gravity[-1], -1., 1.))).max())
    prefix = "chassis_" if chassis_only else "body_"
    linear = float(np.sqrt(np.mean(take(prefix+"linear_speed_max_m_s")**2)))
    angular = float(np.sqrt(np.mean(take(prefix+"angular_speed_max_rad_s")**2)))
    support = float(take("support_weight_fraction").mean())
    translation = float(np.linalg.norm(np.ptp(root, axis=0)))
    passed = (translation <= WINDOW["root_translation_range_m"] and gravity_range <= WINDOW["gravity_range_deg"]
        and linear <= WINDOW["body_linear_speed_rms_m_s"] and angular <= WINDOW["body_angular_speed_rms_rad_s"]
        and np.all(qrange <= np.where(sliders, WINDOW["slider_position_range_m"], WINDOW["rotary_position_range_rad"]))
        and np.all(vrms <= np.where(sliders, WINDOW["slider_speed_rms_m_s"], WINDOW["rotary_speed_rms_rad_s"]))
        and WINDOW["mean_support_weight_fraction"][0] <= support <= WINDOW["mean_support_weight_fraction"][1])
    return dict(passed=bool(passed), seconds=WINDOW["seconds"], chassis_only=chassis_only,
        root_translation_range_m=translation, gravity_range_deg=gravity_range,
        body_linear_speed_rms_m_s=linear, body_angular_speed_rms_rad_s=angular,
        mean_support_weight_fraction=support,
        joint_position_ranges=dict(zip(names, qrange.tolist())), joint_speed_rms=dict(zip(names, vrms.tolist())))


def native_clearance(spec, meshes, body_names, body_poses):
    """Ground clearance of collision hulls transformed by measured body poses."""
    by_name = dict(zip(body_names, body_poses))
    minimum = math.inf
    for body in spec["bodies"]:
        value = by_name[body["name"]]
        world = np.eye(4)
        world[:3, :3] = Rotation.from_quat(value[3:]).as_matrix()
        world[:3, 3] = value[:3]
        for collision in body["collisions"]:
            frame = world@np.array(collision["origin"])
            if collision["type"] == "mesh":
                z = (meshes[collision["file"]]@frame[:3, :3].T+frame[:3, 3])[:, 2].min()
            else:
                axis_z = frame[2, 2]
                z = frame[2, 3]-collision["radius"]*math.sqrt(max(0., 1.-axis_z*axis_z))-collision["length"]*.5*abs(axis_z)
            minimum = min(minimum, float(z))
    return minimum


def group_poses(records, joint_names, *, angle_deg=12., joint_rad=.25, height_m=.015):
    """Greedy measured medoids; retain every raw endpoint and group membership.

    Whole-vehicle yaw, XY translation and driven-wheel phase do not distinguish
    static configurations on a homogeneous flat floor. Gimbal yaw is retained.
    """
    names = [n for n in joint_names if n not in ("L_joint3", "R_joint3", "L_spring_slide", "R_spring_slide")]
    groups = []
    for record in records:
        if record["status"] != "settled":
            continue
        p = record["final_pose"]
        g = np.array(p["gravity_body"])
        q = np.array([p["full_joint_positions"][n] for n in names])
        for group in groups:
            medoid = records[group["representative_record"]]["final_pose"]
            g0 = np.array(medoid["gravity_body"])
            q0 = np.array([medoid["full_joint_positions"][n] for n in names])
            dq = np.arctan2(np.sin(q-q0), np.cos(q-q0))
            if (p["family"] == medoid["family"] and p["support_bodies"] == medoid["support_bodies"]
                    and float(g@g0) >= math.cos(math.radians(angle_deg))
                    and np.abs(dq).max() <= joint_rad
                    and abs(p["root_position_m"][2]-medoid["root_position_m"][2]) <= height_m):
                group["record_indices"].append(record["index"])
                break
        else:
            groups.append(dict(representative_record=record["index"], record_indices=[record["index"]]))
    return groups


def validate_bank(bank, manifest_sha, spec_sha, joint_names):
    if (bank.get("schema") != SCHEMA or bank.get("status") != "completed"
            or bank.get("asset_manifest_sha256") != manifest_sha or bank.get("model_spec_sha256") != spec_sha
            or set(bank.get("tree_joint_names", [])) != set(joint_names)
            or bank.get("motor_effort_peak_nm") != 0.
            or bank.get("all_eight_motor_servos_disabled") is not True
            or bank.get("passive_gas_springs") != "retained"
            or bank.get("state_writes_during_physics") != 0):
        raise ValueError("Need a completed, matching, fully disabled native fallen-pose bank")
    if bank.get("settle_contract") != SETTLE or not bank.get("poses"):
        raise ValueError("No admitted static poses or changed settling criteria")
    if len({p["name"] for p in bank["poses"]}) != len(bank["poses"]):
        raise ValueError("Duplicate measured pose names")
    for pose in bank["poses"]:
        if (set(pose["full_joint_positions"]) != set(joint_names) or pose.get("native_settled") is not True
                or pose["gap_m"] > SETTLE["loop_gap_m"]
                or pose["min_collision_clearance_m"] < -SETTLE["ground_penetration_m"]
                or pose.get("stable_hold_seconds", 0.) < SETTLE["hold_seconds"]
                or not pose.get("support_bodies")):
            raise ValueError("Invalid measured static endpoint")
        values = [*pose["root_position_m"], *pose["root_quaternion_xyzw"], *pose["full_joint_positions"].values()]
        if (not all(math.isfinite(v) for v in values)
                or not math.isclose(sum(v*v for v in pose["root_quaternion_xyzw"]), 1., abs_tol=2e-5)):
            raise ValueError("Nonfinite pose or nonunit native quaternion")
    return bank

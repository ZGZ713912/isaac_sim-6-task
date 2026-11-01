"""Mechanical placements, actual passive motion and failed-bank rejection."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from wheeled_tasks.chassis import fallen_pose_bank as task

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"model/wheel_leg_v2.0.2/training"


def test_gravity_sphere_has_both_signs_and_no_heading_duplicates():
    rotations = task.orientations(45., 45.)
    directions = np.array([Rotation.from_quat(r["quaternion_xyzw"]).inv().apply([0., 0., -1.]) for r in rotations])
    assert len(rotations) == 26
    assert len(np.unique(np.round(directions, 8), axis=0)) == len(rotations)
    assert set(task.family(g) for g in directions) == {"upright", "front", "back", "left", "right", "inverted", "compound"}
    for axis in range(3):
        assert directions[:, axis].min() < -.99 and directions[:, axis].max() > .99


def test_side_names_follow_the_canonical_left_axis_and_belly_is_not_standing():
    assert task.family([0., 1., 0.]) == "left"
    assert task.family([0., -1., 0.]) == "right"
    assert task.posture_class(dict(gravity_body=[0.,0.,-1.], support_bodies=["base_link"])) == "belly_down"
    assert task.posture_class(dict(gravity_body=[0.,0.,-1.], support_bodies=["L_link3","R_link3"])) == "wheel_supported"


@pytest.mark.parametrize("tilt,azimuth", [(0., 45.), (70., 45.), (45., 100.), (float("nan"), 45.)])
def test_invalid_angle_grids_do_not_silently_omit_directions(tilt, azimuth):
    with pytest.raises((ValueError, OverflowError)):
        task.orientations(tilt, azimuth)


def test_full_vehicle_starts_are_closed_and_clear_at_every_orientation():
    from build_full_vehicle_training import ResetGeometry, geometry
    spec = json.loads((BUNDLE/"model_spec.json").read_text())
    bank = json.loads((BUNDLE/"pose_bank.json").read_text())
    starts = task.build_starts(spec, BUNDLE, bank, tilt_step_deg=90., azimuth_step_deg=90.)
    builder = ResetGeometry(spec, BUNDLE)
    assert len(starts) == 6*len(task.SHAPES)
    for start in starts:
        q = start["full_joint_positions"]
        assert np.linalg.norm(geometry.closure_error(spec, q).reshape(-1, 3), axis=1).max() < 1e-7
        rotation = Rotation.from_quat(start["quaternion_xyzw"]).as_matrix()
        assert start["root_position_m"][2]+min(v[2] for _, v in builder.bounds(q, rotation)) == pytest.approx(.002, abs=1e-9)
        if start["leg_shape"].startswith("asymmetric"):
            zero = spec["passive_knee_zero_inner_rad"]
            left = np.rad2deg(q["L_joint2"]+zero["L_joint2"])
            right = np.rad2deg(zero["R_jonit2"]-q["R_jonit2"])
            assert abs(left-right) == pytest.approx(50., abs=1e-4)


def test_rest_gate_includes_free_gimbal_sliders_and_support():
    bodies = torch.zeros(5, 21, 6)
    joints = torch.zeros(5, 20)
    weight = torch.full((5,), 210.)
    force = weight.clone()
    # A still base with a moving gimbal/link is not settled.
    bodies[1, -1, 4] = .2
    joints[2, 16] = .006
    force[3] = 0.
    bodies[4, 2, 0] = float("nan")
    assert task.rest_mask(bodies, joints, [16, 17], force, weight).tolist() == [True, False, False, False, False]


def fixture_bank():
    base = json.loads((BUNDLE/"pose_bank.json").read_text())
    pose = deepcopy(base["poses"][3])
    pose.update(name="test_measured", native_settled=True, support_bodies=["base_link"], stable_hold_seconds=1.1,
        gravity_body=[0., 0., -1.], collector_record_index=0,
        full_joint_velocities=dict.fromkeys(base["tree_joint_names"], 0.), root_com_velocity_world=[0.]*6)
    bank = dict(schema=task.SCHEMA, status="completed", asset_manifest_sha256=base["asset_manifest_sha256"],
        model_spec_sha256=base["model_spec_sha256"], tree_joint_names=base["tree_joint_names"],
        all_eight_motor_servos_disabled=True, motor_effort_peak_nm=0., passive_gas_springs="retained",
        state_writes_during_physics=0, settle_contract=deepcopy(task.SETTLE), poses=[pose], family_counts={"upright":1})
    return base, bank


@pytest.mark.parametrize("change", [dict(status="prepared"), dict(status="failed"), dict(poses=[]),
    dict(motor_effort_peak_nm=.001), dict(all_eight_motor_servos_disabled=False),
    dict(passive_gas_springs="disabled"), dict(state_writes_during_physics=1), dict(model_spec_sha256="other")])
def test_unmeasured_or_wrong_mechanism_banks_are_rejected(change):
    base, bank = fixture_bank()
    bank.update(change)
    with pytest.raises(ValueError):
        task.validate_bank(bank, base["asset_manifest_sha256"], base["model_spec_sha256"], base["tree_joint_names"])


@pytest.mark.parametrize("change", [dict(native_settled=False), dict(stable_hold_seconds=.99), dict(support_bodies=[]),
    dict(gap_m=.004), dict(min_collision_clearance_m=-.004), dict(root_quaternion_xyzw=[0.,0.,0.,2.])])
def test_invalid_static_endpoints_are_rejected(change):
    base, bank = fixture_bank()
    bank["poses"][0].update(change)
    with pytest.raises(ValueError):
        task.validate_bank(bank, base["asset_manifest_sha256"], base["model_spec_sha256"], base["tree_joint_names"])


def test_groups_ignore_world_yaw_and_wheel_phase_but_keep_leg_gimbal_and_support_changes():
    _, bank = fixture_bank()
    pose = bank["poses"][0]
    poses = [deepcopy(pose) for _ in range(5)]
    poses[1]["root_quaternion_xyzw"] = Rotation.from_euler("z", 2.).as_quat().tolist()
    poses[1]["full_joint_positions"]["L_joint3"] += 4.
    poses[2]["full_joint_positions"]["L_joint1"] += .5
    poses[3]["full_joint_positions"]["M_joint1"] += .5
    poses[4]["support_bodies"] = ["M_link2"]
    records = [dict(index=i, status="settled", final_pose=p) for i,p in enumerate(poses)]
    groups = task.group_poses(records, bank["tree_joint_names"])
    assert [g["record_indices"] for g in groups] == [[0,1], [2], [3], [4]]



def test_random_body_angles_and_gimbal_yaw_are_seeded_and_reproducible():
    spec = json.loads((BUNDLE/"model_spec.json").read_text())
    bank = json.loads((BUNDLE/"pose_bank.json").read_text())
    a = task.build_starts(spec, BUNDLE, bank, ("standard",), tilt_step_deg=90., azimuth_step_deg=90., random_count=10, seed=7)
    b = task.build_starts(spec, BUNDLE, bank, ("standard",), tilt_step_deg=90., azimuth_step_deg=90., random_count=10, seed=7)
    assert a == b and len(a) == 16
    yaw = np.array([p["gimbal_yaw_rad"] for p in a])
    assert np.ptp(yaw) > 4. and (np.abs(yaw) <= np.pi).all()
    assert len({tuple(p["quaternion_xyzw"]) for p in a}) == len(a)


def test_chassis_diagnostic_never_hides_moving_legs_or_admits_full_vehicle_rest():
    names = ["base_link", "L_link1", "M_link1", "M_link2"]
    joints = ["L_joint1", "L_spring_slide", "M_joint1", "M_joint2"]
    bv, qv = torch.zeros(3, 4, 6), torch.zeros(3, 4)
    bv[:, 3, 5] = 2.
    qv[:, 2] = 2.
    bv[1, 1, 0] = .03
    qv[2, 1] = .006
    weight = torch.full((3,), 210.)
    assert task.chassis_rest_mask(bv, qv, names, joints, weight, weight).tolist() == [True, False, False]
    assert not task.rest_mask(bv, qv, [1], weight, weight).any()


def test_native_clearance_uses_actual_link_poses_including_gimbal():
    from build_full_vehicle_training import ResetGeometry, geometry
    spec = json.loads((BUNDLE/"model_spec.json").read_text())
    builder = ResetGeometry(spec, BUNDLE)
    q = dict(spec["nominal_joint_pos"], M_joint1=1.4, M_joint2=.2)
    rot = Rotation.from_euler("xyz", [.7, -1.1, .3])
    root = np.eye(4)
    root[:3, :3] = rot.as_matrix()
    root[2, 3] = 1.
    frames = geometry.fk(spec, q, root)
    names = list(frames)
    poses = [np.r_[frames[n][:3, 3], Rotation.from_matrix(frames[n][:3, :3]).as_quat()] for n in names]
    measured = task.native_clearance(spec, builder.meshes, names, poses)
    assert measured == pytest.approx(1.+min(v[2] for _, v in builder.bounds(q, rot.as_matrix())), abs=1e-9)
    # A later native pose drift must be seen without reconstructing a closed tree.
    for p in poses:
        p[2] -= .1
    assert task.native_clearance(spec, builder.meshes, names, poses) == pytest.approx(measured-.1, abs=1e-9)


def test_window_diagnosis_tolerates_contact_chatter_but_rejects_real_motion():
    names = ["L_joint1", "L_spring_slide", "M_joint1", "M_joint2"]
    columns = ["root_"+c for c in "xyz"]+["quat_"+c for c in "xyzw"]+[
        "q/"+n for n in names]+["dq/"+n for n in names]+[
        "body_linear_speed_max_m_s", "body_angular_speed_max_rad_s", "support_weight_fraction",
        "chassis_linear_speed_max_m_s", "chassis_angular_speed_max_rad_s"]
    a = np.zeros((200, len(columns)))
    a[:, columns.index("quat_w")] = 1.
    a[:, columns.index("support_weight_fraction")] = np.tile([.6, 1.4], 100)
    assert task.window_rest(a, columns, names)["passed"]
    assert not task.window_rest(a[:100], columns, names)["passed"]
    a[:, columns.index("q/M_joint1")] = np.linspace(0., .2, len(a))
    assert not task.window_rest(a, columns, names)["passed"]
    assert task.window_rest(a, columns, names, chassis_only=True)["passed"]
    a[:, columns.index("q/L_spring_slide")] = np.linspace(0., .002, len(a))
    assert not task.window_rest(a, columns, names, chassis_only=True)["passed"]

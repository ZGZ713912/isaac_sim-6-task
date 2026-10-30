"""Full vehicle admission, feasible births, deployable odometry and gimbal loads."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from pxr import Usd, UsdPhysics
from scipy.spatial.transform import Rotation

from train_prone_motion import configuration
from wheeled_tasks.chassis.full_robot_v202 import digest
from wheeled_tasks.chassis.full_vehicle_training import authenticate
from wheeled_tasks.chassis.gimbal_motion import GimbalMotion, motion_contract
from wheeled_tasks.chassis.standing_local_odometry import LocalStandingOdometry

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"model/wheel_leg_v2.0.2/training"
PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4.json").read_text())


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    return configuration(tmp_path_factory.mktemp("full_vehicle"), PLAN, num_envs=80, performance_backend="eager")


def test_full_vehicle_is_the_default_v4_with_a_fresh_six_motor_policy(prepared):
    from wheeled_tasks.chassis.prone_motion_cli import train_parser, training_options
    args = train_parser().parse_args([])
    _, default, _ = training_options(args, ROOT)
    cfg, settings, bank = prepared
    assert default["full_vehicle"] == PLAN["full_vehicle"]
    assert (cfg["actor_dim"], cfg["critic_dim"], cfg["action_dim"]) == (35, 87, 6)
    assert cfg["asset_directory"] == str(BUNDLE.relative_to(ROOT))
    manifest = json.loads((BUNDLE/"manifest.json").read_text())
    assert set(manifest["nominal_joint_pos"]) == set(manifest["tree_joint_names"])
    assert bank == BUNDLE/"pose_bank.json"
    assert settings["standing_odometry"] == PLAN["standing_odometry"]
    assert cfg["v5_control"]["leg_kp"] == 160. and cfg["v5_control"]["leg_kd"] == 2.5
    proof = authenticate(cfg, ROOT)
    assert proof["status"] == "authenticated_geometry_derivation"
    assert proof["moving_gimbal_behavior_qualified"] is False


@pytest.mark.parametrize("field,value", [("gimbal_motion", {}), ("zero_friction_bodies", ["L_link3"]),
    ("critic_dim", 83), ("asset_manifest_sha256", "0"*64)])
def test_full_vehicle_rejects_unbound_configuration_changes(prepared, field, value):
    cfg = deepcopy(prepared[0])
    cfg[field] = value
    with pytest.raises(ValueError):
        authenticate(cfg, ROOT)


def test_actual_layer_releases_gimbal_and_keeps_all_six_external_loops():
    stage = Usd.Stage.Open(str(BUNDLE/"robot.usda"))
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]) == 21
    assert len([p for p in stage.Traverse() if p.IsA(UsdPhysics.SphericalJoint)]) == 6
    pitch = UsdPhysics.RevoluteJoint.Get(stage, "/Robot/tree_joints/M_joint2")
    yaw = UsdPhysics.RevoluteJoint.Get(stage, "/Robot/tree_joints/M_joint1")
    assert (pitch.GetLowerLimitAttr().Get(), pitch.GetUpperLimitAttr().Get()) == (-30., 35.)
    assert yaw.GetLowerLimitAttr().Get() == -float("inf") and yaw.GetUpperLimitAttr().Get() == float("inf")


def test_all_full_vehicle_births_have_closed_linkages_and_no_terrain_penetration():
    definition = importlib.util.spec_from_file_location("training_geometry_test", ROOT/"scripts/build_full_vehicle_training.py")
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    spec = json.loads((BUNDLE/"model_spec.json").read_text())
    geometry = module.ResetGeometry(spec, ROOT/"model/wheel_leg_v2.0.2/isaac")
    bank = json.loads((BUNDLE/"pose_bank.json").read_text())
    for p in bank["poses"]:
        q = p["full_joint_positions"]
        gap = np.linalg.norm(module.geometry.closure_error(spec, q).reshape(-1, 3), axis=1).max()
        assert gap < 1e-7
        rotation = Rotation.from_quat(p["root_quaternion_xyzw"]).as_matrix()
        bounds = geometry.bounds(q, rotation)
        minimum = min(v[2]+p["root_position_m"][2] for _, v in bounds)
        assert minimum >= -1e-7
        assert q["M_joint1"] == q["M_joint2"] == 0.
    for name, height in (("stand_low", .23), ("stand", .305), ("stand_high", .42)):
        p = bank["poses"][bank["diagnostic_height_starts"]["mapping"][name]]
        assert abs(p["root_position_m"][2]-height) < 1e-8


def test_full_vehicle_odometry_captures_stationary_reference_and_rejects_wrong_geometry():
    cfg = PLAN["standing_odometry"]
    odometry = LocalStandingOdometry(1, "cpu", .01, cfg, geometry_directory=BUNDLE)
    bank = json.loads((BUNDLE/"pose_bank.json").read_text())
    p = bank["poses"][1]["full_joint_positions"]
    nominal = odometry.coordinate.geometry.nominal
    q = torch.tensor([[p[n] for n in ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")]])
    enabled = torch.ones(1, dtype=torch.bool)
    for tick in range(30):
        frame = torch.zeros(1, 35)
        frame[:, 9] = -1.
        frame[:, 10:14] = q-nominal
        frame[:, 14:16] = odometry.local_wheel_angles(frame, torch.zeros(1, 2), enabled, tick)
        frame = odometry.update(frame, nominal, enabled, tick)
    assert odometry.valid.all() and odometry.reference_active.all()
    assert frame[0, 31] == 1. and torch.equal(frame[:, 14:16], torch.zeros(1, 2))
    bad = deepcopy(cfg)
    bad["geometry_model_spec_sha256"] = "0"*64
    with pytest.raises(ValueError, match="geometry"):
        LocalStandingOdometry(1, "cpu", .01, bad, geometry_directory=BUNDLE)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA absent"))])
def test_gimbal_motion_scans_both_pitch_limits_and_pauses_disabled_lanes(device):
    g = GimbalMotion(32, device, .01, motion_contract(), 103)
    q, dq = torch.zeros(32, 2, device=device), torch.zeros(32, 2, device=device)
    enabled = torch.arange(32, device=device) % 2 == 0
    minimum, maximum = 0., 0.
    for _ in range(1700):
        effort = g.step(q, dq, enabled)
        assert (effort.abs() <= effort.new_tensor([2., 10.])).all()
        assert (effort[~enabled] == 0.).all()
        minimum = min(minimum, float(g.target[enabled, 1].min()))
        maximum = max(maximum, float(g.target[enabled, 1].max()))
    assert abs(minimum-math.radians(-30.)) < .001
    assert abs(maximum-math.radians(35.)) < .001
    assert (g.time[~enabled] == 0.).all()
    assert (g.yaw_velocity[~enabled] == 0.).all()
    assert (g.yaw_goal < 0.).any() and (g.yaw_goal > 0.).any()


def test_gimbal_rng_is_separate_from_policy_and_partial_resets_are_local():
    torch.manual_seed(91)
    expected = torch.rand(7)
    torch.manual_seed(91)
    g = GimbalMotion(4, "cpu", .001, motion_contract(), 103)
    torch.testing.assert_close(torch.rand(7), expected, atol=0., rtol=0.)
    before = g.yaw_goal.clone()
    g.reset(torch.tensor([1]))
    assert torch.equal(before[[0, 2, 3]], g.yaw_goal[[0, 2, 3]])
    same = GimbalMotion(4, "cpu", .001, motion_contract(), 103)
    assert torch.equal(before, same.yaw_goal)


def test_legacy_v3_remains_the_chassis_recipe(tmp_path):
    plan = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v3.json").read_text())
    cfg, settings, bank = configuration(tmp_path, plan, num_envs=80)
    assert cfg["critic_dim"] == 83 and "full_vehicle" not in cfg
    assert "纯底盘" in cfg["asset_directory"] and "standing_odometry" not in settings
    assert "height_matched073" in bank.name

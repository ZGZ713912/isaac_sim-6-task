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
PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4.json").read_text())
BUNDLE = (ROOT/PLAN["mechanism"]["file"]).parent


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
    assert default["mechanical_inner_knee_range_deg"] == [40.,110.]
    assert default["encoder_zero_inner_knee_deg"] == 110.
    manifest = json.loads((BUNDLE/"manifest.json").read_text())
    assert set(manifest["nominal_joint_pos"]) == set(manifest["tree_joint_names"])
    assert bank == ROOT/PLAN["pose_bank"]["file"]
    assert default["schema"] == "wheel-leg-prone-motion-v4"
    assert default["multi_pose_recovery"] == PLAN["multi_pose_recovery"]
    assert settings["standing_odometry"] == PLAN["standing_odometry"]
    assert cfg["v5_control"]["leg_kp"] == 160. and cfg["v5_control"]["leg_kd"] == 2.5
    proof = authenticate(cfg, ROOT)
    assert proof["status"] == "authenticated_geometry_derivation"
    assert proof["knee_inner_limits_deg"] == [40.,110.]
    assert proof["hardware_zero_reference_inner_deg"] == 110.
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
    spec = json.loads((BUNDLE/"model_spec.json").read_text())
    for name,sign in (("L_joint2",1.),("R_jonit2",-1.)):
        joint = next(j for j in spec["joints"] if j["name"] == name)
        usd = UsdPhysics.RevoluteJoint.Get(stage,"/Robot/tree_joints/"+name)
        angles = sorted(sign*np.deg2rad(v)+spec["passive_knee_zero_inner_rad"][name]
            for v in (usd.GetLowerLimitAttr().Get(),usd.GetUpperLimitAttr().Get()))
        np.testing.assert_allclose(np.rad2deg(angles),[40.,110.],atol=2e-6,rtol=0.)


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


def test_full_vehicle_odometry_captures_stationary_reference_and_rejects_wrong_geometry(monkeypatch):
    # A full-vehicle run must not secretly load a historical chassis table
    # before replacing it with its admitted geometry.
    def forbidden_chassis_table():
        raise AssertionError("full-vehicle odometry read the historical chassis lookup")
    monkeypatch.setattr("wheeled_algo.standing_displacement.frozen_rolling_estimator", forbidden_chassis_table)
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


def test_fresh_native_ppo_accepts_the_larger_critic_without_loading_weights():
    from types import SimpleNamespace
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO
    from train_prone_motion import learner_config
    from wheeled_algo.scratch_standing import initialize_fresh
    obs = TensorDict(dict(policy=torch.zeros(8, 35), critic=torch.zeros(8, 87)), batch_size=[8])
    cfg = learner_config(PLAN, "cpu")
    cfg["multi_gpu"] = None
    algorithm = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=8, num_actions=6), cfg, "cpu")
    proof = initialize_fresh(algorithm, torch.zeros(6), initial_std=PLAN["initial_std"])
    actor_input = next(m for m in algorithm.actor.mlp.modules() if isinstance(m, torch.nn.Linear))
    critic_input = next(m for m in algorithm.critic.mlp.modules() if isinstance(m, torch.nn.Linear))
    assert (actor_input.in_features, critic_input.in_features) == (35, 87)
    assert proof["checkpoint_loaded"] is False and proof["teacher_loaded"] is False
    assert proof["optimizer_states"] == 0 and not algorithm.optimizer.state
    assert all(p.requires_grad for model in (algorithm.actor, algorithm.critic) for p in model.parameters())


def test_legacy_v3_remains_the_chassis_recipe(tmp_path):
    plan = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v3.json").read_text())
    cfg, settings, bank = configuration(tmp_path, plan, num_envs=80)
    assert cfg["critic_dim"] == 83 and "full_vehicle" not in cfg
    assert "纯底盘" in cfg["asset_directory"] and "standing_odometry" not in settings
    assert "height_matched073" in bank.name

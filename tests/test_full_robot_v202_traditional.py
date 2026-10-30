"""Canonical frames, supplied limits and shared-control pose generation."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from pxr import Usd, UsdPhysics

from wheeled_tasks.chassis.full_robot_asset import closure_residual, fk, origin, solve_pose
from wheeled_tasks.chassis.traditional_asset import TraditionalAsset
from wheeled_tasks.chassis.traditional_model import equilibrium_pose, linear_seed, virtual_work

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"model/wheel_leg_v2.0.2/isaac"


@pytest.fixture(scope="module")
def asset():
    return TraditionalAsset(BUNDLE)


def test_all_supplied_asset_dependencies_and_new_topology_authenticate(asset):
    assert asset.manifest["control_frame"] == "Xforward_Yleft_Zup"
    assert len(asset.spec["bodies"]) == 21
    assert sum(b["mass"] for b in asset.spec["bodies"]) == pytest.approx(21.44174)
    assert len(asset.manifest["tree_joint_names"]) == 18
    assert len(asset.spec["constraints"]) == 6
    assert sum(j["type"] == "fixed" for j in asset.spec["joints"]) == 2


def test_canonical_base_and_cad_inertias_are_used_without_another_rotation(asset):
    source = BUNDLE.parent/"轮腿整车URDFv2.0.2/urdf/轮腿整车URDFv2.0.2.urdf"
    robot = ET.parse(source).getroot()
    for body in asset.spec["bodies"]:
        inertial = robot.find(f"link[@name='{body['name']}']/inertial")
        frame = origin(inertial.find("origin"))
        a = {k:float(v) for k,v in inertial.find("inertia").attrib.items()}
        inertia = np.array([[a["ixx"],a["ixy"],a["ixz"]],
            [a["ixy"],a["iyy"],a["iyz"]],[a["ixz"],a["iyz"],a["izz"]]])
        assert body["mass"] == float(inertial.find("mass").get("value"))
        np.testing.assert_allclose(body["com"],frame[:3,3],atol=1e-12,rtol=0.)
        np.testing.assert_allclose(body["inertia"],frame[:3,:3]@inertia@frame[:3,:3].T,atol=1e-12,rtol=0.)
    for name in ("L_joint1", "R_joint1"):
        raw = origin(robot.find(f"joint[@name='{name}']/origin"))
        actual = next(j for j in asset.spec["joints"] if j["name"] == name)
        np.testing.assert_allclose(actual["origin"],raw,atol=1e-12,rtol=0.)
    frames = fk(asset.spec,{})
    np.testing.assert_array_equal(frames["base_link"],np.eye(4))
    assert frames["L_link1"][1,3] > .18
    assert frames["R_link1"][1,3] < -.18


def test_supplied_usd_loops_and_mechanical_limits_are_retained(asset):
    stage = Usd.Stage.Open(str(BUNDLE/"robot.usda"))
    loops = [UsdPhysics.Joint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.SphericalJoint)]
    assert len(loops) == 6 and all(j.GetExcludeFromArticulationAttr().Get() for j in loops)
    for joint in asset.spec["joints"]:
        path = "/Robot/tree_joints/"+joint["name"]
        if joint["type"] == "fixed":
            assert stage.GetPrimAtPath(path).IsA(UsdPhysics.FixedJoint)
        elif joint["type"] in ("prismatic","revolute"):
            cls = UsdPhysics.PrismaticJoint if joint["type"] == "prismatic" else UsdPhysics.RevoluteJoint
            factor = 1. if joint["type"] == "prismatic" else 180./np.pi
            usd = cls(stage.GetPrimAtPath(path))
            assert usd.GetLowerLimitAttr().Get() == pytest.approx(factor*float(joint["limit"]["lower"]),abs=1e-5)
            assert usd.GetUpperLimitAttr().Get() == pytest.approx(factor*float(joint["limit"]["upper"]),abs=1e-5)


@pytest.mark.parametrize("length",[.24,.28,.30,.334,.35])
def test_regenerated_equilibrium_virtual_work_and_gains(asset,length):
    q = equilibrium_pose(asset.spec,length)
    assert np.max(abs(closure_residual(asset.spec,q))) < 1e-8
    frames = fk(asset.spec,q)
    for side in ("L","R"):
        value = np.linalg.norm((frames[side+"_link3"][:3,3]-frames[side+"_link1"][:3,3])[[0,2]])
        assert value == pytest.approx(length,abs=2e-6)
    torque,spring,error = virtual_work(asset.spec,q)
    assert np.isfinite(torque).all() and np.isfinite(spring).all() and error < 2e-7
    assert linear_seed(asset.spec,q)["discrete_spectral_radius"] < 1.


def test_pose_solver_rejects_the_new_mechanical_limits(asset):
    q = dict(asset.spec["nominal_joint_pos"])
    q["L_joint2"] = 1.2
    with pytest.raises(ValueError,match="limit"):
        solve_pose(asset.spec,q,q)


@pytest.mark.parametrize("beta", [35., 40., 75., 101., 105.])
def test_measured_thigh_shank_inner_angle_and_full_branch(asset, beta):
    zeros = asset.spec["passive_knee_zero_inner_rad"]
    q = solve_pose(asset.spec, dict(L_joint1=.42, R_joint1=-.42, L_joint3=0., R_joint3=0.,
        L_joint2=np.deg2rad(beta)-zeros["L_joint2"],
        R_jonit2=-(np.deg2rad(beta)-zeros["R_jonit2"])), asset.spec["nominal_joint_pos"])
    frames = fk(asset.spec, q)
    for side in ("L", "R"):
        knee = frames[side+"_link2"][:3,3]
        thigh = frames[side+"_link1"][:3,3]-knee
        shank = frames[side+"_link3"][:3,3]-knee
        # Use the projected mechanical angle, not encoder degrees.
        name = "L_joint2" if side == "L" else "R_jonit2"
        joint = next(j for j in asset.spec["joints"] if j["name"] == name)
        axis = frames[joint["child"]][:3,:3]@joint["axis"]
        thigh -= axis*(axis@thigh)
        shank -= axis*(axis@shank)
        actual = np.rad2deg(np.arccos(np.clip(thigh@shank/np.linalg.norm(thigh)/np.linalg.norm(shank), -1., 1.)))
        assert actual == pytest.approx(beta, abs=2e-6)
    assert np.max(abs(closure_residual(asset.spec, q))) < 1e-8
    assert asset.limits["knee_inner_limits_deg"] == [35., 105.]
    assert asset.curve["geometry_beta_domain_deg"] == [35., 105.]
    assert asset.curve["measured_beta_domain_deg"] == [56., 101.]
    assert asset.curve["right_leg_symmetry_basis"] == "user accepted 2026-10-30"


@pytest.mark.parametrize("beta", [34., 106.])
def test_measured_inner_angle_stops_reject_outside_pose(asset, beta):
    zero = asset.spec["passive_knee_zero_inner_rad"]["L_joint2"]
    with pytest.raises(ValueError, match="limit"):
        solve_pose(asset.spec, dict(L_joint2=np.deg2rad(beta)-zero))

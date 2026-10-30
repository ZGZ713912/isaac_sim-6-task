"""Full-vehicle frame, topology, virtual-work and measured-response contracts."""
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
from pxr import Usd, UsdGeom, UsdPhysics

from wheeled_algo.encoder_leg_geometry import EncoderLegGeometry
from wheeled_tasks.chassis.full_robot_v202 import FullRobotV202, digest

ROOT = Path(__file__).resolve().parents[1]
definition = importlib.util.spec_from_file_location("v202_builder_tests",ROOT / "tools/full_robot_v202.py")
geometry = importlib.util.module_from_spec(definition)
definition.loader.exec_module(geometry)
BUNDLE = ROOT / "model/wheel_leg_v2.0.2/isaac"


@pytest.fixture(scope="module")
def asset():
    return FullRobotV202()


def test_reviewed_inputs_are_unchanged_and_all_bundle_dependencies_authenticate(asset):
    provenance = json.loads((BUNDLE.parent / "source_provenance.json").read_text())
    for name, expected in provenance["source_files_sha256"].items():
        assert digest(geometry.SOURCE / name) == expected
    assert digest(geometry.SOURCE / "urdf/轮腿整车URDFv2.0.1.urdf") == geometry.SOURCE_SHA
    assert digest(geometry.REFERENCE / "model_spec.json") == geometry.REFERENCE_SPEC_SHA
    assert digest(ROOT / "tools/full_robot_v202.py") == provenance["generator_sha256"]
    assert asset.manifest["control_joint_names"] == geometry.ACTIVE
    assert asset.manifest["total_mass_kg"] == pytest.approx(21.44174,abs=1e-12)
    assert len(asset.manifest["rigid_body_names"]) == 21


def test_coordinate_reexpression_preserves_every_visual_com_inertia_and_screw():
    raw = ET.parse(geometry.SOURCE / "urdf/轮腿整车URDFv2.0.1.urdf").getroot()
    normalized = ET.parse(BUNDLE.parent / "轮腿整车URDFv2.0.2/urdf/轮腿整车URDFv2.0.2.urdf").getroot()
    result = geometry.verify_reframing(raw,normalized)
    assert result["passed"] and result["encoder_zero_changes"] == 0
    assert result["active_joint_signs"] == dict.fromkeys(geometry.ACTIVE,1)
    frames = geometry.fk(dict(joints=geometry.parse_joints(normalized)),{})
    for name in ("base_link","M_link1","M_link2"):
        np.testing.assert_allclose(frames[name][:3,:3],np.eye(3),atol=1e-11,rtol=0.)


def test_native_usd_topology_gimbal_variants_and_nominal_loop_anchors(asset):
    fixed = Usd.Stage.Open(str(BUNDLE / "robot.usda"))
    free = Usd.Stage.Open(str(BUNDLE / "robot_gimbal_free.usda"))
    assert len([p for p in fixed.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]) == 21
    assert len([p for p in fixed.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)]) == 2
    assert len([p for p in fixed.Traverse() if p.IsA(UsdPhysics.SphericalJoint)]) == 6
    for name in geometry.GIMBAL:
        path = "/Robot/tree_joints/"+name
        assert fixed.GetPrimAtPath(path).IsA(UsdPhysics.FixedJoint)
        assert free.GetPrimAtPath(path).IsA(UsdPhysics.RevoluteJoint)
    cache = UsdGeom.XformCache()
    for c in asset.spec["constraints"]:
        p = fixed.GetPrimAtPath("/Robot/loop_joints/"+c["name"])
        j = UsdPhysics.SphericalJoint(p)
        assert j.GetExcludeFromArticulationAttr().Get()
        points = []
        for side in (0,1):
            body = fixed.GetPrimAtPath("/Robot/"+c[f"body{side}"])
            world = cache.GetLocalToWorldTransform(body)
            points.append(np.array(world.Transform(tuple(c[f"local_pos{side}_m"]))))
        np.testing.assert_allclose(*points,atol=1e-7,rtol=0.)
    counts = [len(b["collisions"]) for b in asset.spec["bodies"]]
    assert sum(counts) <= 48 and max(counts) <= 8


@pytest.mark.parametrize("beta",[42.,75.,106.])
def test_closed_chain_has_twelve_independent_constraints_and_six_motor_dofs(asset,beta):
    spec = asset.spec
    q = geometry.solve_pose(spec,dict(L_joint1=.42,R_joint1=-.42,L_joint3=0.,R_joint3=0.,
        L_joint2=np.deg2rad(beta)-spec["passive_knee_zero_inner_rad"]["L_joint2"],
        R_jonit2=-(np.deg2rad(beta)-spec["passive_knee_zero_inner_rad"]["R_jonit2"])),spec["nominal_joint_pos"])
    names = asset.manifest["tree_joint_names"]
    jac = []
    for name in names:
        plus,minus = dict(q),dict(q)
        plus[name]+=1e-6;minus[name]-=1e-6
        jac.append((geometry.closure_error(spec,plus)-geometry.closure_error(spec,minus))/2e-6)
    assert np.linalg.matrix_rank(np.column_stack(jac),tol=1e-7)==12
    assert np.max(np.abs(geometry.closure_error(spec,q))) < 1e-8
    for name in asset.manifest["spring_joint_names"]:
        binding = spec["spring_binding"][name]
        assert 0. < binding["compression_at_q_zero_m"]-q[name] < binding["stroke_m"]


def test_spring_force_knots_are_inherited_but_virtual_work_uses_new_geometry(asset):
    old = json.loads((ROOT / "model/gas_spring_compensation_20261028/model.json").read_text())
    new = json.loads((BUNDLE / "gas_spring_model.json").read_text())
    for key in ("compression_knots_m","force_knots_n","force_extension","source_curve_sha256"):
        assert new[key] == old[key]
    control,_ = asset.components(1)
    q = asset.spec["nominal_joint_pos"]
    active = {n:q[n] for n in geometry.ACTIVE}
    angles = torch.tensor([[q[n] for n in ("L_joint1","LL_joint1","R_joint1","RR_joint1")]])
    tau,diagnostics = control.gas_compensation.evaluate(angles)
    assert torch.isfinite(tau).all() and torch.isfinite(diagnostics["force_n"]).all()
    s = torch.tensor([[q[n] for n in asset.manifest["spring_joint_names"]]])
    forces = control.spring_efforts(s)[0].numpy()
    for index,name in enumerate(("L_joint1","LL_joint1","R_joint1","RR_joint1")):
        plus,minus = dict(active),dict(active)
        plus[name]+=1e-5;minus[name]-=1e-5
        p = geometry.solve_pose(asset.spec,plus,q)
        m = geometry.solve_pose(asset.spec,minus,q)
        derivative = np.array([(p[n]-m[n])/2e-5 for n in asset.manifest["spring_joint_names"]])
        assert tau[0,index].item() == pytest.approx(-float(forces@derivative),abs=.002)


def test_measured_wheel_total_inertia_and_fractional_delay_are_used_once(asset):
    source = json.loads((ROOT / "model/wheel_identified_1562_20261028/model.json").read_text())
    _,wheels = asset.components(1)
    for name in ("L_joint3","R_joint3"):
        old,new = source["axes"][name],wheels.model["axes"][name]
        assert new["effective_total_inertia_kg_m2"] == old["effective_total_inertia_kg_m2"]
        assert new["cad_inertia_kg_m2"] == pytest.approx(.00058325,abs=1e-14)
        assert new["cad_inertia_kg_m2"]+new["additional_armature_kg_m2"] == pytest.approx(new["effective_total_inertia_kg_m2"],abs=1e-14)
    command = torch.zeros(1,6);command[:,(2,5)]=1.
    response = torch.stack([wheels.apply_torque(command,torch.zeros_like(command))[:,(2,5)] for _ in range(9)])
    torch.testing.assert_close(response[:5,0,0],torch.zeros(5))
    assert response[5,0,0] == .5 and response[6,0,0] == 1.
    torch.testing.assert_close(response[:3,0,1],torch.zeros(3))
    assert response[3,0,1] == .5 and response[4,0,1] == 1.


def test_encoder_observer_requires_explicit_identity_and_matches_full_linkage_fk(asset):
    table = json.loads((BUNDLE / "encoder_geometry.json").read_text())
    with pytest.raises(ValueError,match="authenticated107"):
        EncoderLegGeometry(table)
    with pytest.raises(ValueError,match="authenticated"):
        EncoderLegGeometry(table,expected_spec_sha256="0"*64)
    observer = asset.encoder_geometry()
    spec = asset.spec
    for beta,left,right in ((50.,-.4,.2),(75.,.35,-.25),(100.,.1,-.5)):
        q = geometry.solve_pose(spec,dict(L_joint1=left,R_joint1=right,L_joint3=0.,R_joint3=0.,
            L_joint2=np.deg2rad(beta)-spec["passive_knee_zero_inner_rad"]["L_joint2"],
            R_jonit2=-(np.deg2rad(beta)-spec["passive_knee_zero_inner_rad"]["R_jonit2"])),spec["nominal_joint_pos"])
        positions = torch.tensor([[q[n] for n in ("L_joint1","LL_joint1","R_joint1","RR_joint1")]])
        output = observer(positions,torch.zeros_like(positions))
        assert output["valid"].all()
        frames = geometry.fk(spec,q)
        expected = [geometry.point(frames[b["name"]],np.array(b["collisions"][0]["origin"])[:3,3])
                    for name in ("L_link3","R_link3") for b in spec["bodies"] if b["name"]==name]
        np.testing.assert_allclose(output["wheel_body_m"][0],expected,atol=2e-5,rtol=0.)
        torch.testing.assert_close(output["knee_deg"],torch.full((1,2),beta),atol=.002,rtol=0.)

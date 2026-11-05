"""Native CAD correspondence and asymmetric reset closure, without launching Kit."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"),str(ROOT / "tools")]
import full_robot_v202 as geometry
from wheeled_tasks.wheelbipe_port.model import (
    BODY_ALIASES, BUNDLE, JOINT_ALIASES, CollisionSupport, PairedTargets, ResetLookup, authenticate_source,
)


def test_original_source_and_native_name_bijection():
    assert len(authenticate_source()["files_sha256"]) == 187
    spec = json.loads((BUNDLE / "model_spec.json").read_text())
    names = [j["name"] for j in spec["joints"] if j["type"] != "fixed"]
    assert set(names) == set(JOINT_ALIASES)
    assert len(set(JOINT_ALIASES.values())) == 18
    assert {b["name"] for b in spec["bodies"]} == set(BODY_ALIASES)
    assert len(set(BODY_ALIASES.values())) == 21


def test_relocated_vendor_documents_keep_their_original_hashes_and_directory_boundary(tmp_path, monkeypatch):
    import hashlib
    import pytest
    from wheeled_tasks.wheelbipe_port import model

    vendor = tmp_path/"third_party/wheelbipe"
    vendor.mkdir(parents=True)
    target = tmp_path/"docs/third_party/wheelbipe/README.md"
    target.parent.mkdir(parents=True)
    target.write_text("original vendor documentation\n")
    manifest = dict(source_commit=model.SOURCE_COMMIT,
        files_sha256={"src/README.md": hashlib.sha256(target.read_bytes()).hexdigest()},
        documentation_locations={"src/README.md": str(target.relative_to(tmp_path))})
    metadata = vendor/"provenance.json"
    metadata.write_text(json.dumps(manifest))
    monkeypatch.setattr(model, "ROOT", tmp_path)
    monkeypatch.setattr(model, "VENDOR", vendor)
    assert model.authenticate_source() == manifest
    target.write_text("modified documentation\n")
    with pytest.raises(ValueError, match="source changed"):
        model.authenticate_source()
    manifest["documentation_locations"]["src/README.md"] = "outside/README.md"
    metadata.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source changed"):
        model.authenticate_source()


def test_independent_left_right_reset_closes_all_six_constraints():
    spec = json.loads((BUNDLE / "model_spec.json").read_text())
    names = [j["name"] for j in spec["joints"] if j["type"] != "fixed"]
    # Reverse order also checks that the policy semantic order is not assumed
    # to be PhysX's actual articulation order.
    names.reverse()
    lookup = ResetLookup(names,"cpu")
    generator = torch.Generator().manual_seed(107)
    length = .16+.20*torch.rand(40,2,generator=generator)
    angle = -2.35+4.70*torch.rand(40,2,generator=generator)
    positions = lookup.sample(length,angle)
    for row in positions.double().numpy():
        q = dict(zip(names,row))
        gap = np.linalg.norm(geometry.closure_error(spec,q).reshape(-1,3),axis=-1)
        assert gap.shape == (6,)
        assert gap.max() < 1.e-5
        for name,sign in (("L_joint2",1.),("R_jonit2",-1.)):
            beta = sign*q[name]+spec["passive_knee_zero_inner_rad"][name]
            assert np.radians(40.)-1.e-6 <= beta <= np.radians(110.)+1.e-6


def test_collision_clearance_matches_native_fk_for_asymmetric_poses():
    from scipy.spatial.transform import Rotation
    from itertools import product
    spec = json.loads((BUNDLE/"model_spec.json").read_text())
    names = [j["name"] for j in spec["joints"] if j["type"]!="fixed"]
    generator = torch.Generator().manual_seed(108)
    q = ResetLookup(names,"cpu").sample(.16+.20*torch.rand(40,2,generator=generator),
        -np.pi/2.+1.25*np.pi*torch.rand(40,2,generator=generator))
    euler = torch.rand(40,3,generator=generator).numpy()*[.3,.3,6.28]-[.15,.15,3.14]
    rotations = Rotation.from_euler("xyz",euler)
    xyzw = rotations.as_quat()
    quaternion = torch.tensor(xyzw[:,[3,0,1,2]],dtype=torch.float32)
    support = CollisionSupport(names,"cpu")
    actual = support.minimum_z(q,quaternion).numpy()
    reference = []
    for row,rotation in zip(q.numpy(),rotations.as_matrix()):
        root = np.eye(4); root[:3,:3] = rotation
        frames = geometry.fk(spec,dict(zip(names,row)),root)
        minimum = np.inf
        for name,(lo,hi) in support.bounds.items():
            corners = np.array(list(product(*zip(lo.numpy(),hi.numpy()))))
            points = corners@frames[name][:3,:3].T+frames[name][:3,3]
            minimum = min(minimum,points[:,2].min())
        reference.append(minimum)
    np.testing.assert_allclose(actual,reference,atol=2.e-7,rtol=1.e-6)
    # The inherited .32m base reset does penetrate some valid long-leg poses.
    penetration = -(actual+.32)
    assert penetration.max()>.02
    lifted = .32+np.maximum(penetration+.002,0.)
    assert np.min(lifted+actual)>=.002-1.e-7


def test_targets_use_actual_110_degree_geometry_and_shared_winding():
    model = json.loads((BUNDLE/"gas_spring_model.json").read_text())
    projector = PairedTargets(model,"cpu")
    generator = torch.Generator().manual_seed(109)
    desired = 40.*torch.randn(100,4,generator=generator)
    current = 40.*torch.randn(100,4,generator=generator)
    result = projector(desired,current).double().unflatten(-1,(2,2))
    actual_delta = projector.sign*(result[...,1]-result[...,0])
    assert torch.all(actual_delta>=projector.bounds[:,0]-1.e-5)
    assert torch.all(actual_delta<=projector.bounds[:,1]+1.e-5)
    assert torch.all((result.mean(-1)-current.double().unflatten(-1,(2,2)).mean(-1)).abs()<=np.pi+1.e-5)
    # A target on the real 110-degree edge must not be silently clipped to
    # the old paired_branch107_v1 constant (approximately 1.369 rad).
    edge = torch.tensor([[0.,projector.bounds[0,1],projector.bounds[1,1],0.]],dtype=torch.float64)
    torch.testing.assert_close(projector(edge,edge),edge,rtol=0.,atol=1.e-12)
    assert projector.bounds[:,1].min()>1.42


def test_unreachable_target_cost_removes_collapsed_leg_reward_loophole():
    model = json.loads((BUNDLE/"gas_spring_model.json").read_text())
    projector = PairedTargets(model,"cpu")
    feasible = torch.tensor([[-.42,.1372233087,.42,-.1372160582]],dtype=torch.float64)
    torch.testing.assert_close(projector.projection_cost(feasible),torch.zeros(1,dtype=torch.float64))
    # Captured from checkpoint10500, height command .305 m: the actor sends
    # large requests which the CAD branch projects to the fully short leg.
    captured = torch.tensor([[1.01177335,-1.25849819,-1.42845762,1.77020109]],dtype=torch.float64)
    # This fixture is already in ACTIVE pair order (L hip/aux, R hip/aux).
    actual = projector(captured,feasible)
    expected = (captured-actual).square().sum(-1)
    torch.testing.assert_close(projector.projection_cost(captured),expected)
    assert projector.projection_cost(captured).item()>5.
    shifted = captured+torch.tensor([[4.,4.,-6.,-6.]],dtype=torch.float64)*torch.pi
    torch.testing.assert_close(projector.projection_cost(shifted),expected)
    # Moving farther outside either mechanical edge must increase the cost,
    # even though the projected target stays at exactly the same stop.
    beyond = feasible.clone()
    beyond[:,1] = beyond[:,0]+projector.bounds[0,1]+.1
    far = beyond.clone(); far[:,1] += .1
    assert projector.projection_cost(far)>projector.projection_cost(beyond)>0.

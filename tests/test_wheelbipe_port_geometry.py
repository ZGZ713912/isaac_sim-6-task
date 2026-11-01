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

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
    BODY_ALIASES, BUNDLE, JOINT_ALIASES, ResetLookup, authenticate_source,
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

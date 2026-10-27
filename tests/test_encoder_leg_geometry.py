"""Check encoder-only positions against independent full-link FK and rates."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

from wheeled_algo.encoder_leg_geometry import EncoderLegGeometry
from wheeled_tasks.chassis.standing_geometry107 import table_path

ROOT = Path(__file__).resolve().parents[1]


def model():
    return EncoderLegGeometry(json.loads(table_path().read_text()))


def test_prone_and_standing_encoders_match_independent_complete_linkage_fk():
    bundle = ROOT / "model/纯底盘_v7_flat_base107/urdf"
    module_spec = importlib.util.spec_from_file_location("independent_prone_fk", bundle / "tools/v6_mechanism.py")
    fk_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(fk_module)
    mechanism = json.loads((bundle / "model_spec.json").read_text())
    bodies = {b["name"]: b for b in mechanism["bodies"]}
    poses = json.loads((ROOT / "reports/robust_policy_repair_20261024/pose_bank_height_matched073.json").read_text())["poses"]
    selected = [poses[i] for i in (0, 29, 203, 232, 301, 316, 429, 430, 431)]
    names = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")
    q = torch.tensor([[p["full_joint_positions"][n] for n in names] for p in selected])
    state = model()(q, torch.zeros_like(q))
    expected = []
    for p in selected:
        frames = fk_module.fk(mechanism, p["full_joint_positions"])
        expected.append([(frames[s+"_link3"] @ np.asarray(bodies[s+"_link3"]["collisions"][0]["origin"]))[:3, 3]
                         for s in ("L", "R")])
    assert state["valid"].all()
    assert np.max(np.linalg.norm(state["wheel_body_m"].numpy()-np.asarray(expected), axis=-1)) < 2e-5
    torch.testing.assert_close(state["knee_deg"], torch.tensor([p["knee_deg"] for p in selected]), atol=.002, rtol=0.)


def test_encoder_rates_match_finite_motion_and_distinguish_folding_from_extension():
    observer = model()
    q = observer.nominal[None].repeat(2, 1)
    velocity = torch.tensor([[.2, .5, -.2, -.5], [-.2, -.5, .2, .5]])
    state = observer(q, velocity)
    epsilon = .001
    plus, minus = observer(q+epsilon*velocity, velocity), observer(q-epsilon*velocity, velocity)
    for position_key, rate_key in (("wheel_body_m", "wheel_body_velocity_m_s"),
                                   ("knee_deg", "knee_rate_deg_s"), ("leg_length_m", "leg_length_rate_m_s")):
        finite_rate = (plus[position_key]-minus[position_key])/(2.*epsilon)
        torch.testing.assert_close(state[rate_key], finite_rate, atol=.003, rtol=.003)
    assert (state["knee_rate_deg_s"][0] > 0.).all()
    assert (state["knee_rate_deg_s"][1] < 0.).all()
    shifted = observer(q+4.*math.pi, velocity)
    torch.testing.assert_close(shifted["wheel_body_m"], state["wheel_body_m"], atol=2e-6, rtol=0.)


def test_outside_closed_chain_branch_is_marked_invalid():
    observer = model()
    q = observer.nominal[None].clone()
    q[0, 1] = q[0, 0]+3.
    assert observer(q, torch.zeros_like(q))["valid"].tolist() == [[False, True]]


def test_small_native_endpoint_residual_is_tolerated_but_real_out_of_branch_input_is_invalid():
    observer = model()
    q = observer.nominal[None].repeat(2, 1)
    q[:, 1] = q[:, 0]+observer.side0_delta_rad[-1]+torch.tensor([.0005, .002])
    state = observer(q, torch.zeros_like(q))
    assert state["valid"][:, 0].tolist() == [True, False]
    torch.testing.assert_close(state["knee_deg"][:, 0], observer.side0_passive_inner_deg[-1].expand(2))

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis import side_fall_study as task
from wheeled_tasks.chassis.leg_motor_friction import LegMotorFriction, declaration

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"model/wheel_leg_v2.0.2/training"


def test_asymmetric_side_grid_preserves_actual_knees_orbits_and_closed_chain():
    from build_full_vehicle_training import geometry
    spec = json.loads((BUNDLE/"model_spec.json").read_text())
    starts = task.build_side_starts(spec,BUNDLE,rolls=(90.,),orbits=(-90.,90.),knee_pairs=((45.,105.),))
    assert len(starts) == 8
    zero = spec["passive_knee_zero_inner_rad"]
    for start in starts:
        q = start["full_joint_positions"]
        assert np.rad2deg([q["L_joint2"]+zero["L_joint2"],zero["R_jonit2"]-q["R_jonit2"]]) == pytest.approx([45.,105.],abs=.001)
        assert task.orbital_angles(spec,q) == pytest.approx(start["orbital_pair_deg"],abs=1e-5)
        assert np.linalg.norm(geometry.closure_error(spec,q)) < 1e-7
        assert min(start["initial_body_clearance_m"].values()) == pytest.approx(.002,abs=1e-10)
    assert starts[0]["full_joint_positions"] == starts[4]["full_joint_positions"]
    assert starts[0]["spawn_family"] == "left"
    assert starts[4]["spawn_family"] == "right"


def fixture_payload():
    s = dict(pair_id=1,name="x",root_position_m=[0.,0.,.2],quaternion_xyzw=[0.,0.,0.,1.],full_joint_positions={"q":.2})
    s["placement_sha256"] = task.placement_fingerprint(s)
    return dict(schema=task.SCHEMA,model_spec_sha256="s",asset_manifest_sha256="a",starts=task.paired([s]))


@pytest.mark.parametrize("kind",["state","same_arm","missing_arm","asset","hash"])
def test_pair_validation_rejects_unequal_initial_states_or_missing_controls(kind):
    payload = fixture_payload()
    if kind == "state":
        payload["starts"][1]["full_joint_positions"]["q"] += .1
    elif kind == "same_arm":
        payload["starts"][1]["leg_friction_enabled"] = False
    elif kind == "missing_arm":
        payload["starts"].pop()
    elif kind == "asset":
        payload["model_spec_sha256"] = "other"
    else:
        payload["starts"][1]["placement_sha256"] = "other"
    with pytest.raises(ValueError):
        task.validate_placements(payload,"s","a")


def test_pairs_have_identical_root_leg_gimbal_state_and_explicit_friction_arm():
    payload = fixture_payload()
    rows = task.validate_placements(payload,"s","a")
    assert [r["leg_friction_enabled"] for r in rows] == [False,True]
    assert rows[0]["placement_sha256"] == rows[1]["placement_sha256"]


def test_imported_loss_dissipates_for_both_velocity_signs_and_has_no_static_hold():
    loss = LegMotorFriction(3,"cpu",declaration(ROOT),root=ROOT)
    velocity = torch.tensor([[0.,0.,0.,0.],[.2,-.2,1.,-1.],[-.2,.2,-1.,1.]])
    torque = loss.effort(velocity)
    assert (torque*velocity <= 0.).all()
    assert torch.equal(torque[0],torch.zeros(4))
    assert torch.allclose(torque[1].abs(),loss.coulomb)
    assert loss.summary()["physical_static_model_enabled"] is False


def test_support_diagnosis_excludes_shell_gimbal_and_transient_impacts():
    from summarize_side_fall_study import trace_metrics
    names = ["base_link","M_link1","M_link2","L_link1","L_link3","R_link1","R_link3"]
    columns = ["contact_z_n/"+n for n in names]+["dq/"+n for n in ("L_joint1","LL_joint1","R_joint1","RR_joint1")]
    trace = np.zeros((200,len(columns)))
    trace[:,0] = 200.
    trace[:,1] = 10.
    assert not trace_metrics(trace,columns,names,210.,"left")["sustained_leg_support"]
    trace[0,3] = 200.
    assert not trace_metrics(trace,columns,names,210.,"left")["sustained_leg_support"]
    trace[:,3] = 12.
    m = trace_metrics(trace,columns,names,210.,"left")
    assert m["sustained_leg_support"]
    assert m["wheel_support"]["mean_weight_fraction"] == 0.
    assert m["leg_structure_support"]["mean_weight_fraction"] == pytest.approx(12./210.)

"""Feasible independent leg cells, held-out sampling and uninterrupted handoff."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.multi_pose_recovery import (FAMILIES,LEG_CLASSES,ResetSampler,RecoveryStages,
    leg_class,validate_bank,recovery_terms)
from wheeled_tasks.chassis.prone_motion_cli import resolve_plan, train_parser, training_options

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v5.json").read_text())
BANK = json.loads((ROOT/PLAN["pose_bank"]["file"]).read_text())


def test_complete_body_by_independent_leg_cells_are_admitted_with_separate_holdout():
    cells = validate_bank(BANK,PLAN["multi_pose_recovery"],BANK["asset_manifest_sha256"],BANK["model_spec_sha256"])
    assert len(cells) == 28
    for pool in cells.values():
        assert {BANK["poses"][i]["recovery"]["split"] for i in pool} == {"train","holdout"}
    assert sum(p.get("recovery",{}).get("dynamic",False) for p in BANK["poses"]) == 24


def test_missing_cell_or_unreviewed_candidate_cannot_silently_use_standard_pose():
    b = deepcopy(BANK)
    b["poses"] = [p for p in b["poses"] if p.get("recovery",{}).get("leg_class") != "front_rear"]
    with pytest.raises(ValueError): validate_bank(b,PLAN["multi_pose_recovery"],b["asset_manifest_sha256"],b["model_spec_sha256"])
    b = deepcopy(BANK); next(p for p in b["poses"] if "recovery" in p)["admitted_as_training_reset"] = False
    with pytest.raises(ValueError): validate_bank(b,PLAN["multi_pose_recovery"],b["asset_manifest_sha256"],b["model_spec_sha256"])


def test_sampler_never_trains_on_holdout_and_retains_every_class_while_curriculum_expands():
    first,last = ResetSampler(BANK,PLAN["multi_pose_recovery"],31),ResetSampler(BANK,PLAN["multi_pose_recovery"],31)
    sampled = [[BANK["poses"][s.sample(u)]["recovery"] for _ in range(4000)] for s,u in ((first,0),(last,9000))]
    for rows in sampled:
        assert {r["split"] for r in rows} == {"train"}
        assert {(r["family"],r["leg_class"]) for r in rows} == {(f,l) for f in FAMILIES for l in LEG_CLASSES}
    assert sum(r["family"] == "belly_down" for r in sampled[0]) > 2*sum(r["family"] == "belly_down" for r in sampled[1])


def test_all_compiled_positions_and_dynamic_velocities_respect_the_actual_closed_chain():
    definition = importlib.util.spec_from_file_location("recovery_geometry_test",ROOT/"tools/full_robot_v202.py")
    geometry = importlib.util.module_from_spec(definition); definition.loader.exec_module(geometry)
    spec = json.loads((ROOT/"model/wheel_leg_v2.0.2/friction_v1/training/model_spec.json").read_text())
    for p in BANK["poses"]:
        q = p["full_joint_positions"]
        assert np.abs(geometry.closure_error(spec,q)).max() < 1e-7
        if "full_joint_velocities" in p:
            dq = p["full_joint_velocities"]
            future = {n:q[n]+dq[n]*1e-6 for n in q}
            assert np.linalg.norm((geometry.closure_error(spec,future)-geometry.closure_error(spec,q))/1e-6) < 1e-4


def test_entry_records_four_actual_leg_classes_without_resetting_position_velocity_or_history():
    c = deepcopy(PLAN["multi_pose_recovery"]); c["upright_hold_seconds"] = .02
    stage = RecoveryStages(4,"cpu",c,.01)
    ids = torch.arange(4); stage.reset(ids,torch.ones(4,dtype=torch.long),ids)
    q,dq,omega = torch.randn(4,4),torch.ones(4,4)*.7,torch.ones(4,3)*.2
    original_q,original_dq = q.clone(),dq.clone()
    orbits = torch.tensor([[.7,.3],[-.5,-.8],[.9,-.6],[-.4,.8]])
    events=[]
    for tick in range(5):
        state = dict(enable=torch.full((4,),tick == 0),elapsed=torch.full((4,),.15+tick*.01),passive=torch.full((4,),tick == 0))
        e,seconds = stage.update(state,eligible=torch.ones(4,dtype=torch.bool),gravity=torch.tensor([[0.,0.,-1.]]).repeat(4,1),
            omega=omega,q=q,dq=dq,orbits=orbits,geometry_valid=torch.ones(4,2,dtype=torch.bool),
            encoder_error=torch.zeros(4),failed=torch.zeros(4,dtype=torch.bool))
        events.append(e)
    assert torch.stack(events).sum(0).tolist() == [1,1,1,1]
    assert stage.entry_leg_class.tolist() == [0,1,2,3]
    assert torch.equal(q,original_q) and torch.equal(dq,original_dq)
    assert torch.equal(stage.entry_dq,dq) and torch.allclose(seconds,torch.full((4,),.04))


def test_time_primary_reward_does_not_reward_idle_pose_or_impose_half_turn_limit():
    c = PLAN["multi_pose_recovery"]
    assert c["leg_speed_limit_rad_s"] is None
    state = dict(height=torch.zeros(2),torque=torch.zeros(2,6),action_delta=torch.zeros(2))
    terms = recovery_terms(state,c,torch.zeros(2))
    assert torch.equal(sum(terms.values()),-torch.ones(2))


def test_cli_exposes_curriculum_and_part_limits_and_keeps_nonrecovery_roles():
    args = train_parser().parse_args(["--recovery-curriculum","all","--recovery-impact-window-ms","30",
        "--recovery-force-limits-n",'{"chassis":500,"gimbal":100,"legs":null,"wheels":null}'])
    _,p,_ = training_options(args,ROOT)
    assert p["multi_pose_recovery"]["curriculum_mode"] == "all"
    assert p["recovery_impact"]["window_seconds"] == .03
    assert p["recovery_impact"]["force_limits_n"]["gimbal"] == 100.
    assert p["roles_per20"] == PLAN["roles_per20"] and p["roles_per20"].count("prone") == 6
    assert resolve_plan(PLAN,max_iterations=5000)["multi_pose_recovery"]["curriculum"][1]["start_update"] == 1000


def test_cold_matrix_keeps_all_independent_holdouts_and_failure_denominators():
    from types import SimpleNamespace
    from evaluate_prone_motion import evaluation_cases,recovery_summary
    env = SimpleNamespace(poses=BANK["poses"],prone_index=3,height_pose={.23:0,.305:1,.42:2})
    cases = evaluation_cases(env,PLAN)
    assert len(cases) == 78
    heldout = [r for r in cases if "recovery_family" in r]
    assert len(heldout) == 56 and len({r["pose_index"] for r in heldout}) == 56
    assert {(r["recovery_family"],r["recovery_leg_class"]) for r in heldout} == {(f,l) for f in FAMILIES for l in LEG_CLASSES}
    examples = [dict(heldout[0],rise_completed=True,terminated=False,active_rise_time_s=2.),
                dict(heldout[0],rise_completed=False,terminated=True,active_rise_time_s=None),
                dict(heldout[0],rise_completed=False,terminated=False,active_rise_time_s=None)]
    group = next(iter(recovery_summary(examples).values()))
    assert (group["cases"],group["successes"],group["failures"],group["unfinished"]) == (3,1,1,2)
    assert group["successful_only_completion_median_s"] == 2.


def test_v5_deadline_excludes_disabled_birth_but_quiet_window_starts_after_completion():
    from test_prone_motion_timing import trace_at_100hz
    from evaluate_prone_motion import review_arrays
    arrays,case = trace_at_100hz()
    arrays["completed_event"][:] = False
    arrays["completed_event"][2009,0] = True  # 20.10s elapsed, 19.95s enabled.
    arrays["enable_event"] = np.zeros_like(arrays["completed_event"])
    arrays["enable_event"][14,0] = True
    for key in ("commands","requested","velocity"):
        arrays[key][:,:,0] = 0.
        arrays[key][6000:6300,0,0] = 2.
    arrays["position"][:,0,0] = np.cumsum(arrays["velocity"][:,0,0])*.01
    row = review_arrays(arrays,[case],np.zeros((1,2)),PLAN)[0]
    assert row["active_rise_time_s"] == pytest.approx(19.95)
    assert row["post_event_start_sample"] == 2010 and row["engineering_screen_pass"]

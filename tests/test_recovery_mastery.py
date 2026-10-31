"""All four independent classes, cold-boundary refresh, and research isolation."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery,CLASSES
from wheeled_tasks.chassis.multi_pose_recovery import ResetSampler
from wheeled_tasks.chassis.prone_motion_cli import resolve_plan,train_parser

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4.json").read_text())
BANK=json.loads((ROOT/PLAN["pose_bank"]["file"]).read_text())
CONFIG=PLAN["multi_pose_recovery"]["mastery_gate"]


def trials():
    return [dict(family="belly_down",leg_class=c,split="holdout",pose_index=4*idx+j,heading_rad=0.,passed=True)
        for idx,c in enumerate(CLASSES) for j in range(4)]


def test_update_numbers_cannot_unlock_complex_poses_before_four_class_mastery():
    s=ResetSampler(BANK,PLAN["multi_pose_recovery"],41)
    rows=[BANK["poses"][s.sample(9999)]["recovery"] for _ in range(400)]
    assert {r["family"] for r in rows}=={"belly_down"}
    assert {r["leg_class"] for r in rows}==set(CLASSES)
    s.mastery.review(trials(),250)
    rows=[BANK["poses"][s.sample(500)]["recovery"] for _ in range(4000)]
    assert len({r["family"] for r in rows})==7
    assert {r["split"] for r in rows}=={"train"}


def test_pooled_success_and_missing_classes_cannot_hide_a_failing_leg_class():
    m=RecoveryMastery(CONFIG);rows=trials();rows[-1]["passed"]=False
    assert not m.review(rows,250)["complex_recovery_unlocked"]
    assert not m.review(trials()[:-4],500)["complex_recovery_unlocked"]
    assert m.review(trials(),750)["unlocked_at"]==750


def test_insufficient_duplicate_or_training_cases_never_admit_mastery():
    m=RecoveryMastery(CONFIG)
    assert not m.review(trials()[::2],250)["all_classes_passed"]
    rows=trials();rows[0]["split"]="train"
    with pytest.raises(ValueError):m.review(rows,500)
    rows=trials();rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError):m.review(rows,500)


def test_streak_does_not_advance_on_repeated_evaluation_and_gate_has_no_budget_extension():
    c=deepcopy(CONFIG);c["required_consecutive_passes"]=2
    m=RecoveryMastery(c)
    assert not m.due(0) and m.due(250)
    assert not m.review(trials(),250)["complex_recovery_unlocked"]
    with pytest.raises(ValueError):m.review(trials(),250)
    assert not m.due(250) and not m.due(251)
    assert m.review(trials(),500)["complex_recovery_unlocked"]
    assert not m.due(750)
    assert resolve_plan(PLAN)["max_updates"]==10000


def test_gate_thresholds_are_cli_configurable_and_all_mode_cannot_bypass_it():
    p=resolve_plan(PLAN,recovery_gate_success_rate=.9,recovery_gate_every=100)
    assert p["multi_pose_recovery"]["mastery_gate"]["minimum_success_rate"]==.9
    assert p["multi_pose_recovery"]["mastery_gate"]["evaluation_interval_updates"]==100
    with pytest.raises(ValueError):resolve_plan(PLAN,recovery_curriculum="all")
    with pytest.raises(SystemExit):train_parser().parse_args(["--checkpoint","validation_final.pt"])


def test_mastery_requires_independent_headings_and_a_complex_stage():
    from wheeled_tasks.chassis.recovery_mastery import validate
    from wheeled_tasks.chassis.multi_pose_recovery import validate as validate_recovery
    bad=deepcopy(CONFIG);bad["evaluation_headings_rad"]=[0.,0.]
    with pytest.raises(ValueError,match="Repeated headings"):validate(bad)
    bad=deepcopy(PLAN["multi_pose_recovery"]);bad["curriculum"]=bad["curriculum"][:1]
    with pytest.raises(ValueError):validate_recovery(bad)


def test_continuation_rejects_different_mechanics_and_checks_each_adam_moment(tmp_path):
    from validate_recovery_continuation import load_candidate,assert_restored
    legacy=json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4_standard_20261031.json").read_text())
    state=dict(iter=250,infos=dict(schema="prone-motion-checkpoint-v1",training_origin="scratch",resolved_plan=legacy),
        actor_state_dict={"x":torch.ones(1)},critic_state_dict={"x":torch.ones(1)},
        optimizer_state_dict=dict(state={0:dict(step=torch.tensor(3.),exp_avg=torch.ones(1))},param_groups=[]))
    path=tmp_path/"old.pt";torch.save(state,path)
    _,loaded,receipt=load_candidate(path,PLAN)
    assert receipt["parent_completed_updates"]==250
    bad=deepcopy(PLAN);bad["leg_motor_friction"]={}
    with pytest.raises(ValueError):load_candidate(path,bad)
    changed=deepcopy(loaded["optimizer_state_dict"]);changed["state"][0]["exp_avg"]+=.01
    with pytest.raises(AssertionError):assert_restored(changed,loaded["optimizer_state_dict"])


def test_friction_migration_only_allows_the_authenticated_parent(tmp_path):
    from validate_recovery_continuation import load_candidate
    receipt=json.loads((ROOT/"model/wheel_leg_v2.0.2/friction_v1/migration.json").read_text())
    parent=json.loads((ROOT/receipt["previous_plan"]["file"]).read_text())
    state=dict(iter=250,infos=dict(schema="prone-motion-checkpoint-v1",training_origin="scratch",resolved_plan=parent),
        actor_state_dict={"x":torch.ones(1)},critic_state_dict={"x":torch.ones(1)},
        optimizer_state_dict=dict(state={0:dict(step=torch.tensor(3.),exp_avg=torch.ones(1))},param_groups=[]))
    path=tmp_path/"parent.pt";torch.save(state,path)
    with pytest.raises(ValueError):load_candidate(path,PLAN)
    _,_,origin=load_candidate(path,PLAN,allow_friction_migration=True)
    assert origin["plant_migration"]["mastery_must_be_remeasured"]
    changed=deepcopy(PLAN);changed["wheel_model"]["sha256"]="0"*64
    with pytest.raises(ValueError):load_candidate(path,changed,allow_friction_migration=True)


def test_mastery_hook_refreshes_observation_before_native_act_without_replacing_learner(tmp_path,monkeypatch):
    import evaluate_recovery_mastery as evaluation
    from wheeled_algo.recovery_mastery import RecoveryMasteryHook
    m=RecoveryMastery(CONFIG)
    actor=torch.nn.Linear(1,1);env=SimpleNamespace(num_envs=1,device="cpu",reset=lambda ids:None,
        get_observations=lambda:torch.tensor([[2.]]))
    calls=[]
    alg=SimpleNamespace(actor=actor,critic=object(),optimizer=object(),storage=SimpleNamespace(step=0),
        act=lambda obs:calls.append(obs.clone()) or obs)
    runner=SimpleNamespace(alg=alg,env=env,device="cpu",logger=SimpleNamespace(writer=None))
    monkeypatch.setattr(evaluation,"evaluate",lambda *a,**kw:dict(cases=trials(),new_PPO=0))
    hook=RecoveryMasteryHook(runner,m,completed=lambda:250,output=tmp_path)
    assert torch.equal(alg.act(torch.tensor([[1.]])),torch.tensor([[2.]]))
    assert m.unlocked and len(calls)==1 and alg.storage.step==0
    hook.close()

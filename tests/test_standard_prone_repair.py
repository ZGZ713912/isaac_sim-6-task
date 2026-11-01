"""The fresh default scope, native physical stops and5mm guard stay coherent."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from train_prone_motion import configuration
from evaluate_recovery_mastery import cases_for
from wheeled_tasks.chassis.global_curriculum import GlobalCurriculum,SKILLS
from wheeled_tasks.chassis.multi_pose_recovery import ResetSampler,validate,validate_bank,validate_native_admission
from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery
from wheeled_tasks.chassis.prone_motion import validate as validate_plan

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4.json').read_text())
BANK=json.loads((ROOT/PLAN['pose_bank']['file']).read_text())


def course():return GlobalCurriculum(PLAN,RecoveryMastery(PLAN['multi_pose_recovery']['mastery_gate']))


def trials():
    return [dict(row,passed=True) for row in cases_for(SimpleNamespace(poses=BANK['poses'],num_envs=256),
        PLAN['multi_pose_recovery']['mastery_gate'])]


def test_only_the_original_standard_posture_is_ever_sampled():
    validate_plan(PLAN)
    validate_bank(BANK,PLAN['multi_pose_recovery'],BANK['asset_manifest_sha256'],BANK['model_spec_sha256'])
    old=json.loads((ROOT/'model/recovery_v202_20261031/pose_bank.json').read_text())
    assert len(BANK['poses'])==4 and sum('recovery' in p for p in BANK['poses'])==1
    for key in ['root_position_m','root_quaternion_xyzw','full_joint_positions']:
        assert BANK['poses'][3][key]==old['poses'][3][key]
    sampler=ResetSampler(BANK,PLAN['multi_pose_recovery'],107)
    for clock in [0,250,1500,5000,9999]:
        assert {sampler.sample(clock) for _ in range(100)}=={3}
    sampler.mastery.unlocked_at=250
    assert {sampler.sample(9999) for _ in range(100)}=={3}


def test_native_admission_and_actual_passive_joint_limits_remain_physical():
    evidence=validate_native_admission(BANK,ROOT,PLAN['timing'])
    assert len(evidence['cases'])==2 and max(r['closure_peak_m'] for r in evidence['cases'])<.000025
    spec=json.loads((ROOT/'model/wheel_leg_v2.0.2/friction_v1/training/model_spec.json').read_text())
    for name,sign in [('L_joint2',1.),('R_jonit2',-1.)]:
        joint=next(j for j in spec['joints'] if j['name']==name)
        inner=sorted(math.degrees(sign*float(joint['limit'][k])+spec['passive_knee_zero_inner_rad'][name]) for k in ['lower','upper'])
        assert inner==pytest.approx([40.,110.],abs=1e-8)


def test_configured_guard_changes_termination_and_cold_admission_together(tmp_path):
    cfg,settings,_=configuration(tmp_path,PLAN)
    assert cfg['closure_gap_termination_m']==settings['closure_gap_m']==.005
    assert PLAN['global_curriculum']['limits']['closure_peak_m']==.005
    assert PLAN['mechanical_inner_knee_range_deg']==[40.,110.]
    for value in [.003,.006]:
        bad=deepcopy(PLAN);bad['closure_guard']['peak_m']=value
        with pytest.raises(ValueError,match='5mm'):validate_plan(bad)


def test_standard_cold_review_has_no_four_class_or_unseen_pose_claim():
    rows=trials();c=course()
    assert len(rows)==4 and {r['split'] for r in rows}=={'standard'}
    assert {r['pose_index'] for r in rows}=={3} and len({r['heading_rad'] for r in rows})==4
    assert c.recovery.review(rows,250)['all_classes_passed']
    c.levels.update({s:len(c.config['skills'][s]['levels']) for s in SKILLS});c.retained_levels=dict(c.levels)
    assert c.all_mastered and not c.complex_passed
    assert ('complex',0) not in c.jobs(500)
    summaries={s:dict(passed=True) for s in [*SKILLS,'belly']}
    c.finish_review(summaries,500)
    assert c.domains_admitted
    rows[0]['passed']=False;c.recovery.review(rows,750)
    assert not c.all_mastered and c.domain_context()==('nominal',0.)
    restored=course();restored.load_state_dict(c.state_dict())
    assert not restored.recovery.unlocked and not restored.all_mastered


def test_standard_scope_cannot_silently_unlock_complex_births():
    bad=deepcopy(PLAN['multi_pose_recovery']);bad['curriculum'][-1]['family_weights']['left_side']=.1
    bad['curriculum'][-1]['family_weights']['belly_down']=.9
    with pytest.raises(ValueError,match='Standard'):validate(bad)
    bad=deepcopy(PLAN);bad['global_curriculum'].pop('recovery_scope')
    with pytest.raises(ValueError,match='scope disagree'):validate_plan(bad)

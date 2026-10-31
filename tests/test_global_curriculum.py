"""Admission cannot be bought with elapsed updates, pooled wins or zero motion."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.global_curriculum import GlobalCurriculum,SKILLS,assessment_descriptors,validate
from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery
from wheeled_tasks.chassis.prone_motion import commands,domain_stage
from wheeled_tasks.chassis.prone_motion_cli import resolve_plan,train_parser,validate_resolved
from evaluate_global_curriculum import assess_trace

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4.json').read_text())
CONFIG=PLAN['global_curriculum']


def course():
    return GlobalCurriculum(PLAN,RecoveryMastery(PLAN['multi_pose_recovery']['mastery_gate']))


def rows(skill,level):
    return [dict(r,passed=True) for r in assessment_descriptors(CONFIG,skill,level)]


def admit(c,skill,level,update):
    return c.review_skill(skill,level,rows(skill,level),update)


def test_all_initial_lanes_are_literal_nominal_standing_and_updates_never_unlock_anything():
    c=course()
    for n in (20,32,256):
        for update in (0,1000,4500,9999,10000):
            cases=c.training_descriptors(n,update)
            assert len(cases)==n and {r['role'] for r in cases}=={'stand'}
            assert all(r['nominal_birth'] and r['target']==[0.,0.,.32] for r in cases)
            assert domain_stage(PLAN,update,c)==('nominal',0.)
    with pytest.raises(ValueError):admit(c,'straight',0,250)


def test_standing_then_independent_belly_straight_and_height_branches():
    c=course();bad=rows('standing',0);bad[0]['passed']=False
    assert not c.review_skill('standing',0,bad,250)['passed']
    admit(c,'standing',0,500)
    kinds={r['curriculum_skill'] for r in c.training_descriptors(256,1)}
    assert kinds=={'standing','recovery','straight','static_height'}
    assert c.learning_level('turn') is None and c.learning_level('spin') is None
    assert c.learning_level('straight')==0 and not c.recovery.unlocked
    with pytest.raises(ValueError):c.review_complex([],750)


def test_complex_recovery_keeps_every_family_leg_denominator_and_streak():
    from wheeled_tasks.chassis.multi_pose_recovery import FAMILIES,LEG_CLASSES
    c=course();c.levels['standing']=1;c.recovery.unlocked_at=250
    c.config['required_consecutive_passes']=2
    values=[dict(family=f,leg_class=leg,split='holdout',pose_index=i*2+j,heading_rad=0.,passed=True)
        for i,(f,leg) in enumerate((f,l) for f in FAMILIES if f!='belly_down' for l in LEG_CLASSES) for j in range(2)]
    bad=deepcopy(values);bad[-1]['passed']=False
    assert not c.review_complex(bad,500)['passed'] and not c.complex_passed
    assert c.review_complex(values,750)['passed'] and not c.complex_passed
    assert c.review_complex(values,1000)['passed'] and c.complex_passed


def test_speed_turn_spin_and_height_prerequisites_are_at_matching_levels():
    c=course();admit(c,'standing',0,250)
    admit(c,'straight',0,500);admit(c,'straight',1,750)
    assert c.learning_level('straight')==1  # 1m/s waits for low turns/spin.
    assert c.learning_level('turn')==c.learning_level('spin')==0
    assert c.learning_level('moving_height') is None
    admit(c,'static_height',0,500);admit(c,'static_height',1,750)
    assert c.learning_level('moving_height')==0
    for s in ('turn','spin'):
        admit(c,s,0,1000);admit(c,s,1,1250)
    assert c.learning_level('straight')==2
    admit(c,'straight',2,1500)
    assert c.learning_level('straight')==2 and c.learning_level('braking')==0
    admit(c,'braking',0,1750)
    assert c.learning_level('straight')==3
    assert c.learning_level('turning_height')==c.learning_level('spinning_height')==0


def test_missing_direction_repetition_and_one_bad_height_group_do_not_unlock():
    c=course();c.levels.update(standing=1,straight=2,static_height=2,turn=2)
    c.config['minimum_success_rate']=.9
    values=rows('turning_height',0);values[0]['passed']=False
    assert not c.review_skill('turning_height',0,values,250)['passed']
    assert c.levels['turning_height']==0
    with pytest.raises(ValueError):c.review_skill('turning_height',0,values,250)
    assert not c.review_skill('turning_height',0,rows('turning_height',0)[:-1],500)['passed']
    values=rows('turning_height',0);values.append(deepcopy(values[0]))
    with pytest.raises(ValueError):c.review_skill('turning_height',0,values,750)


def test_higher_skill_replay_retains_all_groups_levels_and_four_arc_directions():
    c=course()
    c.levels={s:len(CONFIG['skills'][s]['levels']) for s in SKILLS}
    counts={s:0 for s in CONFIG['training_weights']};seen=set();levels=set()
    for generation in range(20):
        cases=c.training_descriptors(256,generation)
        assert all(sum(r['curriculum_skill']==s for r in cases)>0 for s in counts)
        for r in cases:
            counts[r['curriculum_skill']]+=1
            if r['curriculum_skill']=='turning_height':seen.add((r['role'],r['target'][2]))
            if r['curriculum_skill']=='straight':levels.add(abs(r['target'][0]))
    assert len(seen)==8 and levels=={.2,.5,1.,2.}
    assert abs(counts['standing']/20-64)<=1 and abs(counts['recovery']/20-76.8)<=1


def synthetic_trace(skill,level=0):
    cases=assessment_descriptors(CONFIG,skill,level);n=len(cases);steps=2000;dt=.01
    times=torch.arange(1,steps+1,dtype=torch.float64)*dt
    targets=torch.tensor([r['target'] for r in cases],dtype=torch.float64)
    cmd=np.stack([commands(targets,t.repeat(n),torch.full((n,),float('nan')),
        torch.zeros(n,dtype=torch.bool),PLAN,moving_height=torch.tensor([r['moving_height'] for r in cases]),
        abrupt_brake=torch.tensor([r['abrupt_brake'] for r in cases])).numpy() for t in times])
    yaw=np.array([r['heading_rad'] for r in cases])[None]+np.cumsum(cmd[:,:,1]*dt,axis=0)
    pos=np.zeros((steps,n,3));pos[:,:,0]=np.cumsum(cmd[:,:,0]*np.cos(yaw)*dt,axis=0)
    pos[:,:,1]=np.cumsum(cmd[:,:,0]*np.sin(yaw)*dt,axis=0)
    v=np.zeros_like(pos);v[:,:,0]=cmd[:,:,0]
    w=np.zeros_like(pos);w[:,:,2]=cmd[:,:,1]
    gravity=np.zeros_like(pos);gravity[:,:,2]=-1
    a=dict(time=np.repeat(times.numpy()[:,None],n,axis=1),command=cmd,position=pos,velocity=v,omega=w,
        gravity=gravity,height=cmd[:,:,2].copy(),gap=np.zeros((steps,n)),terminated=np.zeros((steps,n),bool),
        valid=np.ones((steps,n),bool),air_run=np.zeros((steps,n,2)))
    return a,cases,np.zeros((n,2))


def test_native_metric_gate_cannot_pass_low_speed_by_standing_or_spin_by_wheel_command():
    a,cases,xy=synthetic_trace('straight')
    assert all(r['passed'] for r in assess_trace(a,cases,xy,CONFIG['limits'],.01,2000))
    a['velocity']*=0
    assert not any(r['passed'] for r in assess_trace(a,cases,xy,CONFIG['limits'],.01,2000))
    a,cases,xy=synthetic_trace('spin',3)
    assert all(r['passed'] for r in assess_trace(a,cases,xy,CONFIG['limits'],.01,2000))
    a['omega']*=-1
    assert not any(r['passed'] for r in assess_trace(a,cases,xy,CONFIG['limits'],.01,2000))


def test_combined_height_changes_during_motion_and_abrupt_brake_checks_actual_stop():
    a,cases,xy=synthetic_trace('moving_height')
    assert (np.abs(a['command'][599,:,:2]).sum(axis=1)>0).all()
    assert (a['command'][599,:,2]==.305).all() and (a['command'][899,:,2]!=.305).all()
    assert all(r['passed'] for r in assess_trace(a,cases,xy,CONFIG['limits'],.01,2000))
    a['height'][:]=.305
    assert not any(r['passed'] for r in assess_trace(a,cases,xy,CONFIG['limits'],.01,2000))
    a,cases,xy=synthetic_trace('braking')
    assert np.abs(a['command'][1198,:,0]).min()==1 and not a['command'][1199,:,:2].any()
    assert all(r['passed'] for r in assess_trace(a,cases,xy,CONFIG['limits'],.01,2000))
    a['velocity'][1200:1600,:,0]=1.
    assert not any(r['passed'] for r in assess_trace(a,cases,xy,CONFIG['limits'],.01,2000))


def test_randomization_needs_all_skills_retention_and_minimum_training_per_domain():
    c=course();all_pass={s:dict(passed=True) for s in (*SKILLS,'belly','complex')}
    c.finish_review(all_pass,9000)
    assert c.domain_context()==('nominal',0.)
    # Synthetic admission is solely a unit fixture, never learner evidence.
    c.levels={s:len(CONFIG['skills'][s]['levels']) for s in SKILLS}
    c.recovery.unlocked_at=250;c.complex_passed=True
    c.finish_review(all_pass,9250)
    assert c.domain_context()==('sensors_only',.25)
    before=c.domain_context();c.finish_review(all_pass,9251)
    assert c.domain_context()==before
    bad=deepcopy(all_pass);bad['spin']['passed']=False
    c.finish_review(bad,9500);assert c.domain_context()==before
    c.finish_review(all_pass,9750);assert c.domain_context()!=before
    before=c.domain_context();c.finish_review(all_pass,10000)
    assert c.domain_context()==before and c.max_updates==10000


def test_contract_cycles_cli_thresholds_legacy_runtime_and_formal_no_checkpoint():
    bad=deepcopy(CONFIG);bad['skills']['standing']['requires']=[dict(skill='straight',levels=1)]
    with pytest.raises(ValueError,match='Cyclic'):validate(bad)
    p=resolve_plan(PLAN,curriculum_eval_every=100,curriculum_success_rate=.9,curriculum_required_passes=2)
    assert p['global_curriculum']['evaluation_interval_updates']==100
    assert p['global_curriculum']['required_consecutive_passes']==2
    legacy=json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4_recovery_20261031.json').read_text())
    p=resolve_plan(legacy)
    for key in ('curriculum_eval_every','curriculum_success_rate','curriculum_required_passes',
                'position_iterations','velocity_iterations'):
        del p['runtime_cli']['options'][key]
    assert validate_resolved(p)==p
    with pytest.raises(SystemExit):train_parser().parse_args(['--checkpoint','old.pt'])


def test_final_matrix_preserves_78_cases_and_adds_all_height_combinations_and_braking():
    from evaluate_prone_motion import evaluation_cases
    import play_prone_motion as play
    bank=json.loads((ROOT/PLAN['pose_bank']['file']).read_text())
    env=SimpleNamespace(poses=bank['poses'],prone_index=3,height_pose={.23:0,.305:1,.42:2})
    old=json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4_recovery_20261031.json').read_text())
    cases=evaluation_cases(env,PLAN)
    assert len(cases)==96 and cases[:78]==evaluation_cases(env,old)
    combined=[c for c in cases if c.get('moving_height')]
    assert len(combined)==16 and len({(r['role'],r['target'][2]) for r in combined})==16
    args=play.parser().parse_args(['--checkpoint','example.pt','--scene','reverse','--height','.23','--abrupt-brake'])
    case=play.case_for(args,env,PLAN)
    assert case['moving_height'] and case['abrupt_brake'] and case['target'][0]==-2.
    with pytest.raises(ValueError):play.case_for(args,env,old)


def test_global_hook_refreshes_observations_keeps_adam_and_rejects_partial_rollout(tmp_path,monkeypatch):
    import evaluate_global_curriculum as evaluation
    from wheeled_algo.global_curriculum import GlobalCurriculumHook
    c=course();actor=torch.nn.Linear(1,1);critic=torch.nn.Linear(1,1)
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()))
    env=SimpleNamespace(num_envs=20,device='cpu',dt=.001,policy_dt=.01,
        reset=lambda ids:None,get_observations=lambda:torch.tensor([[2.]]))
    calls=[];alg=SimpleNamespace(actor=actor,critic=critic,optimizer=optimizer,storage=SimpleNamespace(step=0),
        act=lambda obs:calls.append(obs.clone()) or obs)
    runner=SimpleNamespace(alg=alg,env=env,device='cpu',logger=SimpleNamespace(writer=None))
    monkeypatch.setattr(evaluation,'evaluate_skill',lambda *a,**kw:(dict(cases=rows('standing',0)),dict(x=np.zeros(1))))
    hook=GlobalCurriculumHook(runner,c,completed=lambda:250,output=tmp_path)
    assert torch.equal(alg.act(torch.tensor([[1.]])),torch.tensor([[2.]]))
    assert c.levels['standing']==1 and not optimizer.state and len(calls)==1
    alg.storage.step=1
    # Standing admission made belly tests newly eligible at this same update;
    # no second review may start inside the now nonempty rollout.
    assert not c.due(250)
    alg.act(torch.tensor([[3.]]));assert len(calls)==2
    with pytest.raises(RuntimeError):hook.review(500)
    hook.close()


def test_recovery_cold_review_restores_legacy_role_pool_even_without_global_override():
    import random
    from evaluate_recovery_mastery import evaluate
    bank=json.loads((ROOT/PLAN['pose_bank']['file']).read_text())
    original=['stand','forward','reverse','prone']*5
    env=SimpleNamespace(num_envs=20,device='cpu',dt=.001,policy_dt=.01,poses=bank['poses'],
        evaluation_cases=None,fixed_cases=None,course_roles=list(original),generator=torch.Generator(),
        domain=SimpleNamespace(rng=torch.Generator()),recovery_sampler=SimpleNamespace(rng=random.Random(3),class_cursor={}))
    env.get_observations=lambda:torch.zeros(20,1)
    def reset(ids):
        if env.evaluation_cases is not None:env.course_roles=[c['role'] for c in env.evaluation_cases]
    def step(action):
        d=dict(stable=torch.zeros(20,dtype=torch.bool),recovered=torch.zeros(20,dtype=torch.bool),
            terminated=torch.ones(20,dtype=torch.bool),gap=torch.zeros(20),
            recovery_completed_event=torch.zeros(20,dtype=torch.bool),recovery_active_seconds=torch.zeros(20))
        return env.get_observations(),torch.zeros(20),torch.ones(20,dtype=torch.bool),dict(diagnostics=d)
    env.reset=reset;env.step=step
    result=evaluate(env,torch.nn.Identity(),PLAN['multi_pose_recovery']['mastery_gate'])
    assert env.course_roles==original and env.evaluation_cases is None
    assert len(result['cases'])==16 and not any(r['passed'] for r in result['cases'])

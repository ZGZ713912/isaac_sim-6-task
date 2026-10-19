"""Seal the bounded population trial after every native owner exits."""
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;RUN=ROOT/'runs/flat_base_20261019'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
path=RUN/'feedback1264_population_queue001.json';q=read(path)
out=HERE/'feedback1264_population_completion001.json'
if out.exists():raise SystemExit('Preserve completed trial seal')
if q['status']!='complete' or any(Path('/proc/'+str(q[k])).exists() for k in ('pid','worker_pid')):
    raise SystemExit('Wait for the queue and all native owners to exit')
contract_path=HERE/'continuation1264_population_contract001.json';contract=read(contract_path)
folder=RUN/'pilot024';final=folder/'model_001264.pt';d=read(folder/'progress.json')
info=torch.load(final,map_location='cpu',weights_only=True)['infos']
native_path=HERE/'feedback1264_native_milestones.json';native=read(native_path)
checks=dict(all_five_children_exit0=len(q['history'])==5 and all(r['exit_code']==0 for r in q['history']),
    first_native_verified=q.get('first_native_verification_exit_code')==0,
    exact_frozen_sources=all(sha(ROOT/n)==v for n,v in q['source_sha256'].items()),
    exact_contract=sha(contract_path)==q['contract_sha256'],
    exact_parent_seals=sha(HERE/'population_capacity_completion002.json')==contract['capacity_completion_sha256']
        and sha(HERE/'feedback1248_height_coverage_completion001.json')==contract['parent_height_completion_sha256']
        and sha(HERE/'feedback1248_behavior_review001.json')==contract['parent_behavior_review_sha256'],
    actual_complete_checkpoint=d['status']=='block_complete' and sha(final)==q['checkpoint_sha256']==d['final_checkpoint_sha256'],
    actual_final_counts=(info['successful_updates_total'],info['actor_updates_total'],info['active_branch_actor_updates'],
        info['feedback_branch_actor_updates'],info['post_split_actor_updates'],info['inherited_shared_feedback_updates'],
        info['consumed_optimizer_updates_total'],info['critic_warmup_remaining'],info['training_transitions_total'])
        ==(1264,1084,44,706,402,304,1265,0,67633152),
    native_training_and_height_metrics=native['passed'] and all(native['checks'].values()),
    no_more_training_heldout_or_hardware=not q['subsequent_training_queued'] and not q['heldout_seed_used']
        and not q['hardware_motion_started'])
events=EventAccumulator(str(folder),size_guidance={'scalars':0});events.Reload()
scalars={tag:events.Scalars(tag) for tag in events.Tags()['scalars']}
failure_events=scalars.get('/reward/failure_event',[])
checks['all16_finite_failure_event_points']=len(failure_events)==16 and all(
    math.isfinite(point.value) for point in failure_events)
cohorts={}
for row in q['history'][2:]:
    name=row['phase'];evaluation_path=RUN/name/'evaluation.json';evaluation=read(evaluation_path)
    cases=evaluation['result']['cases'];prefix=evaluation['tensorboard_namespace'];expected={prefix+'/passed_cases':sum(c['passed'] for c in cases.values())}
    for case_name,case in cases.items():
        for metric in ('survival_rate','height_mae_m','vx_mae_m_s','yaw_mae_rad_s','stand_drift_max_m','planar_speed_mean_m_s','closure_gap_max_m'):
            expected[prefix+'/'+case_name+'/'+metric]=case[metric]
    ok=[]
    for tag,value in expected.items():
        points=scalars.get(tag,[]);ok.append(len(points)==1 and points[0].step==1264
            and math.isclose(points[0].value,value,rel_tol=2e-7,abs_tol=2e-8))
    checks[name+'_identity']=evaluation['status']=='evaluated' and sha(evaluation_path)==row['evaluation_sha256']
    checks[name+'_RSL_events']=all(ok)
    checks[name+'_comparisons_bound']=all(sha(ROOT/c['path'])==c['sha256'] for c in row['comparisons'].values())
    cohorts[name]=dict(evaluation_sha256=sha(evaluation_path),traces_sha256=sha(RUN/name/'traces.npz'),
        passed_cases=sum(c['passed'] for c in cases.values()),requested_cases=len(cases),
        full_horizon_episodes=sum(c['full_horizon_episodes'] for c in cases.values()),
        requested_episodes=sum(c['requested_episodes'] for c in cases.values()),comparisons=row['comparisons'],
        terminations={n:sum(c['termination_reasons'][n] for c in cases.values()) for n in next(iter(cases.values()))['termination_reasons']})
if not all(checks.values()):print(json.dumps(checks,indent=2));raise SystemExit('Trial identity or event proof failed')
out.write_text(json.dumps(dict(schema='flat-feedback1264-population-completion-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    passed=True,checks=checks,queue_sha256=sha(path),contract_sha256=sha(contract_path),checkpoint_sha256=sha(final),
    native_verification_sha256=sha(native_path),cohorts=cohorts,
    failure_event_nonzero_points=sum(point.value!=0 for point in failure_events),
    event_file_sha256={p.name:sha(p) for p in folder.glob('events.out.tfevents.*')},
    behavior_qualified=False,all_release_cohorts_complete=False,heldout_seed_used=False,hardware_motion_started=False),indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),cohorts=cohorts),indent=2))

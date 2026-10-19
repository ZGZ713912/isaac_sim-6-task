"""Read all sixteen native updates and height strata from the existing RSL event files."""
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;RUN=ROOT/'runs/flat_base_20261019/pilot025'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=HERE/'low_gate1280_training_review001.json'
if out.exists():raise SystemExit('Preserve the completed training review')
progress=json.loads((RUN/'progress.json').read_text())
native_path=HERE/'low_gate_native1280.json'
if progress['status']!='block_complete' or progress['successful_updates_total']!=1280 or not native_path.exists():
    raise SystemExit('Wait for all actual updates and the native final proof')
native=json.loads(native_path.read_text())
if not native['passed']:raise SystemExit('The native clock/height proof did not pass')
events=EventAccumulator(str(RUN),size_guidance={'scalars':0});events.Reload();tags=set(events.Tags()['scalars']);cache={}
def scalar(tag):
    if tag not in cache:cache[tag]={row.step:row.value for row in events.Scalars(tag)} if tag in tags else {}
    return cache[tag]
steps=list(range(1265,1281));native_steps=list(range(1264,1280))
if sorted(scalar('FlatTransfer/actor_updates_total'))!=steps:
    raise SystemExit('The sixteen completed update event points are incomplete')
if any(scalar('FlatTransfer/training_transitions_total')[n]!=67633152+(n-1264)*4096*96 for n in steps):
    raise SystemExit('The cumulative population exposure clock changed')
groups=('stand_low','stand_mid','stand_high','backward_low','forward_low','backward_edge3','forward_edge3',
        'turn_left_edge4','turn_right_edge4','start_stop_1_0','start_stop_3_0','curve_left','curve_right')
birth_groups=('stand_low_birth','stand_mid_birth','stand_high_birth','backward_edge3_start','forward_edge3_start')
rows={}
for namespace,names in (('FlatBehavior',groups),('FlatBirth',birth_groups)):
    for group in names:
        for mode in ('','/rmcs_direct','/legacy_ramp'):
            view=namespace+'/'+group+mode
            if view+'/sample_frames' not in tags:continue
            for band in ('','/height_low','/height_nominal','/height_high'):
                prefix=view+band;frames=scalar(prefix+'/sample_frames')
                if sorted(frames)!=steps:raise SystemExit('Incomplete frame counts: '+prefix)
                for metric in ('vx_mae_m_s','yaw_mae_rad_s','height_mae_m','planar_speed_mean_m_s','terminated_frame_fraction'):
                    values=scalar(prefix+'/'+metric);windows={}
                    for label,selected in (('first4',steps[:4]),('final4',steps[-4:]),('all16',steps)):
                        count=sum(frames[n] for n in selected)
                        valid=[n for n in selected if frames[n]>0]
                        if any(n not in values or not math.isfinite(values[n]) for n in valid):
                            raise SystemExit('Nonempty stratum has missing or nonfinite errors: '+prefix+'/'+metric)
                        windows[label]=dict(sample_frames=int(count),nonempty_updates=len(valid),
                            frame_weighted_mean=sum(frames[n]*values[n] for n in valid)/count if count else None)
                    rows[prefix+'/'+metric]=windows
losses={}
for name in ('value','teacher','teacher_action_rmse','teacher_fraction','approx_policy_kl','policy_clip_fraction',
             'actor_gradient_norm','critic_gradient_norm','actor_learning_rate','critic_learning_rate'):
    values=scalar('Loss/'+name)
    if sorted(values)!=native_steps or not all(math.isfinite(v) for v in values.values()):
        raise SystemExit('Missing native PPO diagnostics: '+name)
    losses[name]=dict(points=16,minimum=min(values.values()),maximum=max(values.values()),
        first4=sum(values[n] for n in native_steps[:4])/4,final4=sum(values[n] for n in native_steps[-4:])/4)
failure=scalar('/reward/failure_event')
if sorted(failure)!=native_steps:raise SystemExit('Missing native failure-event reward points')
result=dict(schema='flat-low-demand1280-height-stratified-training-review-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),source='existing RSL-RL TensorBoard event files',
    checkpoint_sha256=sha(RUN/'model_001280.pt'),native_verification_sha256=sha(native_path),
    custom_completed_update_steps=steps,native_iteration_steps=native_steps,
    inherited_successful_training_frames=67633152,additional_successful_training_frames=6291456,
    final_successful_training_frames=73924608,all_discarded_optimizer_consumption_retained=True,
    actor_updates_in_block=16,additional_critic_warmup_updates=0,num_envs=4096,
    failure_event=dict(points=16,nonzero_points=sum(v!=0 for v in failure.values()),minimum=min(failure.values()),maximum=max(failure.values())),
    metrics=rows,PPO_diagnostics=losses,
    event_file_sha256={p.name:sha(p) for p in RUN.glob('events.out.tfevents.*')},
    scope='Sampled training observations only; includes transients after the existing 1s warmup. First/final windows and command modes have different reset and command exposure. No causal population benefit or behavior qualification follows from these means.',
    behavior_qualified=False,new_optimizer_updates_performed_by_reader=0)
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(rows=len(rows),failure_event=result['failure_event'],PPO_diagnostics=losses),indent=2))

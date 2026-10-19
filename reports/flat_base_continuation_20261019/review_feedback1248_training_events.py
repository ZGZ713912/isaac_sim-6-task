"""Read native TensorBoard metrics, including the actual PPO rate diagnostics."""
import hashlib
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/flat_base_20261019/pilot023'
a=EventAccumulator(str(RUN),size_guidance={'scalars':0});a.Reload()
scalar=lambda t:{x.step:x.value for x in a.Scalars(t)}
expected=list(range(1185,1249))
if sorted(scalar('FlatTransfer/actor_updates_total'))!=expected:
    raise SystemExit('Expected all64 completed native update events')
split=scalar("FlatTransfer/post_split_actor_updates")
if sorted(split)!=expected or any(split[step] != 322+max(0,step-1184) for step in expected):
    raise SystemExit("Post-split TensorBoard clock differs from native completed actor updates")
rows={}
for namespace,names in (('FlatBirth',('stand_low_birth','stand_mid_birth','stand_high_birth','forward_edge3_start','backward_edge3_start')),
                        ('FlatBehavior',('stand_low','stand_mid','stand_high','forward_low','backward_low',
                                         'forward_edge3','backward_edge3'))):
    for name in names:
        frames=scalar(f'{namespace}/{name}/sample_frames')
        for metric in ('vx_mae_m_s','height_mae_m','planar_speed_mean_m_s','terminated_frame_fraction'):
            values=scalar(f'{namespace}/{name}/{metric}')
            windows={}
            for window,start,end in (('first10_actor',1185,1194),('final10_actor',1239,1248)):
                selected=[s for s in range(start,end+1) if s in values and frames.get(s,0)>0]
                count=sum(frames[s] for s in selected)
                windows[window]=dict(points=len(selected),sample_frames=count,
                    mean=sum(values[s]*frames[s] for s in selected)/count if count else None)
            selected=[s for s in range(1185,1249) if s in values and frames.get(s,0)>0]
            windows['all_actor_updates']=dict(points=len(selected),
                minimum=min((values[s] for s in selected),default=None),
                maximum=max((values[s] for s in selected),default=None),
                maximum_step=max(selected,key=values.get) if selected else None)
            rows[f'{namespace}/{name}/{metric}']=windows
native_steps=list(range(1184,1248))
diagnostics={}
for metric,unit in (('approx_policy_kl','nats per action sample'),('policy_clip_fraction','fraction'),
                    ('actor_gradient_norm','L2 before clipping'),('critic_gradient_norm','L2 before clipping'),
                    ('actor_learning_rate','Adam actor group rate'),('critic_learning_rate','Adam critic group rate'),
                    ('teacher','normalized action-error squared'),('teacher_fraction','fraction of current training frames'),
                    ('teacher_action_rmse','raw action RMSE on selected current frames')):
    values=scalar('Loss/'+metric)
    if sorted(values)!=native_steps:
        raise SystemExit('Incomplete native diagnostic events: '+metric)
    diagnostics[metric]=dict(points=len(values),unit=unit,minimum=min(values.values()),maximum=max(values.values()),
        first10=sum(values[s] for s in native_steps[:10])/10,final10=sum(values[s] for s in native_steps[-10:])/10,
        first10_actor=sum(values[s] for s in range(1184,1194))/10,
        final10_actor=sum(values[s] for s in range(1238,1248))/10)
failure=scalar('/reward/failure_event')
if sorted(failure)!=native_steps:
    raise SystemExit('Incomplete iteration-level reward events')
report=dict(schema='flat-feedback1248-training-event-review-v1',source='existing RSL-RL TensorBoard event files',
    checkpoint_sha256=hashlib.sha256((RUN/'model_001248.pt').read_bytes()).hexdigest(),custom_update_steps=expected,
    native_iteration_steps=[1184,1247],sampled_training_behavior_only=True,behavior_qualified=False,
    inherited_shared_feedback_updates=304,post_split_actor_updates=386,critic_only_adaptation_updates=0,actor_updates_in_block=64,
    failure_event=dict(points=len(failure),minimum=min(failure.values()),maximum=max(failure.values()),nonzero_points=sum(v!=0 for v in failure.values())),
    PPO_diagnostics=diagnostics,metrics=rows)
report['event_file_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RUN.glob('events.out.tfevents.*'))}
report['clock_scope']='50Hz policy;1000Hz physics/fresh-feedback PD, unchanged v16 full resume with64actor updates and no new critic warmup'
output=Path(__file__).with_name('feedback1248_training_event_review001.json')
if output.exists():
    raise SystemExit('Preserve the completed event review')
output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(failure_event=report['failure_event'],PPO_diagnostics=diagnostics,
    metrics={n:r for n,r in rows.items() if n.endswith('/vx_mae_m_s')})))

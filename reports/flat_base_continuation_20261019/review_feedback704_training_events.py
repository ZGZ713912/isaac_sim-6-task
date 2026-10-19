"""Read native RSL-RL TensorBoard data; this does not qualify behavior."""
import hashlib
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/flat_base_20261019/pilot014'
a=EventAccumulator(str(RUN),size_guidance={'scalars':0});a.Reload()
scalar=lambda t:{x.step:x.value for x in a.Scalars(t)}
updates=scalar('FlatTransfer/actor_updates_total')
expected=list(range(641,705))
if sorted(updates)!=expected:
    raise SystemExit('Expected all 64 completed native update events')
rows={}
for namespace,names in (('FlatBirth',('stand_low_birth','stand_mid_birth','stand_high_birth','forward_edge3_start','backward_edge3_start')),
                        ('FlatBehavior',('stand_low','stand_mid','stand_high','forward_edge3','backward_edge3'))):
    for name in names:
        frames=scalar(f'{namespace}/{name}/sample_frames')
        for metric in ('vx_mae_m_s','height_mae_m','planar_speed_mean_m_s','terminated_frame_fraction'):
            values=scalar(f'{namespace}/{name}/{metric}')
            windows={}
            for window,start,end in (('pre_actor_last10',651,660),('final10',695,704)):
                selected=[s for s in range(start,end+1) if s in values and frames.get(s,0)>0]
                count=sum(frames[s] for s in selected)
                windows[window]=dict(points=len(selected),sample_frames=count,
                    mean=sum(values[s]*frames[s] for s in selected)/count if count else None)
            rows[f'{namespace}/{name}/{metric}']=windows
failure=scalar('/reward/failure_event')
if sorted(failure)!=list(range(640,704)):
    raise SystemExit('Expected all64 native iteration-level reward events')
report=dict(schema='flat-feedback704-training-event-review-v1',source='existing RSL-RL TensorBoard event files',
 checkpoint_sha256=hashlib.sha256((RUN/'model_000704.pt').read_bytes()).hexdigest(),steps=expected,
 sampled_training_behavior_only=True,behavior_qualified=False,
 failure_event=dict(points=len(failure),native_iteration_steps=[640,703],minimum=min(failure.values()),maximum=max(failure.values()),nonzero_points=sum(v!=0 for v in failure.values())),
 metrics=rows)
Path(__file__).with_name('feedback704_training_event_review001.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(failure_event=report['failure_event'],metrics={n:r for n,r in rows.items() if n.endswith('/vx_mae_m_s')})))

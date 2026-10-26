"""Export completed trial curves directly from the native TensorBoard event."""
import json,time
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
from train_robust_getup import digest,write_json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
run=B/'pilot013_projected_entry';p=json.loads((run/'progress.json').read_text());assert p['status']=='completed' and p['new_optimizer_updates']==8
events=list(run.glob('events.out.tfevents.*'));assert len(events)==1
event=events[0];a=EventAccumulator(str(event),size_guidance={'scalars':0}).Reload()
tags=[('EntryProjection/far_signed_lead_min_rad','Far phase signed lead (rad)',.014),('EntryProjection/stopped_target_peak_rad','Stopped target peak (rad)',.006),('EntryProjection/recorded_step_target_peak_rad','Recorded target step (rad)',.005),('Loss/value','Native critic loss',None),('Loss/surrogate','Native surrogate loss',None),('UnifiedGetup/recovered_events','Training recovered events per step',None)]
fig,axes=plt.subplots(2,3,figsize=(13,7));records={}
for ax,(tag,title,line) in zip(axes.flat,tags,strict=True):
 rows=a.Scalars(tag);assert len(rows)==8,(tag,len(rows))
 records[tag]=[dict(step=r.step,value=r.value,wall_time=r.wall_time) for r in rows]
 ax.plot([r.step+1 for r in rows],[r.value for r in rows],marker='o',markersize=3)
 if line is not None:ax.axhline(line,color='tab:red',linestyle='--',linewidth=1)
 ax.set_title(title);ax.set_xlabel('Completed branch learning clock');ax.grid(alpha=.25)
fig.suptitle('Projected Adam pilot013: 8 bounded native updates\nTraining curves do not qualify physical recovery')
fig.tight_layout(rect=(0,0,1,.92));target=R/'projected_entry_TB001.png';fig.savefig(target,dpi=160);plt.close(fig)
write_json(R/'projected_entry_TB_export001.json',dict(schema='TensorBoard-event-export-v1',event=str(event),event_sha256=digest(event),records=records,plot=str(target),plot_sha256=digest(target),native_event_steps='RSL iteration; completed branch clock equals event step+1',new_PPO=0,qualification=None,created_unix=time.time()))
print(json.dumps(dict(plot=str(target),scalar_count=sum(map(len,records.values())))))

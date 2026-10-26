"""Export the paired training curves only from the native RSL event files."""
import json
from pathlib import Path
import sys

sys.path[:0]=['src','scripts']
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from train_robust_getup import digest,write_json

R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
tags=[('Standing/request_position_radius_m','Standing radius (m)'),
      ('Standing/dual_support_fraction','Two-wheel support fraction'),
      ('Standing/position_deadband_cost_density','Position cost density'),
      ('Standing/drift_paid_density','Paid drift cost density'),
      ('LegQuality/physical_speed_peak_rad_s','Physical leg speed peak (rad/s)')]
fig,axes=plt.subplots(3,2,figsize=(11,9),layout='constrained');axes=axes.ravel();receipts=[]
for name,label in [('pilot010_quiet_control','5 cm reward'),('pilot011_quiet_tight','2 cm reward')]:
    run=B/name
    assert json.loads((run/'progress.json').read_text())['status']=='completed'
    files=sorted(run.glob('events.out.tfevents.*'))
    assert len(files)==1
    event=files[0];acc=EventAccumulator(str(event),size_guidance={'scalars':0}).Reload()
    records={}
    for ax,(tag,title) in zip(axes,tags,strict=False):
        samples=acc.Scalars(tag)
        assert len(samples)==8
        steps=[x.step-1500 for x in samples];values=[x.value for x in samples]
        ax.plot(steps,values,marker='o',label=label);ax.set(title=title,xlabel='Native rollout / PPO update')
        ax.grid(alpha=.2);records[tag]=dict(step=[x.step for x in samples],value=values)
    receipts.append(dict(run=str(run),event=str(event),event_sha256=digest(event),curves=records))
for ax in axes[:5]:
    ax.legend()
axes[-1].axis('off');axes[-1].text(0.,.8,'8 independent updates per branch\nSame initialization and first native rollout\nTraining scalars do not qualify behavior',fontsize=11,va='top')
fig.suptitle('Quiet feedback reward contrast - native TensorBoard events')
target=R/'quiet_reward_AB_training001.png';fig.savefig(target,dpi=170);plt.close(fig)
write_json(R/'quiet_reward_AB_TB_export001.json',dict(schema='native-RSL-event-figure-v1',sources=receipts,
    figure=str(target),figure_sha256=digest(target),qualification=None,hardware_started=False))
print(target)

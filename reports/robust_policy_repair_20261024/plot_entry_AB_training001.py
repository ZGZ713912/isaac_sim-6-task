"""Export paired training curves directly from the existing RSL TensorBoard events."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
r=Path('reports/robust_policy_repair_20261024');b=Path('runs/robust_policy_repair_20261024');arms=[('pilot007_entry_support0','Original reward','#526781'),('pilot008_entry_support1','Support cost = 1','#cc6d29')]
fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),layout='constrained')
channels=[('UnifiedGetup/recovery_support_cost_density','Support cost density'),('Loss/value','Critic loss'),('Loss/surrogate','PPO surrogate loss'),('UnifiedGetup/recovered_events','Recovered-event logger signal'),('LegQuality/physical_speed_peak_rad_s','Rollout leg-speed logger [rad/s]'),('LegQuality/paid_density','Rollout leg-quality cost density')]
sources=[]
for name,label,color in arms:
 folder=b/name;p=json.loads((folder/'progress.json').read_text());assert p['status']=='completed' and p['new_optimizer_updates']==8
 acc=EventAccumulator(str(folder),size_guidance={'scalars':0});acc.Reload()
 files=list(folder.glob('events.out.tfevents.*'));sources.extend(dict(path=str(x),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in files)
 for ax,(tag,title) in zip(axes.flat,channels):
  values=acc.Scalars(tag);assert len(values)==8,(name,tag,len(values));ax.plot([v.step-1500 for v in values],[v.value for v in values],marker='o',markersize=3,label=label,color=color);ax.set_title(title,fontsize=10);ax.set_xlabel('Native PPO update in this arm');ax.set_xticks(range(1,9));ax.grid(alpha=.22)
for ax in axes.flat:ax.legend(fontsize=8)
fig.suptitle('Paired entry-only learning: 8 updates per arm\nTraining logger signals; fixed native behavior is assessed separately',fontsize=13)
target=r/'entry_AB_training001.png';fig.savefig(target,dpi=145);plt.close(fig)
(r/'entry_AB_training001.sources.json').write_text(json.dumps(dict(schema='tensorboard-curve-export-v1',events=sources,tags=[x[0] for x in channels],output=str(target),new_PPO=0,qualification=None),indent=2)+'\n');print(target)

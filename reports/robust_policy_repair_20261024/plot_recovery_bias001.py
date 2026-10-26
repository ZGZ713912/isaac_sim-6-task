"""Export authenticated native traces through existing TensorBoard events."""
from pathlib import Path
import hashlib,json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'runs/robust_policy_repair_20261024';R=Path(__file__).resolve().parent
cases=['right_symmetric__clean','left_symmetric__combined'];runs=[('Corrected source','diagnosis012_final_circular_fixed','#596670'),('Neural bias 0.15','diagnosis014_recovery_bias015_fixed','#237bb2'),('Neural bias 0.30','diagnosis014_recovery_bias030_fixed','#2b8c68')];tags=['body_height_m','wheel_clearance_mean_m','leg_speed_peak_rad_s'];inputs=[]
for label,run,col in runs:
 path=B/run/'evaluation';rep=json.loads((path/'report.json').read_text());raw=path/'trajectory.npz';assert hashlib.sha256(raw.read_bytes()).hexdigest()==rep['trajectory_sha256'];z=np.load(raw)
 with SummaryWriter(str(path/'metrics'),filename_suffix='.recovery-bias-native-trace') as w:
  for case in cases:
   lane=next(x for x in rep['lanes'] if x['name']==case);i=lane['lane'];active=z['alive'][:,i]&~z['passive'][:,i];series=[z['height'][:,i],z['wheel_clearance'][:,i].mean(-1),z['leg_speed_peak_rad_s'][:,i]]
   for tag,data in zip(tags,series):
    for step in np.flatnonzero(active):w.add_scalar('RecoveryBiasTrace/'+case+'/'+tag,float(data[step]),int(step+1))
   inputs.append(dict(evaluation=str(path),trajectory_sha256=rep['trajectory_sha256'],case=case,actual_family=lane['post_settle_family'],survived=lane['survived'],recovered=lane['recovered'],followup=lane['followup_success'],termination_causes=lane['termination_causes']))
fig,axes=plt.subplots(3,2,figsize=(12,8),constrained_layout=True)
for label,run,color in runs:
 ev=EventAccumulator(str(B/run/'evaluation/metrics'),size_guidance={'scalars':0});ev.Reload()
 for c,case in enumerate(cases):
  for row,tag in enumerate(tags):
   vals=ev.Scalars('RecoveryBiasTrace/'+case+'/'+tag);data={v.step:v.value for v in vals};steps=np.array(sorted(data));ax=axes[row,c];ax.plot(steps*.005,[data[s] for s in steps],color=color,label=label,lw=1.2);ax.set(xlabel='Native time (s)',xlim=(0,8));ax.grid(alpha=.2)
for i,title in enumerate(['Settled upright, legs initially above body','Actual side fall, combined IMU domain']):axes[0,i].set_title(title)
for row,unit in enumerate(['Body height (m)','Mean wheel clearance (m)','Leg speed, 1 ms peak (rad/s)']):
 for ax in axes[row]:ax.set_ylabel(unit)
for ax in axes[0]:ax.axhline(.19,color='#a26735',ls='--',lw=.7)
for ax in axes[1]:ax.axhline(.003,color='#a26735',ls='--',lw=.7)
axes[0,0].legend(fontsize=8)
fig.suptitle('Common leg-drive bias: zero-PPO paired intervention\nSame seed / states / domains; recovery and stability still require confirmation',fontsize=13)
fig.supxlabel('Active control only. Source side-fall ends at impact protection. No hardware qualification.',fontsize=9)
p=R/'recovery_common_bias_native001.png';fig.savefig(p,dpi=170);plt.close(fig);(R/'recovery_common_bias_plot001.json').write_text(json.dumps(dict(schema='native-tensorboard-plot-v1',output=str(p),source='existing native evaluation TensorBoard events',inputs=inputs,policy_step_seconds=.005,new_PPO=0,qualification=None,hardware_started=False),indent=2)+'\n');print(p)

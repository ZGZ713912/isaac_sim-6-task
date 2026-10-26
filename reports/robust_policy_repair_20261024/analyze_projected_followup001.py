"""Decompose actual fixed-trace followup failures without changing any gate."""
import json,math,time
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
import numpy as np
import torch
from train_robust_getup import digest,write_json
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
torch.set_num_threads(1)
def longest(mask):
 x=np.r_[False,mask,False].astype(np.int8);starts=np.flatnonzero(np.diff(x)==1);ends=np.flatnonzero(np.diff(x)==-1)
 return float((ends-starts).max()*.005) if len(starts) else 0.
results=[]
for label,directory in [('best','diagnosis025_motion_damping080_fixed'),('projected8','pilot013_projected_fixed')]:
 path=B/directory/'evaluation/trajectory.npz';z=np.load(path);rep=json.loads((path.parent/'report.json').read_text());rows=[]
 for lane in rep['lanes']:
  if not lane['recovery']:continue
  i=lane['lane'];t=z['time'][:,i];active=z['alive'][:,i]&~z['passive'][:,i]
  obs=z['observations'][:,i];gate=handoff_entry_gate(torch.from_numpy(obs.copy())).ravel().numpy()
  ever=z['recovered'][:,i];ids=np.flatnonzero(ever);first=float(t[ids[0]]) if len(ids) else None
  conditions=dict(tilt=z['gravity'][:,i,2]<-math.cos(math.radians(15)),angular=np.linalg.norm(z['omega'][:,i,:2],axis=-1)<.7,
    height=z['height'][:,i]>.19,contact=z['wheel_contact'][:,i].all(-1),leg_speed=np.abs(z['motor_velocity'][:,i][:,[0,1,3,4]]).max(-1)<3.,
    vx=np.abs(z['velocity'][:,i,0]-z['commands'][:,i,0])<.25,yaw=np.abs(z['omega'][:,i,2]-z['commands'][:,i,1])<.4)
  reconstructed=np.logical_and.reduce(list(conditions.values()))&~z['passive'][:,i]
  assert np.array_equal(reconstructed,z['stable'][:,i] & conditions['vx'] & conditions['yaw'])
  post=active&(t>=max(4.,(first or 8.)+.5));tail=active&(t>=6.)
  follow=active&ever&reconstructed
  contact_break=post&~conditions['contact']
  row=dict(lane=i,name=lane['name'],recovered=lane['recovered'],followup=lane['followup_success'],survived=lane['survived'],
    recovered_s=first,longest_follow_s=longest(follow),tail_stable_fraction=float(z['stable'][tail,i].mean()) if tail.any() else None,
    post_frames=int(post.sum()),post_failure_frames={k:int((post&~v).sum()) for k,v in conditions.items()},
    post_entry_gate_mean=float(gate[post].mean()) if post.any() else None,
    tail_entry_gate_mean=float(gate[tail].mean()) if tail.any() else None,
    contact_break_clearance_peak_m=float(z['wheel_clearance_peak'][contact_break,i].max()) if contact_break.any() else None,
    contact_break_air_run_peak_s=float(z['wheel_air_run_peak'][contact_break,i].max()) if contact_break.any() else None,
    tail_height_mean_m=float(z['height'][tail,i].mean()) if tail.any() else None,
    tail_vx_MAE_m_s=float(np.abs(z['velocity'][tail,i,0]-z['commands'][tail,i,0]).mean()) if tail.any() else None,
    tail_yaw_MAE_rad_s=float(np.abs(z['omega'][tail,i,2]-z['commands'][tail,i,1]).mean()) if tail.any() else None)
  rows.append(row)
 results.append(dict(label=label,trace=str(path),sha256=digest(path),rows=rows))
changes=[]
for a,b in zip(results[0]['rows'],results[1]['rows'],strict=True):
 assert a['name']==b['name']
 if (a['recovered'],a['followup'])!=(b['recovered'],b['followup']):changes.append(dict(name=a['name'],before=a,after=b))
o=dict(schema='projected-PPO-fixed-followup-failure-audit-v1',models=results,behavior_changed_rows=changes,
 scope='Exact decomposition of recorded stability/tracking predicates; no contact, speed, duration or other acceptance threshold changed.',
 new_PPO=0,actual_experiment_consumed=2630,repair_optimizer_updates=250,qualification=None,created_unix=time.time())
write_json(R/'projected_followup_audit001.json',o);print(json.dumps(dict(changes=changes),indent=2))

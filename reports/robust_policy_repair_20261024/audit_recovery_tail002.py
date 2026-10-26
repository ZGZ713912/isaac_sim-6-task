"""Offline tail state and head-output diagnosis on complete native traces."""
from pathlib import Path
import sys,json,hashlib
sys.path[:0]=['src','scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from wheeled_algo.getup_symmetry import canonical_recovery_logits
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.robust_getup import tilt_gate

torch.set_num_threads(1)
base=Path('runs/robust_policy_repair_20261024'); result=dict(schema='recovery-tail-state-head-audit-v2',scope='Sensor logits and recorded physical terminal state; no new closed-loop trajectories or PPO',runs=[],qualification=None,hardware_started=False)
for run,cp in [('diagnosis012_final_circular_fixed','final_pair_circular_corrected002/model_migrated.pt'),('pilot006_final_pairs_eval','pilot006_final_pairs_side/model_final.pt')]:
 p=base/run/'evaluation'; z=np.load(p/'trajectory.npz'); rep=json.loads((p/'report.json').read_text()); model,info=load_mean(base/cp); rows=[]
 for i,l in enumerate(rep['lanes'][:18]):
  mask=(z['time'][:,i]>=6.)&z['alive'][:,i]&~z['passive'][:,i]
  if not mask.any(): continue
  obs=torch.tensor(z['observations'][mask,i]); frame=torch.cat([obs[:,:14],torch.zeros_like(obs[:,14:16]),obs[:,16:]],-1)
  with torch.no_grad():
   raw=canonical_recovery_logits(model.recovery,frame,model.feature_scale);o=model(obs);fallen=tilt_gate(obs,model.stance_aware_recovery,model.recovery_shape_aware);entry=handoff_entry_gate(obs)
  pair=obs[:,10:14].numpy().reshape(-1,2,2); common=np.arctan2(np.sin(pair).sum(-1),np.cos(pair).sum(-1)); a=o[:,:4].numpy().reshape(-1,2,2)*.25; target_common=np.arctan2(np.sin(a).sum(-1),np.cos(a).sum(-1)); dif=np.arctan2(np.sin(target_common-common),np.cos(target_common-common))
  desiredraw=raw[:,:4].numpy().reshape(-1,2,2)
  rows.append(dict(lane=i,name=l['name'],actual_family=l['post_settle_family'],recovered=l['recovered'],followup=l['followup_success'],height_mean_m=float(z['height'][mask,i].mean()),body_tilt_mean_deg=float(np.rad2deg(np.arccos(np.clip(-z['gravity'][mask,i,2],-1,1))).mean()),common_sin_mean=np.sin(common).mean(0).tolist(),common_cos_mean=np.cos(common).mean(0).tolist(),target_common_error_mean_rad=dif.mean(0).tolist(),raw_common_mean=desiredraw.mean(-1).mean(0).tolist(),raw_knee_difference_mean=(desiredraw[:,:,1]-desiredraw[:,:,0]).mean(0).tolist(),fallen_gate_mean=float(fallen.mean()),entry_gate_mean=float(entry.mean()),wheel_clearance_mean_m=z['wheel_clearance'][mask,i].mean(0).tolist(),wheel_speed_mean_rad_s=z['motor_velocity'][mask,i][:,[2,5]].mean(0).tolist()))
 result['runs'].append(dict(run=run,checkpoint=cp,trajectory_sha256=hashlib.sha256((p/'trajectory.npz').read_bytes()).hexdigest(),rows=rows))
p=Path('reports/robust_policy_repair_20261024/recovery_tail_state_audit002.json');p.write_text(json.dumps(result,indent=2)+'\n')
for r in result['runs']:
 print(r['run'])
 for x in r['rows']:
  if not x['followup']: print({k:x[k] for k in ['lane','name','actual_family','height_mean_m','body_tilt_mean_deg','raw_common_mean','target_common_error_mean_rad','common_cos_mean','fallen_gate_mean','entry_gate_mean']})

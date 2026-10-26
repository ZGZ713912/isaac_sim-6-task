"""Read-only entry-head endpoint changes on the same recorded failure states."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path[:0]=['src','scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.robust_getup import tilt_gate
from wheeled_algo.getup_symmetry import canonical_recovery_logits

torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=Path('runs/robust_policy_repair_20261024');source=b/'entry_optimizer_support0_source001/model_migrated.pt';old,oi=load_mean(source);new,ni=load_mean(a.checkpoint);run=b/'diagnosis012_final_arithmetic_nominal/robust';rep=json.loads((run/'report.json').read_text());z=np.load(run/'trajectory.npz');rows=[]
for lane in rep['lanes']:
 if not lane.get('recovery') or lane['recovered']:continue
 i=lane['lane'];valid=z['alive'][:,i]&~z['passive'][:,i]&(z['time'][:,i]>=11.)
 if not valid.any():continue
 obs=torch.tensor(z['observations'][valid,i]);frame=torch.cat((obs[:,:14],torch.zeros_like(obs[:,14:16]),obs[:,16:]),-1)
 with torch.inference_mode():
  x,y=old(obs),new(obs);entry=handoff_entry_gate(obs);fallen=tilt_gate(obs,old.stance_aware_recovery,old.recovery_shape_aware)
  ro=canonical_recovery_logits(old.entry_recovery,frame,old.feature_scale);rn=canonical_recovery_logits(new.entry_recovery,frame,new.feature_scale)
  common=lambda r:r[:,:4].unflatten(-1,(2,2)).mean(-1)
  delta=torch.atan2(((y[:,:4]-x[:,:4])*.25).sin(),((y[:,:4]-x[:,:4])*.25).cos())
 rows.append(dict(name=lane['name'],actual_family=lane.get('post_settle_family'),recorded_height_mean=float(z['height'][valid,i].mean()),recorded_wheel_clearance_mean=np.mean(z['wheel_clearance'][valid,i],axis=0).tolist(),entry_gate_mean=float(entry.mean()),fallen_gate_mean=float(fallen.mean()),old_common_logit_mean=common(ro).mean(0).tolist(),new_common_logit_mean=common(rn).mean(0).tolist(),physical_motor_target_delta_mean_rad=delta.mean(0).tolist(),physical_motor_target_delta_max_rad=float(delta.abs().max()),wheel_action_delta_mean=(y[:,4:]-x[:,4:]).mean(0).tolist(),frames=int(valid.sum())))
out=dict(schema='entry-endpoint-failed-tail-replay-v2',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoint=str(a.checkpoint),sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),source_clock=oi['completed_updates'],new_clock=ni['completed_updates'],trace=str(run/'trajectory.npz'),trace_sha256=hashlib.sha256((run/'trajectory.npz').read_bytes()).hexdigest(),rows=rows,scope='Actor changes at identical recorded failed states; no new closed-loop or on-policy likelihood evidence.',new_PPO=0,qualification=None,hardware_started=False);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

"""Observed post-fall action sensitivity; no optimizer and no causal rollout claim."""
from pathlib import Path
import sys,json,time
sys.path[:0]=['src','scripts']
import numpy as np,torch
from export_robust_getup import load_mean,digest
from train_robust_getup import write_json
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.robust_getup import tilt_gate,quiet_gate
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024');torch.set_num_threads(1)
cp=B/'motion_damping080002/model_probe.pt';mean,_=load_mean(cp);mean.requires_grad_(True)
p=B/'diagnosis026_motion_damping080_nominal_pair/robust';report=json.loads((p/'report.json').read_text());t=np.load(p/'trajectory.npz')
rows=[]
for lane in report['lanes']:
 if not lane.get('in_motion_fall'):continue
 n=lane['lane'];live=t['alive'][:,n];fallen=t['in_motion_fall'][:,n]&live
 indices=np.flatnonzero(fallen)
 if not len(indices):continue
 start=float(t['time'][indices[0],n]);last=indices[t['time'][indices,n] <= start+.100001]
 obs=torch.from_numpy(t['observations'][last,n].copy())
 a=mean(obs);objective=(a*torch.tensor([1.,1.,1.,1.,.1,.1])).sum()
 groups={name:list(getattr(mean,name).parameters()) for name in ['entry_recovery','recovery','quiet','quiet_low','quiet_high']}
 norms={}
 for name,params in groups.items():
  gradients=torch.autograd.grad(objective,params,retain_graph=True,allow_unused=True)
  norms[name]=float(torch.stack([g.square().sum() for g in gradients if g is not None]).sum().sqrt()) if any(g is not None for g in gradients) else 0.
 entry=handoff_entry_gate(obs).detach();full=tilt_gate(obs,True,True).detach();quiet=quiet_gate(obs).detach()
 rows.append(dict(lane=n,name=lane['name'],first_observed_fall_s=start,termination_s=lane['termination_time_s'],
  measured_contact_impulse_ns=lane['max_contact_impulse_ns'],termination_causes=lane['termination_causes'],frames=len(last),
  window_seconds=[float(t['time'][last[0],n]),float(t['time'][last[-1],n])],entry_gate_peak=float(entry.max()),fallen_gate_mean=float(full.mean()),
  quiet_gate_peak=float(quiet.max()),output_sensitivity_parameter_gradient_L2=norms))
checks=dict(moving_fall_examples_present=len(rows)>=2,entry_paths_exact_zero=all(x['entry_gate_peak']==0. and x['output_sensitivity_parameter_gradient_L2']['entry_recovery']==0. for x in rows),
 fallen_paths_nonzero=all(x['output_sensitivity_parameter_gradient_L2']['recovery']>0. for x in rows),quiet_paths_exact_zero=all(x['quiet_gate_peak']==0. for x in rows))
out=dict(schema='actual-moving-fall-actor-scope-v1',checks=checks,checkpoint=str(cp),checkpoint_sha256=digest(cp),trace=str(p/'trajectory.npz'),trace_sha256=digest(p/'trajectory.npz'),rows=rows,
 objective='sum four leg actions plus.1*sum wheel actions; output sensitivity only, not PPO/GAE gradient or native closed-loop improvement',
 interpretation='Entry-only optimization cannot directly change these observed first100ms actions. Recovery-head sensitivity is present; earlier entry learning can still alter arrival states.',
 new_PPO=0,optimizer_calls=0,actual_experiment_consumed=2630,qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'moving_fall_actor_scope001.json',out);print(json.dumps(out,indent=2));assert all(checks.values()),checks

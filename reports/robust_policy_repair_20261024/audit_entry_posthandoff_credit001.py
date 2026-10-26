"""Check direct entry-weight credit on actual post-handoff oscillatory inputs."""
import json,time
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
import numpy as np,torch
from export_robust_getup import load_mean
from train_robust_getup import digest,write_json
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024');torch.set_num_threads(1)
cp=B/'motion_damping080002/model_probe.pt';mean,_=load_mean(cp);mean.requires_grad_(False);mean.entry_recovery.requires_grad_(True)
params=list(mean.entry_recovery.parameters());path=B/'diagnosis025_motion_damping080_fixed/evaluation/trajectory.npz';z=np.load(path);rep=json.loads((path.parent/'report.json').read_text());rows=[]
weights=torch.tensor([1.,2.,3.,5.,7.,11.])
def norm(values):
 out=mean(torch.from_numpy(values.copy()));grad=torch.autograd.grad((out*weights).sum(),params,allow_unused=True)
 return float(torch.sqrt(sum(g.square().sum() for g in grad if g is not None)))
for name in ['real_start_3__clean','real_start_5__pitch_plus2','real_start_6__pitch_plus2']:
 i=next(r['lane'] for r in rep['lanes'] if r['name']==name);x=z['observations'][:,i];e=handoff_entry_gate(torch.from_numpy(x.copy())).detach().ravel().numpy()
 mask=z['alive'][:,i]&(z['time'][:,i]>=6.)&(e==0.)&(np.linalg.norm(z['omega'][:,i,:2],axis=-1)>.7)
 ids=np.flatnonzero(mask);chosen=ids[np.linspace(0,len(ids)-1,min(48,len(ids))).astype(int)]
 assert len(chosen)>0
 rows.append(dict(name=name,eligible_frames=len(ids),audited_frames=len(chosen),entry_weight_gradient_l2=norm(x[chosen])))
x=z['observations'].reshape(-1,35);e=handoff_entry_gate(torch.from_numpy(x.copy())).detach().ravel().numpy();ids=np.flatnonzero(e>.8);chosen=ids[np.linspace(0,len(ids)-1,48).astype(int)];positive=norm(x[chosen])
checks=dict(all_posthandoff_gradients_exact_zero=all(r['entry_weight_gradient_l2']==0. for r in rows),positive_entry_contrast_nonzero=positive>0.)
o=dict(schema='recorded-entry-direct-credit-scope-v1',checks=checks,rows=rows,active_entry_contrast_gradient_l2=positive,
 checkpoint=str(cp),checkpoint_sha256=digest(cp),trace=str(path),trace_sha256=digest(path),
 interpretation='At these observed gate-zero inputs, entry weights cannot directly change corrective actions. Entry PPO can still affect arrival states and therefore indirectly change later trajectories. No claim that the full closed-loop gradient is zero.',
 actual_optimizer_steps=0,new_PPO=0,actual_experiment_consumed=2630,qualification=None,created_unix=time.time())
write_json(R/'entry_posthandoff_credit_scope001.json',o);print(json.dumps(o,indent=2));assert all(checks.values()),checks

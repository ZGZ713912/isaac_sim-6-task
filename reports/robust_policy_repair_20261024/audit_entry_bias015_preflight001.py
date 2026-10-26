"""Separate the native regression's fallen and upright-entry head changes."""
import sys,json,hashlib
from pathlib import Path
sys.path[:0]=['src','scripts']
import numpy as np,torch
from export_robust_getup import load_mean
from wheeled_algo.gap_entry_handoff import handoff_entry_gate

torch.set_num_threads(1);b=Path('runs/robust_policy_repair_20261024');original=b/'final_pair_circular_corrected002/model_migrated.pt';reference=b/'recovery_common_bias015_probe001/model_probe.pt';candidate=b/'entry_bias015_preserved001/model_migrated.pt';a,ai=load_mean(original);ref,ri=load_mean(reference);c,ci=load_mean(candidate);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();src=torch.load(original,map_location='cpu',weights_only=False);dst=torch.load(candidate,map_location='cpu',weights_only=False)
def exact(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b
checks=dict(original_actor_tensors_exact=all(torch.equal(v,dst['actor_state_dict'][k]) for k,v in src['actor_state_dict'].items()),critic_exact=exact(src['critic_state_dict'],dst['critic_state_dict']),optimizer_exact=exact(src['optimizer_state_dict'],dst['optimizer_state_dict']),clock_exact=ci['completed_updates']==ai['completed_updates'],entry_all_frozen=not any(p.requires_grad for p in c.entry_recovery.parameters()))
for k in ['domain_rng','environment_rng','torch_rng','cuda_rng']:checks[k+'_exact']=exact(src['infos'][k],dst['infos'][k])
rows=[]
with torch.inference_mode():
 for run in ['diagnosis012_final_circular_fixed','diagnosis012_final_arithmetic_nominal/robust','diagnosis012_final_arithmetic_rmcs']:
  d=b/run
  if (d/'evaluation/trajectory.npz').exists():d=d/'evaluation'
  p=d/'trajectory.npz';z=np.load(p);flat=z['observations'].reshape(-1,35);outside=0;wheel=0;maxfull=0.;inside=0
  for obs in torch.tensor(flat).split(2048):
   av,rv,cv=a(obs),ref(obs),c(obs);g=handoff_entry_gate(obs).ravel();outside+=int((av[g==0]!=cv[g==0]).any(-1).sum());wheel+=int((av[:,4:]!=cv[:,4:]).any(-1).sum());full=g==1
   if full.any():
    diff=(rv[full,:4]-cv[full,:4])*.25;physical=torch.atan2(diff.sin(),diff.cos()).abs().max().item();maxfull=max(maxfull,physical);inside+=int(full.sum())
  rows.append(dict(run=run,frames=len(flat),trajectory_sha256=sha(p),entry_zero_different_frames=outside,wheels_different_frames=wheel,entry_full_frames=inside,entry_full_physical_target_max_delta_rad=maxfull))
  assert outside==0 and wheel==0 and maxfull<=1e-6,rows[-1]
assert all(checks.values()),checks
out=dict(schema='entry-bias015-head-separation-preflight-v1',checkpoint=str(candidate),sha256=sha(candidate),source_sha256=sha(original),entry_reference_sha256=sha(reference),checks=checks,replays=rows,scope='Only learned upright-entry weights differ. Entry0 original bit exact; entry1 physically matches biased shared mean within float32 arithmetic.',new_PPO=0,actual_experiment_consumed=2587,qualification=None,hardware_started=False);Path('reports/robust_policy_repair_20261024/entry_bias015_preflight001.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

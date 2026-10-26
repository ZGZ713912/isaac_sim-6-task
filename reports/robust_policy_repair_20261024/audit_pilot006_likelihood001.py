"""Read-only deterministic endpoint replay; this is not on-policy update KL."""
import sys, json, hashlib
from pathlib import Path
sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from wheeled_algo.gap_recovery_exploration import GapRecoveryNoiseMean
from wheeled_algo.robust_getup import tilt_gate
from wheeled_algo.gap_entry_handoff import handoff_entry_gate

torch.set_num_threads(1)
base = Path('runs/robust_policy_repair_20261024')
source = base/'final_pair_circular_corrected002/model_migrated.pt'
final = base/'pilot006_final_pairs_side/model_final.pt'
a, ai = load_mean(source)
b, bi = load_mean(final)
state = torch.load(source, map_location='cpu', weights_only=False)
s = ai['settings']
dist = GapRecoveryNoiseMean(a, s['quiet_exploration_fraction'],s.get('quiet_wheel_exploration_fraction'),s.get('gap_recovery_leg_exploration_fraction',1.))
with torch.no_grad(): dist.log_std.copy_(state['actor_state_dict']['mlp.log_std'])

def stats(x):
 x = np.asarray(x)
 return {'count':int(x.size),'mean':float(x.mean()) if x.size else None,'p99':float(np.quantile(x,.99)) if x.size else None,'max':float(x.max()) if x.size else None}

receipt = dict(schema='pilot006-endpoint-replay-likelihood-audit-v1',source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),candidate_checkpoint=str(final),candidate_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),comparison='32 update endpoints on fixed evaluation observations, not PPO minibatch/on-policy KL',runs=[],qualification=None,hardware_started=False)
for relative in ['diagnosis012_final_circular_fixed/evaluation/trajectory.npz','diagnosis012_final_arithmetic_nominal/robust/trajectory.npz']:
 path=base/relative
 if not path.exists(): continue
 z=np.load(path); frames=z['observations']; flat=frames.reshape(-1,35)
 deltas=[]; physical=[]; kls=[]; gates=[]
 with torch.inference_mode():
  for o in torch.tensor(flat).split(2048):
   da=dist(o); ma=da[:,0]; mb=b(o); delta=mb-ma
   std=da[:,1].clamp(np.log(.003),np.log(.8)).exp()
   physical_delta=torch.atan2((delta[:,:4]*.25).sin(),(delta[:,:4]*.25).cos())
   deltas.append(delta.numpy());physical.append(physical_delta.numpy()); kls.append((.5*(delta/std).square().sum(-1)).numpy());gates.append(torch.maximum(tilt_gate(o,a.stance_aware_recovery,a.recovery_shape_aware),handoff_entry_gate(o)).numpy().ravel())
 d=np.concatenate(deltas); p=np.concatenate(physical); kl=np.concatenate(kls); g=np.concatenate(gates)
 receipt['runs'].append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=int(len(d)),raw_action_abs_delta=stats(abs(d)),physical_leg_abs_delta_rad=stats(abs(p)),raw_gaussian_KL=stats(kl),raw_leg_delta_over_pi_frames=int((abs(d[:,:4]*.25)>np.pi).any(-1).sum()),gate_active_frames=int((g>0).sum()),active_raw_gaussian_KL=stats(kl[g>0]),quiet_raw_gaussian_KL=stats(kl[g==0])))
out=Path('reports/robust_policy_repair_20261024/pilot006_likelihood_audit001.json');out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

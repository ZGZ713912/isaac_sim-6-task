"""Prove the gradient subspace and neutral phase guard before any admission."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path[:0]=['src','scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest,write_json
from wheeled_algo.periodic_entry_actor import periodic_canonical_entry_logits
from entry_common_guard001 import attach_common_gradient_mask,PhaseGuard

torch.set_num_threads(1)
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
cp=B/'function_root_entry002/model_initial.pt'
mean,info=load_mean(cp);original=deepcopy(mean.state_dict())
mean.entry_recovery.requires_grad_(True)
handles=attach_common_gradient_mask(mean.entry_recovery)
frames=np.load(B/'diagnosis024_entry_phase_fit_fixed/evaluation/trajectory.npz')['observations'].reshape(-1,35)
x=torch.from_numpy(frames[::max(1,len(frames)//512)][:512].copy())
frame=torch.cat((x[:,:14],torch.zeros_like(x[:,14:16]),x[:,16:]),-1)
raw=periodic_canonical_entry_logits(mean.entry_recovery,frame,mean.feature_scale)
# A deliberately asymmetric probe exercises common, knee, wheel and basis
# gradients. No parameter update or optimizer step is performed.
weights=raw.new_tensor([.2,-.7,.4,.9,-.6,.3])
(raw*weights).square().mean().backward()
params=dict(mean.entry_recovery.named_parameters())
checks=dict(feature_gradients_zero=all(torch.count_nonzero(p.grad)==0 for n,p in params.items() if not n.startswith('4.')),
    last_gradients_present=all(float(params[n].grad.abs().max())>0 for n in ('4.weight','4.bias')),
    common_rows_equal=all(torch.equal(params[n].grad[0],params[n].grad[1]) and torch.equal(params[n].grad[2],params[n].grad[3]) for n in ('4.weight','4.bias')),
    wheel_gradients_zero=all(torch.count_nonzero(params[n].grad[4:])==0 for n in ('4.weight','4.bias')),
    all_weights_unchanged=all(torch.equal(v,mean.state_dict()[k]) for k,v in original.items()))
for p in mean.parameters():
    p.grad=None
guard=PhaseGuard(mean,R/'entry_phase_path_dataset001.npz')
metrics,neutral=guard.measure();checks.update({'neutral_'+k:v for k,v in neutral.items()})
out=dict(schema='common-gradient-and-phase-guard-preflight-v1',checks={k:bool(v) for k,v in checks.items()},
    checkpoint=str(cp),checkpoint_sha256=digest(cp),source_clock=info['completed_updates'],
    gradient_requires_grad_parameters=sum(p.numel() for p in params.values()),effective_common_parameters=258,
    neutral_guard=metrics,phase_frames=len(guard.observations),
    mask_source_sha256=digest(R/'entry_common_guard001.py'),
    hypothesis='Keep the original feature basis and knee/wheel mappings while optimizing only the existing two common rows. Stop before another rollout if a completed update violates the existing phase margins.',
    status='engineering_preflight_only_no_new_PPO_admitted',new_PPO=0,optimizer_steps=0,
    qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'entry_common_guard_preflight001.json',out)
print(json.dumps(out,indent=2));assert all(checks.values()),checks

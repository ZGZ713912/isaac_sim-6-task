"""Audit native GAE/value error and local sampled-action credit, no updates."""
import json
from pathlib import Path
import sys
import time
sys.path[:0]=['src','scripts']
import numpy as np
import torch
from train_robust_getup import digest,write_json
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.robust_getup import quiet_gate,tilt_gate

torch.set_num_threads(1)
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
run=B/'credit_rollout001';receipt=json.loads((run/'rollout_diagnosis.json').read_text())
assert json.loads((R/'credit_rollout_supervisor001.json').read_text())['status']=='completed_diagnosis'
data=run/'native_rollout_and_GAE.npz';assert digest(data)==receipt['data_sha256']
z=np.load(data);learning=z['learning_mask'];bootstrap=z['bootstrap_mask'];values=z['values'][...,0]
rewards=z['rewards'][...,0];dones=z['dones'][...,0].astype(bool)
v=np.where(learning,values,0.);returns=np.zeros_like(values);adv=np.zeros(values.shape[1],dtype=np.float32)
for t in reversed(range(len(v))):
    next_valid=bootstrap[t].copy()
    if t+1<len(v):next_valid&=learning[t+1];next_v=v[t+1]
    else:next_v=z['last_critic_values'].reshape(-1)
    continues=learning[t]&next_valid&~dones[t]
    adv=np.where(learning[t],rewards[t]-v[t]+np.where(continues,.998*next_v+.998*.99*adv,0.),0.)
    returns[t]=adv+v[t]
raw_adv=returns-v
return_error=float(np.abs(returns-z['returns'][...,0]).max())
obs=z['observations__policy'];x=torch.from_numpy(obs.reshape(-1,35).copy())
entry=handoff_entry_gate(x).ravel().numpy().reshape(learning.shape)
quiet=quiet_gate(x).ravel().numpy().reshape(learning.shape)
fallen=tilt_gate(x,True,True).ravel().numpy().reshape(learning.shape)
groups=dict(all=learning,full_entry=learning&(entry>.99),partial_entry=learning&(entry>0)&(entry<=.99),
    fallen=learning&(fallen>.99)&(entry==0),quiet=learning&(quiet>.99)&(entry==0))
rows=[]
mu=z['distribution_param_0'];std=z['distribution_param_1'];actions=z['actions']
assert std.min()>0
score=(actions-mu)/(std**2)*raw_adv[...,None]
q=obs[...,10:14].reshape(*learning.shape,2,2)
phase=np.arctan2(np.sin(q).sum(-1),np.cos(q).sum(-1))
direction=-np.sign(np.sin(phase))
common_score=score[...,:4].reshape(*learning.shape,2,2).sum(-1)
for name,mask in groups.items():
    if not mask.any():continue
    a,b=values[mask],returns[mask];error=b-a
    far=mask[...,None]&(np.abs(phase)>=.75)
    rows.append(dict(group=name,transitions=int(mask.sum()),predicted_value_mean=float(a.mean()),
        target_return_mean=float(b.mean()),target_return_std=float(b.std()),
        value_RMSE=float(np.sqrt(np.mean(error**2))),value_bias=float((a-b).mean()),
        explained_variance=float(1.-np.var(error)/max(np.var(b),1e-20)),
        advantage_mean=float(raw_adv[mask].mean()),advantage_std=float(raw_adv[mask].std()),
        local_wheel_down_common_score=float((common_score*direction)[far].mean()) if far.any() else None))
config=json.loads((run/'agent_config.json').read_text());assert config['critic']['activation']=='elu'
model=torch.nn.Sequential(torch.nn.Linear(83,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),
    torch.nn.Linear(128,64),torch.nn.ELU(),torch.nn.Linear(64,1))
source=Path(json.loads((R/'credit_rollout_plan001.json').read_text())['source'])
state=torch.load(source,map_location='cpu',weights_only=False)
model.load_state_dict({k.removeprefix('mlp.'):v for k,v in state['critic_state_dict'].items()})
subset=z['observations__critic'].reshape(-1,83)[::31].copy()
with torch.inference_mode():prediction=model(torch.from_numpy(subset)).numpy().reshape(-1)
critic_cpu_error=float(np.abs(prediction-values.reshape(-1)[::31]).max())
prior=json.loads((B/'pilot009_periodic_motion_entry/first_rollout_hashes.json').read_text())['tensors']
mapping={'observations/'+k.removeprefix('observations__') if k.startswith('observations__') else k:v for k,v in receipt['tensors'].items()}
first_exact={k:mapping[k]==v for k,v in prior.items()}
checks=dict(native_GAE_reconstructed=return_error<=1e-4,source_critic_reproduced=critic_cpu_error<=1e-4,
    first_rollout_matches_prior=all(first_exact.values()),all_restore_and_no_PPO=all(json.loads((R/'credit_rollout_supervisor001.json').read_text())['checks'].values()))
out=dict(schema='native-rollout-credit-audit-v1',checks=checks,rows=rows,
    return_reconstruction_peak_error=return_error,source_critic_CPU_peak_error=critic_cpu_error,
    first_rollout_exact=first_exact,data_sha256=digest(data),source_sha256=digest(source),
    native_advantage_normalization_per_minibatch=receipt['normalize_advantage_per_mini_batch'],
    score_scope='Local Gaussian mean-shift score times raw native GAE, not the shared-weight PPO gradient or a causal advantage estimate.',
    limitations='GAE targets bootstrap from the source critic and are not independent Monte Carlo truth. Group errors and score alignment alone do not identify a reward or critic cause.',
    new_PPO=0,optimizer_steps=0,actual_experiment_consumed=2622,repair_optimizer_updates=242,
    qualification=None,created_unix=time.time())
write_json(R/'credit_rollout_audit001.json',out);print(json.dumps(out,indent=2));assert all(checks.values())

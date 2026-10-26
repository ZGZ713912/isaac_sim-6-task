"""Require trajectory identity and reconstruct the single native reward change."""
import json
from pathlib import Path
import sys
import time

sys.path[:0]=['src','scripts']
import numpy as np
import torch
from train_robust_getup import digest,write_json
from wheeled_tasks.chassis.quiet_standing import standing_position_cost

torch.set_num_threads(1)
R=Path('reports/robust_policy_repair_20261024')
B=Path('runs/robust_policy_repair_20261024')
plan=json.loads((R/'diagnosis_plan030.json').read_text())
queue=json.loads((R/'diagnosis_queue030.json').read_text())
assert queue['status']=='completed'
bad=[n for n,h in plan['source_sha256'].items() if digest(Path(n))!=h]
assert not bad,bad
paths=[B/('diagnosis030_quiet_reward_'+x) for x in ('control','tight')]
traces=[np.load(p/'evaluation/trajectory.npz') for p in paths]
a,b=traces
assert a.files==b.files
excluded={'native_reward','reward_density'}
equal={k:np.array_equal(a[k],b[k]) for k in a.files if k not in excluded}
checks=dict(frozen_inputs_exact=True,non_reward_trajectory_exact=all(equal.values()),
            reward_intervention_present=not np.array_equal(a['native_reward'],b['native_reward']))
assert a['local_position'].shape[1]==21
T,N=a['time'].shape
anchor=np.zeros((N,2),np.float32);active=np.zeros(N,bool);radius=np.zeros((T,N),np.float32)
for t in range(T):
    requested=~a['passive'][t] | a['enable_event'][t]
    start=requested & ~active
    anchor[start]=a['local_position'][t,start,:2]
    active=requested
    displacement=np.where(active[:,None],a['local_position'][t,:,:2]-anchor,0.)
    radius[t]=np.sqrt(np.sum(displacement**2,axis=-1))
r=torch.from_numpy(radius)
request=torch.from_numpy(~a['passive']|a['enable_event'])
expected_density=-6.*request*(torch.from_numpy(a['time'])>=.2)*(standing_position_cost(r,.02)-standing_position_cost(r,.05))
expected_reward=expected_density*.005*torch.from_numpy(~a['passive'])
density=torch.from_numpy(b['reward_density']-a['reward_density'])
reward=torch.from_numpy(b['native_reward']-a['native_reward'])
de=float((density-expected_density).abs().max());re=float((reward-expected_reward).abs().max())
checks.update(position_reward_density_reconstructed=de<=1e-4,
              native_reward_reconstructed=re<=1e-6,
              penalty_never_rewards_more=bool((reward<=1e-7).all()),
              intermediate_radius_has_credit=bool(((r>.02)&(r<.05)&(expected_reward<0)).any()))
row=[]
report=json.loads((paths[0]/'evaluation/report.json').read_text())
for lane in report['lanes']:
    i=lane['lane'];mask=torch.from_numpy(a['alive'][:,i]&~a['passive'][:,i])
    row.append(dict(name=lane['name'],max_radius_m=float(r[mask,i].max()),
        native_reward_sum_before=float(torch.from_numpy(a['native_reward'][:,i])[mask].sum()),
        native_reward_sum_after=float(torch.from_numpy(b['native_reward'][:,i])[mask].sum()),
        extra_position_penalty_sum=float(-expected_reward[mask,i].sum())))
out=dict(schema='quiet-reward-native-pair-audit-v1',checks=checks,
    traces=[dict(path=str(p/'evaluation/trajectory.npz'),sha256=digest(p/'evaluation/trajectory.npz')) for p in paths],
    non_reward_arrays_exact=equal,density_residual_peak=de,reward_residual_peak=re,
    tolerances=dict(density=1e-4,reward=1e-6),rows=row,
    inputs_exact_count=len(plan['source_sha256']),
    decision='bounded_training_pair_may_be_reviewed' if all(checks.values()) else 'reject_PPO_admission',
    new_PPO=0,actual_experiment_consumed=2605,qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'quiet_reward_native_audit001.json',out)
print(json.dumps({k:v for k,v in out.items() if k not in ('rows','non_reward_arrays_exact')},indent=2))
assert all(checks.values()),checks

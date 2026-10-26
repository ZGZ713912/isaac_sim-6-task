"""Read actual paid rewards versus entry behavior; no reward reconstruction."""
import json
from pathlib import Path
import sys
import time
sys.path[:0]=['src','scripts']
import numpy as np
import torch
from train_robust_getup import digest,write_json
from wheeled_algo.gap_entry_handoff import handoff_entry_gate

torch.set_num_threads(1)
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
run=B/'diagnosis025_motion_damping080_fixed';trace=run/'evaluation/trajectory.npz'
report=json.loads((run/'evaluation/report.json').read_text());z=np.load(trace)
rows=[]
for lane in report['lanes']:
    if not lane['behavior_case'].startswith('real_start'):continue
    i=lane['lane'];active=z['alive'][:,i]&~z['passive'][:,i]
    k=np.flatnonzero(active);times=z['time'][:,i];tail=active&(times>=times[k[-1]]-2.)
    raw=z['native_reward'][:,i];density=z['reward_density'][:,i];potential=z['reward_potential_progress'][:,i]
    powers=np.power(.998,np.arange(len(k)))
    observations=torch.from_numpy(z['observations'][:,i].copy())
    gate=handoff_entry_gate(observations).ravel().numpy()
    rows.append(dict(name=lane['name'],recovered=lane['recovered'],followup=lane['followup_success'],
        survived=lane['survived'],active_seconds=.005*len(k),
        discounted_native_return=float((raw[k]*powers).sum()),
        discounted_potential_progress=float((potential[k]*powers).sum()),
        actual_reward_sum=float(raw[k].sum()),tail_density_per_s=float(density[tail].mean()),
        tail_stable_fraction=float(z['stable'][tail,i].mean()),
        tail_entry_gate_mean=float(gate[tail].mean()),
        tail_true_wheel_clearance_peak_m=float(z['wheel_clearance'][tail,i].max()),
        tail_true_height_mean_m=float(z['height'][tail,i].mean())))
summary={}
for label,select in [('followup',lambda x:x['followup']),('no_followup',lambda x:not x['followup'])]:
    group=[x for x in rows if select(x)]
    summary[label]=dict(count=len(group),**{key:float(np.mean([x[key] for x in group])) for key in
        ['discounted_native_return','discounted_potential_progress','tail_density_per_s','tail_stable_fraction','tail_true_wheel_clearance_peak_m']})
out=dict(schema='native-entry-reward-outcome-audit-v1',trace=str(trace),sha256=digest(trace),rows=rows,summary=summary,
    gamma=.998,source_clock=1501,new_PPO=0,actual_experiment_consumed=2622,
    scope='Actual mean-policy fixed development returns with no critic bootstrap; conditional outcomes and initial states differ. Does not estimate the native sampled PPO advantage or prove a reward/critic cause.',
    qualification=None,created_unix=time.time())
write_json(R/'entry_reward_outcome001.json',out);print(json.dumps(out,indent=2))

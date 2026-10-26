"""Audit a completed bounded quiet pilot from its sealed native tensors."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path[:0]=['src','scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest,write_json
from wheeled_algo.periodic_entry_actor import exact_state_equal
from wheeled_algo.robust_getup import quiet_gate

parser=argparse.ArgumentParser()
parser.add_argument('--plan',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
torch.set_num_threads(1)
plan=json.loads(args.plan.read_text());source=Path(plan['source_checkpoint'])
run=Path(plan['training_output']);progress=json.loads((run/'progress.json').read_text())
assert progress['status']=='completed'
target=run/'model_final.pt'
a,b=[torch.load(p,map_location='cpu',weights_only=False) for p in (source,target)]
allowed={f'mlp.mean.quiet_velocity.{h}.weight' for h in ('low','mid','high')}
changed=[k for k,v in a['actor_state_dict'].items() if not torch.equal(v,b['actor_state_dict'][k])]
checks=dict(exactly_three_feedback_tensors_changed=set(changed)==allowed,
    all_other_actor_and_buffers_exact=all(torch.equal(v,b['actor_state_dict'][k])
        for k,v in a['actor_state_dict'].items() if k not in allowed),
    completed_updates=int(b['infos']['completed_updates'])==1509,
    eight_updates_charged=progress['new_optimizer_updates']==progress['new_optimizer_attempts']==8,
    only24_parameters=progress['trainable_actor_parameters']==24,
    frozen_actor_runtime_assertions=progress.get('frozen_motion_recovery_unchanged') is True
        and progress.get('frozen_quiet_exploration_unchanged') is True,
    source_hash=digest(source)==plan['source_checkpoint_sha256'],
    reward_setting_exact=b['infos']['settings']['quiet_sway_m']==plan['standing_sway_m'])
old=a['optimizer_state_dict'];new=b['optimizer_state_dict']
checks['old_actor_Adam_exact']=all(exact_state_equal(old['state'][i],new['state'][i]) for i in range(25))
checks['three_fresh_feedback_Adam_steps96']=all(i not in old['state'] and float(new['state'][i]['step'])==96.
    and all(torch.isfinite(new['state'][i][k]).all() for k in ('exp_avg','exp_avg_sq')) for i in (25,26,27))
checks['critic_actually_changed']=not exact_state_equal(a['critic_state_dict'],b['critic_state_dict'])
checks['finite_actor_critic']=all(bool(torch.isfinite(v).all()) for key in ('actor_state_dict','critic_state_dict') for v in b[key].values())
before,_=load_mean(source);after,_=load_mean(target)
trace=Path('runs/robust_policy_repair_20261024/diagnosis013_low_diagonal_fixed/evaluation/trajectory.npz')
frames=np.load(trace)['observations'].reshape(-1,35)
outside_count=outside_different=0;peak=0.
with torch.inference_mode():
    for values in np.array_split(frames,(len(frames)+511)//512):
        obs=torch.from_numpy(values.copy());x,y=before(obs),after(obs)
        outside=quiet_gate(obs).ravel()==0.
        outside_count+=int(outside.sum());outside_different+=int((x[outside]!=y[outside]).any(-1).sum())
        peak=max(peak,float((x-y).abs().max()))
checks['outside_quiet_function_exact']=outside_different==0
out=dict(schema='native-quiet24-endpoint-audit-v1',checks=checks,source=str(source),source_sha256=digest(source),
    checkpoint=str(target),checkpoint_sha256=digest(target),changed_actor_tensors=changed,
    feedback_weight_change_peak=max(float((a['actor_state_dict'][k]-b['actor_state_dict'][k]).abs().max()) for k in allowed),
    recorded_function_probe=dict(trace=str(trace),sha256=digest(trace),frames=len(frames),
        outside_count=outside_count,outside_different=outside_different,all_output_change_peak=peak),
    new_PPO_completed=8,source_clock=1501,branch_clock=1509,qualification=None,
    hardware_started=False,created_unix=time.time())
write_json(args.output,out)
print(json.dumps(out,indent=2));assert all(checks.values()),checks

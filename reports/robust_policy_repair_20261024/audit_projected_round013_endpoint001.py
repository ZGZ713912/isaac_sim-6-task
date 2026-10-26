"""Audit the bounded projected-Adam endpoint and its charged lineage."""
import json,time
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest,write_json
from wheeled_algo.periodic_entry_actor import exact_state_equal
from wheeled_algo.gap_entry_handoff import handoff_entry_gate

torch.set_num_threads(1)
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
plan=json.loads((R/'plan013_projected_entry001.json').read_text());run=Path(plan['training_output'])
p=json.loads((run/'progress.json').read_text()); assert p['status'] in ('completed','stopped')
n=p['new_optimizer_updates']; assert n==p['new_optimizer_attempts'] and 1<=n<=8
source=Path(plan['source_checkpoint']);target=run/('model_final.pt' if p['status']=='completed' else 'model_stopped.pt')
a,b=[torch.load(x,map_location='cpu',weights_only=False) for x in (source,target)]
guard=json.loads((run/'entry_phase_guard.json').read_text());info=b['infos']['entry_projected_optimizer']
allowed={'mlp.mean.entry_recovery.4.weight','mlp.mean.entry_recovery.4.bias'}
changed=[k for k,v in a['actor_state_dict'].items() if not torch.equal(v,b['actor_state_dict'][k])]
old,new=a['optimizer_state_dict'],b['optimizer_state_dict']
checks=dict(
    endpoint_clock=b['infos']['completed_updates']==1501+n,
    declared_budget=n<=8 and (p['status']!='completed' or n==8),
    attempted_charged_exact=p['new_optimizer_attempts']==n,
    all_projected_steps_recorded=len(guard['updates'])==info['completed_projected_PPO_updates']==n,
    projection_metadata_matches=info['updates']==guard['updates'],
    projection_version=info['schema']=='fixed-basis-common-projected-Adam-v1',
    projection_source_exact=info['source_sha256']==digest(source),
    wrapper_bound=info['wrapper_sha256']==digest(R/'run_projected_entry_pilot001.py'),
    projector_bound=info['projector_sha256']==digest(R/'common_phase_projector001.py'),
    only_final_common_tensors_changed=set(changed)==allowed,
    non_entry_and_basis_exact=all(torch.equal(v,b['actor_state_dict'][k]) for k,v in a['actor_state_dict'].items() if k not in allowed),
    wheel_rows_exact=all(torch.equal(a['actor_state_dict'][k][4:],b['actor_state_dict'][k][4:]) for k in allowed),
    all_old_actor_Adam_exact=all(exact_state_equal(old['state'].get(i),new['state'].get(i)) for i in range(28)),
    all_six_entry_Adam_steps=all(i not in old['state'] and float(new['state'][i]['step'])==12.*n for i in range(28,34)),
    basis_Adam_moments_zero=all(torch.count_nonzero(new['state'][i][k])==0 for i in range(28,32) for k in ('exp_avg','exp_avg_sq')),
    critic_changed=not exact_state_equal(a['critic_state_dict'],b['critic_state_dict']),
    frozen_inputs_exact=all(digest(Path(x))==h for x,h in plan['source_sha256'].items()),
    initial_restore_all=all(json.loads((Path(plan['supervisor_output'])/'initial_restore.json').read_text())['checks'].values()),
    first_native_rollout_exact=all(json.loads((R/'projected_entry_initial_readback001.json').read_text())['first_native_rollout_comparison'].values()))
before,_=load_mean(source);after,_=load_mean(target);checks['strict_V3_decode']=True
trace=B/'diagnosis025_motion_damping080_fixed/evaluation/trajectory.npz'
frames=np.load(trace)['observations'].reshape(-1,35)
outside_count=outside_different=wheel_different=0
with torch.inference_mode():
    for values in np.array_split(frames,(len(frames)+511)//512):
        x=torch.from_numpy(values.copy());v,w=before(x),after(x);outside=handoff_entry_gate(x).ravel()==0
        outside_count+=int(outside.sum());outside_different+=int((v[outside]!=w[outside]).any(-1).sum())
        wheel_different+=int((v[:,4:]!=w[:,4:]).any(-1).sum())
checks.update(outside_entry_function_exact=outside_different==0,wheel_function_exact=wheel_different==0)
checks={k:bool(v) for k,v in checks.items()}
phases=[dict(clock=row['learning_clock'],checks=row['checks'],metrics=row['metrics']) for row in guard['updates']]
out=dict(schema='projected-Adam-native-endpoint-v1',checks=checks,checkpoint=str(target),checkpoint_sha256=digest(target),
    changed_actor_tensors=changed,source=str(source),source_sha256=digest(source),source_clock=1501,branch_clock=1501+n,
    real_PPO_attempts=n,real_PPO_completed=n,actual_experiment_consumed=2622+n,repair_optimizer_updates=242+n,
    phase_checks=phases,phase_guard_stop=guard['stop_requested'],
    all_development_phase_checks_pass=all(all(row['checks'].values()) for row in guard['updates']),
    frozen_inputs=len(plan['source_sha256']),
    function_probe=dict(trace=str(trace),sha256=digest(trace),frames=len(frames),outside_count=outside_count,
        outside_different=outside_different,wheel_different=wheel_different),
    algorithm='Projected Adam: native gradients and moments plus explicit post-update constrained coefficient projection.',
    interpretation='Engineering audit only; this receipt does not qualify physical behavior or authorize extension.',
    qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'projected_round013_endpoint001.json',out);print(json.dumps(out,indent=2));assert all(checks.values()),checks

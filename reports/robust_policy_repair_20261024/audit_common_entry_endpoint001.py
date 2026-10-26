"""Seal the real common-only phase-guard stop, without a native behavior claim."""
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
from wheeled_algo.gap_entry_handoff import handoff_entry_gate

torch.set_num_threads(1)
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
plan=json.loads((R/'plan012_common_entry002.json').read_text())
run=Path(plan['training_output']);source=Path(plan['source_checkpoint']);target=run/'model_stopped.pt'
report=json.loads((run/'progress.json').read_text());guard=json.loads((run/'entry_phase_guard.json').read_text())
a,b=[torch.load(p,map_location='cpu',weights_only=False) for p in (source,target)]
allowed={'mlp.mean.entry_recovery.4.weight','mlp.mean.entry_recovery.4.bias'}
changed=[k for k,v in a['actor_state_dict'].items() if not torch.equal(v,b['actor_state_dict'][k])]
old,new=a['optimizer_state_dict'],b['optimizer_state_dict']
checks=dict(one_completed_attempt=report['new_optimizer_attempts']==report['new_optimizer_updates']==1,
    phase_guard_sealed=report['status']=='stopped' and guard['stop_requested'],
    clock1502=b['infos']['completed_updates']==1502,
    only_final_common_tensors_changed=set(changed)==allowed,
    non_entry_and_basis_exact=all(torch.equal(v,b['actor_state_dict'][k]) for k,v in a['actor_state_dict'].items() if k not in allowed),
    wheel_rows_exact=all(torch.equal(a['actor_state_dict'][k][4:],b['actor_state_dict'][k][4:]) for k in allowed),
    all_old_actor_Adam_exact=all(exact_state_equal(old['state'].get(i),new['state'].get(i)) for i in range(28)),
    all_six_entry_Adam_steps12=all(i not in old['state'] and float(new['state'][i]['step'])==12. for i in range(28,34)),
    basis_Adam_moments_zero=all(torch.count_nonzero(new['state'][i][k])==0 for i in range(28,32) for k in ('exp_avg','exp_avg_sq')),
    critic_changed=not exact_state_equal(a['critic_state_dict'],b['critic_state_dict']),
    frozen_inputs_exact=all(digest(Path(n))==h for n,h in plan['source_sha256'].items()),
    initial_restore_all=all(json.loads((Path(plan['supervisor_output'])/'initial_restore.json').read_text())['checks'].values()))
before,_=load_mean(source);after,_=load_mean(target);checks['strict_V3_decode']=True
trace=B/'diagnosis025_motion_damping080_fixed/evaluation/trajectory.npz'
frames=np.load(trace)['observations'].reshape(-1,35)
outside_count=outside_different=wheel_different=0
with torch.inference_mode():
    for values in np.array_split(frames,(len(frames)+511)//512):
        x=torch.from_numpy(values.copy());p,q=before(x),after(x)
        outside=handoff_entry_gate(x).ravel()==0
        outside_count+=int(outside.sum());outside_different+=int((p[outside]!=q[outside]).any(-1).sum())
        wheel_different+=int((p[:,4:]!=q[:,4:]).any(-1).sum())
checks['outside_entry_function_exact']=outside_different==0
checks['wheel_function_exact']=wheel_different==0
checks={k:bool(v) for k,v in checks.items()}
first=json.loads((run/'first_rollout_hashes.json').read_text())['tensors']
prior=json.loads((B/'pilot009_periodic_motion_entry/first_rollout_hashes.json').read_text())['tensors']
first_comparison={k:v==prior.get(k) for k,v in first.items()}
out=dict(schema='native-common-only-entry-endpoint-v1',checks=checks,
    checkpoint=str(target),checkpoint_sha256=digest(target),source=str(source),source_sha256=digest(source),
    changed_actor_tensors=changed,phase_guard=guard['updates'][-1],
    source_clock=1501,branch_clock=1502,actual_new_PPO=1,
    actual_experiment_consumed=2622,repair_optimizer_updates=242,
    first_rollout_comparison_against009=first_comparison,
    function_probe=dict(trace=str(trace),sha256=digest(trace),frames=len(frames),outside_count=outside_count,
        outside_different=outside_different,wheel_different=wheel_different),
    frozen_inputs=len(plan['source_sha256']),decision='Reject continuation and adoption: preset phase margin failed. Seven unused updates closed.',
    native_behavior_evaluation_performed=False,qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'common_entry_endpoint001.json',out)
print(json.dumps(out,indent=2));assert all(checks.values()),checks
progress=json.loads((R/'progress.json').read_text())
progress.update(status='common_entry_phase_guard_stop_sealed',new_PPO_attempted=242,new_PPO_completed=242,
    repair_optimizer_updates=242,actual_experiment_consumed=2622,new_PPO_admitted=0,
    active_training=None,active_supervisor=None,active_job=None,training_admission=None,
    latest_research_decision=out['decision'],
    planned_PPO='No new PPO admitted. Native common-only update1 failed development direction margin; remaining7 closed.',
    lineage='Best current branch1501. Independent012 stopped at1502 after1 actual update; cumulative repair242 and experiment2622. No refunds.',updated_unix=time.time())
progress['experiments'].append(dict(name='pilot012_common_entry002',updates=1,planned_updates=8,
    clock=1502,status='phase_guard_stopped',checkpoint=str(target),checkpoint_sha256=digest(target)))
write_json(R/'progress.json',progress)

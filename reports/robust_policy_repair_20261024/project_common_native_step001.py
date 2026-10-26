"""One CPU projection of a charged native PPO step into existing phase limits.

This is a documented weight edit, never an optimizer-equivalent continuation.
No native job or additional PPO is authorized by this script.
"""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time

sys.path[:0]=['src','scripts',str(Path(__file__).parent)]
import numpy as np
import osqp
from scipy import sparse
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest,write_json
from wheeled_algo.entry_checkpoint import entry_digest
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.getup_symmetry import reflect_observation
from wheeled_algo.paired_recovery import wrap
from wheeled_algo.periodic_entry_actor import exact_state_equal,periodic_canonical_entry_logits
from entry_common_guard001 import PhaseGuard

torch.set_num_threads(1)
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
OUT=R/'native_common_projection001.json'
assert not OUT.exists()
source=B/'pilot012_common_entry002/model_stopped.pt'
reference=B/'function_root_entry002/model_initial.pt'
raw_state,root=[torch.load(p,map_location='cpu',weights_only=False) for p in (source,reference)]
assert raw_state['infos']['completed_updates']==1502 and root['infos']['completed_updates']==1501
mean,_=load_mean(reference);prefix=mean.entry_recovery[:-1]
bits=torch.tensor([[float(bool(mask&(1<<j))) for j in range(4)] for mask in range(16)])
feature_source=R/'fit_entry_phase_path001.py'
nodes=[n for n in ast.parse(feature_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('hidden','design')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(feature_source),'exec'),globals())

def materialize(obs):
    matrices=[];outputs=[]
    with torch.inference_mode():
        for batch in obs.split(512):
            l,r,old=design(batch)
            matrices.append(torch.stack((l,r),1).reshape(-1,258).numpy())
            outputs.append(old.reshape(-1).numpy())
    return np.concatenate(matrices),np.concatenate(outputs)

started=time.time()
data=np.load(R/'entry_phase_path_dataset001.npz')
observations=torch.from_numpy(data['observations'].copy())
desired=data['desired_common'].reshape(-1);phase=data['phase'][data['pose_ids']]
gate=handoff_entry_gate(observations).numpy()
far=((np.abs(phase)>=.75)&(gate>.99)).reshape(-1)
near=((np.abs(phase)<1e-6)&(gate>.99)).reshape(-1)
X,original=materialize(observations)
trace=B/'diagnosis025_motion_damping080_fixed/evaluation/trajectory.npz'
recorded=np.load(trace)['observations'][::5].reshape(-1,35)
recorded=torch.from_numpy(recorded.copy());recorded=recorded[handoff_entry_gate(recorded).ravel()>0]
Xa,_=materialize(recorded)
W='mlp.mean.entry_recovery.4.weight';C='mlp.mean.entry_recovery.4.bias'
proposed=[]
for first in (0,2):
    dw=(raw_state['actor_state_dict'][W]-root['actor_state_dict'][W])[first:first+2].mean(0)
    db=(raw_state['actor_state_dict'][C]-root['actor_state_dict'][C])[first:first+2].mean()
    proposed.append(torch.cat((dw,db[None])).double().numpy())
proposed=np.concatenate(proposed)
G=X.T@X/len(X)+Xa.T@Xa/len(Xa)+1e-4*np.eye(258)
rhs=G@proposed
pool=np.flatnonzero(far|near)
PX=X[pool].copy();is_far=far[pool];sign=np.sign(desired[pool])
PX[is_far]*=sign[is_far,None]
lo=np.full(len(pool),-np.inf);hi=np.full(len(pool),np.inf)
lo[is_far]=.25-sign[is_far]*original[pool[is_far]]
lo[~is_far]=-.06-original[pool[~is_far]]
hi[~is_far]=.06-original[pool[~is_far]]
selected=set(np.linspace(0,len(pool)-1,min(512,len(pool)),dtype=int).tolist())
rounds=[];vector=np.zeros(258);status='not_solved'
for iteration in range(8):
    ids=np.array(sorted(selected));solver=osqp.OSQP()
    solver.setup(P=sparse.csc_matrix(np.triu(G)),q=-rhs,A=sparse.csc_matrix(PX[ids]),
        l=lo[ids],u=hi[ids],verbose=False,eps_abs=1e-7,eps_rel=1e-7,
        max_iter=20000,time_limit=45.,polishing=True)
    solver.warm_start(x=vector);result=solver.solve();status=result.info.status
    row=dict(iteration=iteration+1,status=status,selected_constraints=len(ids),solver_seconds=result.info.run_time)
    rounds.append(row)
    if status not in ('solved','solved inaccurate'):break
    vector=result.x;value=PX@vector;violation=np.maximum(lo-value,value-hi)
    row.update(violated_constraints=int((violation>2e-6).sum()),max_constraint_violation=float(violation.max()))
    print(json.dumps(row),flush=True)
    if violation.max()<=2e-6:break
    worst=np.argsort(violation)[::-1]
    missing=[int(i) for i in worst if violation[i]>2e-6 and int(i) not in selected][:256]
    if not missing:break
    selected.update(missing)
candidate=deepcopy(raw_state)
for first,side in ((0,0),(2,1)):
    v=torch.tensor(vector[side*129:(side+1)*129],dtype=torch.float32)
    for row in (first,first+1):
        candidate['actor_state_dict'][W][row]=root['actor_state_dict'][W][row]+v[:-1]
        candidate['actor_state_dict'][C][row]=root['actor_state_dict'][C][row]+v[-1]
probe=deepcopy(mean)
probe.load_state_dict({k.removeprefix('mlp.mean.'):v for k,v in candidate['actor_state_dict'].items() if k.startswith('mlp.mean.')})
guard=PhaseGuard(mean,R/'entry_phase_path_dataset001.npz');guard.mean=probe
phase_metrics,phase_checks=guard.measure()
raw_mean,_=load_mean(source)
projected_delta=[];native_delta=[]
with torch.inference_mode():
    for x in recorded.split(512):
        a,b,c=mean(x),probe(x),raw_mean(x)
        projected_delta.append(wrap((b[:,:4]-a[:,:4])*.25).numpy())
        native_delta.append(wrap((c[:,:4]-a[:,:4])*.25).numpy())
projected_delta=np.concatenate(projected_delta);native_delta=np.concatenate(native_delta)
projected_rms=float(np.sqrt(np.mean(projected_delta**2)));native_rms=float(np.sqrt(np.mean(native_delta**2)))
checks=dict(solver_solved=status in ('solved','solved inaccurate'),
    full_constraint_pool=rounds[-1].get('max_constraint_violation',np.inf)<=2e-6,
    **{'phase_'+k:v for k,v in phase_checks.items()},
    only_last_entry_tensors=all(torch.equal(v,candidate['actor_state_dict'][k]) for k,v in root['actor_state_dict'].items() if k not in (W,C)),
    wheels_exact=all(torch.equal(root['actor_state_dict'][k][4:],candidate['actor_state_dict'][k][4:]) for k in (W,C)),
    critic_Adam_RNG_exact=all(exact_state_equal(raw_state[k],candidate[k]) for k in raw_state if k not in ('actor_state_dict','infos')),
    meaningful_recorded_target_change=float(np.max(np.abs(projected_delta)))>=1e-4,
    retains_native_change_rms=projected_rms>=.15*native_rms,
    bounded_recorded_change=float(np.max(np.abs(projected_delta)))<=.005)
out=dict(schema='nearest-feasible-native-common-step-v1',checks={k:bool(v) for k,v in checks.items()},
    source=str(source),source_sha256=digest(source),reference=str(reference),reference_sha256=digest(reference),
    source_learning_clock=1502,phase_dataset_sha256=digest(R/'entry_phase_path_dataset001.npz'),
    real_trace=str(trace),real_trace_sha256=digest(trace),real_entry_frames=len(recorded),
    feature_source_sha256=digest(feature_source),rounds=rounds,phase_metrics=phase_metrics,
    objective='Closest to the actual258-coefficient PPO proposal in equal-weight phase and recorded-entry output metrics plus1e-4 coefficient regularization.',
    fixed_limits=dict(far_common_logit_min=.25,stopped_common_logit_max=.06,
        max_cut_rounds=8,max_new_cuts_per_round=256,solver_seconds_per_round=45.,
        min_native_rms_fraction=.15,min_peak_change_rad=.0001,max_peak_change_rad=.005),
    metrics=dict(native_delta_rms_rad=native_rms,projected_delta_rms_rad=projected_rms,
        projected_delta_peak_rad=float(np.max(np.abs(projected_delta))),
        retained_rms_fraction=projected_rms/max(native_rms,1e-30),coefficient_change_l2=float(np.linalg.norm(vector)),elapsed_seconds=time.time()-started),
    new_PPO=0,actual_experiment_consumed=2622,repair_optimizer_updates=242,
    optimizer_continuation='Not coordinate equivalent; keep sealed native Adam only for provenance. No further PPO admitted.',
    qualification=None,hardware_started=False,created_unix=time.time())
if all(checks.values()):
    candidate['infos']['function_root_learning']['entry_state_sha256']=entry_digest(candidate['actor_state_dict'])
    candidate['infos']['post_PPO_phase_projection']=dict(schema='offline-common-phase-projection-v1',
        source_checkpoint=str(source.resolve()),source_sha256=digest(source),reference=str(reference.resolve()),
        reference_sha256=digest(reference),script=str(Path(__file__)),script_sha256=digest(Path(__file__)),
        new_PPO=0,actual_learning_clock=1502,optimizer='Retained for provenance; not an equivalent continuation',
        constraints='Original far.25/stopped.06 logits over54450 development frames',effective_parameters=258)
    target=B/'native_common_projected001/model_probe.pt';target.parent.mkdir(parents=True,exist_ok=False)
    torch.save(candidate,target);load_mean(target)
    out.update(checkpoint=str(target),checkpoint_sha256=digest(target),decision='CPU checks passed; one fixed native evaluation may be reviewed, no PPO.')
else:out['decision']='Reject candidate; no native or PPO admitted.'
write_json(OUT,out);print(json.dumps(out,indent=2));assert all(checks.values()),checks

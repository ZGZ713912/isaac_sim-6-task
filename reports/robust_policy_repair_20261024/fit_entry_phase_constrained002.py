"""Bounded convex common-output fit with explicit direction/stop constraints."""
from copy import deepcopy
from pathlib import Path
import ast
import json
import math
import sys
import time

sys.path[:0] = ['src','scripts']
import numpy as np
import osqp
from scipy import sparse
import torch
from build_v6_height_lookup import digest
from export_robust_getup import load_mean
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.getup_symmetry import reflect_observation
from wheeled_algo.paired_recovery import wrap
from wheeled_algo.periodic_entry_actor import periodic_canonical_entry_logits

torch.set_num_threads(1)
B = Path('runs/robust_policy_repair_20261024')
R = Path('reports/robust_policy_repair_20261024')
OUTPUT = R/'entry_phase_constrained_feasibility002.json'
PROGRESS = R/'entry_phase_constrained_progress002.json'
assert not OUTPUT.exists() and not PROGRESS.exists()
SOURCE = B/'periodic_entry_encoder001/model_probe.pt'
FEATURE_SOURCE = R/'fit_entry_phase_path001.py'
DATA = R/'entry_phase_path_dataset001.npz'
initial = json.loads((R/'entry_phase_path_feasibility001.json').read_text())
assert initial['source_sha256']==digest(SOURCE) and initial['dataset_sha256']==digest(DATA)
mean,_ = load_mean(SOURCE)
prefix = mean.entry_recovery[:-1]
bits = torch.tensor([[float(bool(mask & (1<<j))) for j in range(4)] for mask in range(16)])
# Reuse exactly the independently checked feature algebra, without rerunning
# the historical solve or overwriting any prior receipt.
nodes = [n for n in ast.parse(FEATURE_SOURCE.read_text()).body
         if isinstance(n,ast.FunctionDef) and n.name in ('hidden','design')]
assert len(nodes)==2
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(FEATURE_SOURCE),'exec'),globals())
z = np.load(DATA)
targets = torch.tensor(z['observations'])
desired = z['desired_common'].reshape(-1)
phase = z['phase'][z['pose_ids']]
gate = handoff_entry_gate(targets).numpy()
far = ((np.abs(phase)>=.75)&(gate>.99)).reshape(-1)
near = ((np.abs(phase)<1e-6)&(gate>.99)).reshape(-1)

anchors = []
for source in initial['anchor_sources']:
    path = Path(source['path'])
    assert digest(path)==source['sha256']
    trace = np.load(path)
    report = json.loads((path.parent/'report.json').read_text())
    for lane in report['lanes']:
        if lane['recovered']:
            i = lane['lane']
            obs = torch.tensor(trace['observations'][::10,i])
            active = torch.tensor((trace['alive'][::10,i]&~trace['passive'][::10,i]).copy())
            select = active & (handoff_entry_gate(obs)[:,0]>0.)
            if select.any():
                anchors.append(obs[select])
anchor = torch.cat(anchors)


def materialize(obs):
    X,original = [],[]
    with torch.inference_mode():
        for batch in obs.split(512):
            l,r,old = design(batch)
            X.append(torch.stack((l,r),1).reshape(-1,258).numpy())
            original.append(old.reshape(-1).numpy())
    return np.concatenate(X),np.concatenate(original)


started = time.time()
Xa,oa = materialize(anchor)
X,old = materialize(targets)
anchor_weight,ridge = 100.,1e-5
gram = anchor_weight*(Xa.T@Xa)/len(Xa)+(X.T@X)/len(X)+ridge*np.eye(258)
rhs = X.T@(desired-old)/len(X)
pool = np.flatnonzero(far|near)
PX = X[pool].copy()
lo,hi = np.full(len(pool),-np.inf),np.full(len(pool),np.inf)
is_far = far[pool]
sign = np.sign(desired[pool])
PX[is_far] *= sign[is_far,None]
lo[is_far] = .25-sign[is_far]*old[pool[is_far]]
lo[~is_far] = -.06-old[pool[~is_far]]
hi[~is_far] = .06-old[pool[~is_far]]
selected = set(np.linspace(0,len(pool)-1,min(512,len(pool)),dtype=int).tolist())
rounds = []
vector = np.array(initial['edit_vector'])
status = 'not_solved'
for iteration in range(8):
    ids = np.array(sorted(selected))
    A = sparse.csc_matrix(np.concatenate((Xa,PX[ids])))
    lower = np.concatenate((np.full(len(Xa),-.2),lo[ids]))
    upper = np.concatenate((np.full(len(Xa),.2),hi[ids]))
    solver = osqp.OSQP()
    solver.setup(P=sparse.csc_matrix(np.triu(gram)),q=-rhs,A=A,l=lower,u=upper,
        verbose=False,eps_abs=1e-6,eps_rel=1e-6,max_iter=20000,time_limit=45.,polishing=True)
    solver.warm_start(x=vector)
    result = solver.solve()
    status = result.info.status
    row = dict(iteration=iteration+1,status=status,selected_constraints=len(ids),
               solver_iterations=result.info.iter,solver_seconds=result.info.run_time)
    rounds.append(row)
    if status not in ('solved','solved inaccurate'):
        print(json.dumps(row),flush=True)
        break
    vector = result.x
    values = PX@vector
    violation = np.maximum(lo-values,values-hi)
    row.update(max_constraint_violation=float(violation.max()),
               violated_constraints=int((violation>1e-5).sum()))
    PROGRESS.write_text(json.dumps(dict(status='running',rounds=rounds,new_PPO=0,
        actual_experiment_consumed=2603,updated_unix=time.time()),indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if violation.max()<=1e-5:
        break
    # Fixed cut budget: at most 256 worst missing constraints per round.
    worst = np.argsort(violation)[::-1]
    added = [int(i) for i in worst if violation[i]>1e-5 and int(i) not in selected][:256]
    if not added:
        break
    selected.update(added)

output = old+X@vector
anchor_delta = Xa@vector
probe = deepcopy(mean)
for side in range(2):
    v = torch.tensor(vector[side*129:(side+1)*129],dtype=torch.float32)
    for row in (2*side,2*side+1):
        probe.entry_recovery[-1].weight.data[row] += v[:-1]
        probe.entry_recovery[-1].bias.data[row] += v[-1]
with torch.inference_mode():
    final = torch.cat([probe(x)[:,:4] for x in targets.split(512)])*.25
    lead = wrap(final-targets[:,10:14]).unflatten(-1,(2,2)).mean(-1).numpy().reshape(-1)
    actual_common = []
    for obs in targets[::101].split(512):
        frame = torch.cat((obs[:,:14],torch.zeros_like(obs[:,14:16]),obs[:,16:]),-1)
        raw = periodic_canonical_entry_logits(probe.entry_recovery,frame,probe.feature_scale)
        actual_common.append(raw[:,:4].unflatten(-1,(2,2)).mean(-1).numpy())
actual_common = np.concatenate(actual_common).reshape(-1)
design_error = float(np.max(np.abs(actual_common-output.reshape(-1,2)[::101].reshape(-1))))
signed_output = output*np.sign(desired)
signed_lead = lead*np.sign(desired)
metrics = dict(target_states=len(targets),anchor_frames=len(anchor),constraint_pool=len(pool),
    far_common_signed_min=float(signed_output[far].min()),
    far_lead_signed_min_rad=float(signed_lead[far].min()),
    far_lead_correct_fraction=float(np.mean(signed_lead[far]>.003)),
    near_common_max=float(np.max(np.abs(output[near]))),
    near_lead_max_rad=float(np.max(np.abs(lead[near]))),
    anchor_change_rms=float(np.sqrt(np.mean(anchor_delta**2))),
    anchor_change_max=float(np.max(np.abs(anchor_delta))),
    target_rmse=float(np.sqrt(np.mean((output-desired)**2))),
    independent_design_error=design_error,edit_vector_l2=float(np.linalg.norm(vector)),
    elapsed_seconds=time.time()-started)
checks = dict(solver_solved=status in ('solved','solved inaccurate'),
    complete_direction_margin=metrics['far_common_signed_min']>=.24998,
    complete_actual_target_direction=metrics['far_lead_correct_fraction']==1.,
    stopped_leg_common=metrics['near_common_max']<=.06002,
    stopped_leg_target=metrics['near_lead_max_rad']<=.006,
    retention_rms=metrics['anchor_change_rms']<=.05,
    retention_max=metrics['anchor_change_max']<=.20002,
    design_matches_NN=design_error<1e-5,finite=bool(np.isfinite(vector).all()))
out = dict(schema='entry-phase-constrained-fit-v1',source=str(SOURCE),source_sha256=digest(SOURCE),
    feature_source=str(FEATURE_SOURCE),feature_sha256=digest(FEATURE_SOURCE),
    dataset=str(DATA),dataset_sha256=digest(DATA),predeclared_parameters=dict(
        maximum_cut_rounds=8,maximum_new_cuts_per_round=256,solver_time_limit_seconds_per_round=45,
        direction_logit_margin=.25,stopped_leg_common_bound=.06,anchor_max_change=.2,
        anchor_weight=anchor_weight,ridge=ridge),rounds=rounds,metrics=metrics,checks=checks,
    edit_vector=vector.tolist(),decision='eligible_for_checkpoint_preflight' if all(checks.values()) else 'reject_without_native_or_PPO',
    scope='All phase states are development constraints. The soft teacher magnitude RMSE is descriptive; explicit direction, stop and successful-trajectory retention are required. No behavioral holdout or native contact proof.',
    new_PPO=0,gradient_optimizer_updates=0,actual_experiment_consumed=2603,
    qualification=None,hardware_started=False,created_unix=time.time())
OUTPUT.write_text(json.dumps(out,indent=2)+'\n')
PROGRESS.write_text(json.dumps(dict(status='completed',decision=out['decision'],rounds=rounds,
    updated_unix=time.time(),new_PPO=0,actual_experiment_consumed=2603),indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('edit_vector','rounds')},indent=2),flush=True)

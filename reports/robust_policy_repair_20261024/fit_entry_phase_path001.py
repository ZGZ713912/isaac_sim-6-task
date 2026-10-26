"""One CPU feasibility solve over complete feasible leg phases; no native/PPO."""
from copy import deepcopy
from pathlib import Path
import hashlib
import itertools
import json
import math
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from build_robust_getup_bank import Builder, digest
from export_robust_getup import load_mean
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.getup_symmetry import reflect_observation
from wheeled_algo.paired_recovery import paired_offsets, wrap
from wheeled_algo.periodic_entry_actor import periodic_canonical_entry_logits

torch.set_num_threads(1)
B = Path('runs/robust_policy_repair_20261024')
R = Path('reports/robust_policy_repair_20261024')
SOURCE = B/'periodic_entry_encoder001/model_probe.pt'
OUTPUT = R/'entry_phase_path_feasibility001.json'
DATA = R/'entry_phase_path_dataset001.npz'
assert not OUTPUT.exists() and not DATA.exists()
started = time.time()
mean, _ = load_mean(SOURCE)
prefix = mean.entry_recovery[:-1]
bits = torch.tensor([[float(bool(mask & (1 << j))) for j in range(4)] for mask in range(16)])


def hidden(frame):
    angles = frame[:, 10:14]
    phase = ((angles.abs()-(math.pi-.35))/.35).clamp(0., 1.)
    weights = .5*phase.square()*(3.-2.*phase)
    other = angles-2.*math.pi*angles.sign()
    q = angles[:, None, :]+bits[None, :, :]*(other-angles)[:, None, :]
    expanded = frame[:, None, :].expand(-1, 16, -1)
    charts = torch.cat((expanded[:, :, :10], q, expanded[:, :, 14:]), -1)
    values = prefix(torch.tanh(charts.reshape(-1,35)*mean.feature_scale)).reshape(-1,16,128)
    probability = torch.where(bits[None] > 0., weights[:,None], 1.-weights[:,None]).prod(-1)
    base = prefix(torch.tanh(frame*mean.feature_scale))
    h = torch.where(weights.amax(-1,keepdim=True)>0.,
                    base+(probability[:,:,None]*(values-base[:,None])).sum(1),base)
    return torch.cat((h,torch.ones_like(h[:,:1])),-1)


def design(obs):
    frame = torch.cat((obs[:,:14],torch.zeros_like(obs[:,14:16]),obs[:,16:]),-1)
    direct, reflected = hidden(frame), hidden(reflect_observation(frame))
    w = ((obs[:,8:9]/.15+1.)*.5).clamp(0.,1.)
    # Reflection negates and exchanges left/right common outputs. Two independent
    # last-layer vectors therefore require the full 258-column design.
    left = torch.cat(((1.-w)*direct,-w*reflected),-1)
    right = torch.cat((-w*reflected,(1.-w)*direct),-1)
    raw = periodic_canonical_entry_logits(mean.entry_recovery,frame,mean.feature_scale)
    common = raw[:,:4].unflatten(-1,(2,2)).mean(-1)
    return left.double(), right.double(), common.double()


builder = Builder(Path('model/纯底盘_v7_flat_base107/urdf'),
                  digest(Path('model/纯底盘_v7_flat_base107/urdf/manifest.json')))
knees = [50.,67.854970759,95.]
phases = [-3.05,-2.5,-1.75,-1.,-.45,0.,.45,1.,1.75,2.5,3.05]
side_states = []
for k,beta in enumerate(knees):
    base = builder.align_wheels(builder.knee_pose(beta,beta))
    for p,phase in enumerate(phases):
        q = builder.common_rotate(base,phase,phase)
        builder.validate_q(q)
        side_states.append((k,p,q))
    print(f'closed-chain continuation knee={beta:g} complete',flush=True)

nominal = np.array([builder.nominal[n] for n in builder.controls])
leg_control = [0,1,3,4]
pose_q, pose_phase, pose_index, full_q, closure, physical_knees = [],[],[],[],[],[]
for left,right in itertools.product(side_states,repeat=2):
    # The asset has independent L/R trees. Merge full solved trees and verify
    # every closure and spring constraint again; never merge four motor q only.
    q = {n: (left[2][n] if n.startswith('L') else right[2][n]) for n in builder.nominal}
    check = builder.validate_q(q)
    pose_q.append((np.array([q[n] for n in builder.controls])-nominal)[leg_control])
    pose_phase.append([phases[left[1]],phases[right[1]]])
    pose_index.append([left[0],right[0],left[1],right[1]])
    full_q.append([q[n] for n in builder.nominal])
    closure.append(check['gap_m'])
    physical_knees.append(check['knee_deg'])

pose_q = torch.tensor(np.asarray(pose_q),dtype=torch.float32)
pose_phase = torch.tensor(pose_phase,dtype=torch.float32)
pose_index = np.array(pose_index)
teacher = -.9*torch.tanh(pose_phase.sin()/.2)
# Actual observed rad/s, encoded with the verified SCUT35 factor 0.1.
velocity_patterns = [(0.,0.),(.5,.5),(1.5,1.5),(1.5,0.),(0.,1.5)]
gravity = [(0.,0.,-1.),(math.sin(.05),0.,-math.cos(.05)),
           (-math.sin(.05),0.,-math.cos(.05)),(0.,math.sin(.05),-math.cos(.05)),
           (0.,-math.sin(.05),-math.cos(.05))]
rows, labels, indices = [],[],[]
for speed,lead,g in itertools.product(velocity_patterns,(0.,.03),gravity):
    obs = torch.zeros((len(pose_q),35),dtype=torch.float32)
    obs[:,3] = .305*5.
    obs[:,7:10] = torch.tensor(g)
    obs[:,10:14] = wrap(pose_q)
    signs = teacher.sign()
    obs[:,16:20] = (.1*signs*torch.tensor(speed)).repeat_interleave(2,-1)
    prior = pose_q+(lead*signs).repeat_interleave(2,-1)
    obs[:,22:26] = paired_offsets(prior,wrap(pose_q))/.25
    obs[:,28] = 1.
    rows.append(obs)
    labels.append(teacher)
    indices.append(np.arange(len(pose_q)))
targets = torch.cat(rows)
desired = torch.cat(labels).double()
pose_ids = np.concatenate(indices)
train = torch.tensor((pose_index[pose_ids].sum(-1)%3)!=0)
np.savez_compressed(DATA,observations=targets.numpy(),desired_common=desired.numpy(),
    pose_ids=pose_ids,train=train.numpy(),full_q=np.array(full_q),
    joint_names=np.array(list(builder.nominal)),phase=pose_phase.numpy(),
    physical_knees=np.array(physical_knees),closure_m=np.array(closure),pose_index=pose_index)

anchors,anchor_sources = [],[]
for name in ('diagnosis012_final_arithmetic_fixed/evaluation',
             'diagnosis012_final_arithmetic_nominal/robust',
             'diagnosis012_final_arithmetic_rmcs/evaluation'):
    directory = B/name
    report = json.loads((directory/'report.json').read_text())
    z = np.load(directory/'trajectory.npz')
    for lane in report['lanes']:
        if not lane['recovered']:
            continue
        i = lane['lane']
        obs = torch.tensor(z['observations'][::10,i])
        active = torch.tensor((z['alive'][::10,i]&~z['passive'][::10,i]).copy())
        select = active & (handoff_entry_gate(obs)[:,0]>0.)
        if select.any():
            anchors.append(obs[select])
    anchor_sources.append(dict(path=str(directory/'trajectory.npz'),sha256=digest(directory/'trajectory.npz')))
anchor = torch.cat(anchors)

anchor_weight,ridge = 100.,1e-5
gram = ridge*torch.eye(258,dtype=torch.float64)
rhs = torch.zeros(258,dtype=torch.float64)
with torch.inference_mode():
    for obs in anchor.split(512):
        l,r,_ = design(obs)
        gram += anchor_weight*(l.T@l+r.T@r)/(2*len(anchor))
    for begin in range(0,len(targets),512):
        selected = train[begin:begin+512]
        if not selected.any():
            continue
        obs = targets[begin:begin+512][selected]
        y = desired[begin:begin+512][selected]
        l,r,old = design(obs)
        gram += (l.T@l+r.T@r)/(2*int(train.sum()))
        rhs += (l.T@(y[:,0]-old[:,0])+r.T@(y[:,1]-old[:,1]))/(2*int(train.sum()))
    vector = torch.linalg.solve(gram,rhs)
    def predict(obs):
        l,r,old = design(obs)
        delta = torch.stack((l@vector,r@vector),-1)
        return old+delta,delta
    anchor_delta = torch.cat([predict(x)[1] for x in anchor.split(512)])
    output = torch.cat([predict(x)[0] for x in targets.split(512)])

probe = deepcopy(mean)
for side in range(2):
    v = vector[side*129:(side+1)*129].float()
    for row in (2*side,2*side+1):
        probe.entry_recovery[-1].weight.data[row] += v[:-1]
        probe.entry_recovery[-1].bias.data[row] += v[-1]

with torch.inference_mode():
    sample = targets[::101]
    frame = torch.cat((sample[:,:14],torch.zeros_like(sample[:,14:16]),sample[:,16:]),-1)
    raw = periodic_canonical_entry_logits(probe.entry_recovery,frame,probe.feature_scale)
    actual_common = raw[:,:4].unflatten(-1,(2,2)).mean(-1).double()
    design_error = float((actual_common-predict(sample)[0]).abs().max())
    flow = torch.cat([probe(x)[:,:4] for x in targets.split(512)])*.25
    delta = wrap(flow-targets[:,10:14]).unflatten(-1,(2,2)).mean(-1)
    phase_for_rows = pose_phase[pose_ids]
    far = (phase_for_rows.abs()>=.75) & (handoff_entry_gate(targets)>.99)
    signed_output = output*desired.sign()
    signed_flow = delta*desired.sign()

metrics = dict(feasible_full_states=len(pose_q),observation_states=len(targets),
    anchor_frames=len(anchor),train_frames=int(train.sum()),heldout_frames=int((~train).sum()),
    closure_peak_m=max(closure),physical_knee_min_deg=float(np.min(physical_knees)),
    physical_knee_max_deg=float(np.max(physical_knees)),
    anchor_common_change_rms=float(anchor_delta.square().mean().sqrt()),
    anchor_common_change_max=float(anchor_delta.abs().max()),
    target_rmse=float((output-desired).square().mean().sqrt()),
    heldout_rmse=float((output[~train]-desired[~train]).square().mean().sqrt()),
    far_common_signed_min=float(signed_output[far].min()),
    far_common_correct_fraction=float((signed_output[far]>.08).double().mean()),
    far_target_lead_signed_min_rad=float(signed_flow[far].min()),
    far_target_lead_correct_fraction=float((signed_flow[far]>.003).double().mean()),
    independent_design_error=design_error,edit_vector_l2=float(vector.norm()),
    elapsed_seconds=time.time()-started)
checks = dict(geometry=metrics['closure_peak_m']<=1e-8,
    design_matches_actual_NN=design_error<1e-5,
    retention_rms=metrics['anchor_common_change_rms']<=.05,
    retention_max=metrics['anchor_common_change_max']<=.2,
    heldout_rmse=metrics['heldout_rmse']<=.15,
    complete_common_direction=metrics['far_common_correct_fraction']==1.,
    complete_actual_target_direction=metrics['far_target_lead_correct_fraction']==1.,
    finite=bool(torch.isfinite(vector).all()))
result = dict(schema='entry-full-phase-path-feasibility-v1',source=str(SOURCE),source_sha256=digest(SOURCE),
    dataset=str(DATA),dataset_sha256=digest(DATA),metrics=metrics,checks=checks,
    anchor_sources=anchor_sources,teacher='Offline periodic toward-wheel-down common output: -0.9*tanh(sin(phase)/0.2).',
    method='One predeclared 258-parameter existing last-layer ridge solve; independent common outputs; no gradient optimizer.',
    anchor_weight=anchor_weight,ridge=ridge,edit_vector=vector.tolist(),
    scope='Feasible closed chains, knee/spring travel and actor function directions. No contact dynamics or recovery proof; interleaved configuration holdout is development only.',
    decision='eligible_for_checkpoint_preflight' if all(checks.values()) else 'reject_fit_without_native_or_PPO',
    new_PPO=0,gradient_optimizer_updates=0,actual_experiment_consumed=2603,
    qualification=None,hardware_started=False,created_unix=time.time())
OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('edit_vector','anchor_sources')},indent=2),flush=True)

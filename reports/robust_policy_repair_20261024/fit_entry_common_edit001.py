"""Closed-form common-output edit feasibility; no checkpoint or native run."""
from pathlib import Path
import hashlib
import json
import math
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.getup_symmetry import reflect_observation
from wheeled_algo.periodic_entry_actor import periodic_canonical_entry_logits

torch.set_num_threads(1)
B = Path('runs/robust_policy_repair_20261024')
R = Path('reports/robust_policy_repair_20261024')
source = B/'periodic_entry_encoder001/model_probe.pt'
mean, _ = load_mean(source)
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
    return torch.where(weights.amax(-1, keepdim=True)>0.,
                       base+(probability[:,:,None]*(values-base[:,None])).sum(1), base)


def features(obs):
    frame = torch.cat((obs[:,:14], torch.zeros_like(obs[:,14:16]), obs[:,16:]), -1)
    w = ((obs[:,8:9]/.15+1.)*.5).clamp(0.,1.)
    direct = hidden(frame)
    h = direct+w*(hidden(reflect_observation(frame))-direct)
    return torch.cat((h, torch.ones_like(h[:,:1])), -1)


def common_component(obs):
    frame = torch.cat((obs[:,:14], torch.zeros_like(obs[:,14:16]), obs[:,16:]), -1)
    raw = periodic_canonical_entry_logits(mean.entry_recovery, frame, mean.feature_scale)
    return .25*(raw[:,0]+raw[:,1]-raw[:,2]-raw[:,3])


# Retain every successful entry/fallen source trajectory, across all three banks.
anchors = []
anchor_sources = []
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
        select = active & (handoff_entry_gate(obs)[:,0] > 0.)
        if select.any():
            anchors.append(obs[select])
    anchor_sources.append(dict(trace=str(directory/'trajectory.npz'),
        sha256=hashlib.sha256((directory/'trajectory.npz').read_bytes()).hexdigest()))

# A measured wheels-up stall labels the desired common direction. Geometry is
# used only to select offline labels. It never becomes a runtime actor input.
target_obs = []
target_sources = []
directory = B/'diagnosis021_periodic_entry_fixed/evaluation'
report = json.loads((directory/'report.json').read_text())
z = np.load(directory/'trajectory.npz')
for lane in report['lanes']:
    if lane['recovered'] or not lane['survived']:
        continue
    i = lane['lane']
    obs = torch.tensor(z['observations'][::2,i])
    times = z['time'][::2,i]
    clear = z['wheel_clearance'][::2,i].min(-1)
    active = z['alive'][::2,i]&~z['passive'][::2,i]
    pair = obs[:,10:14].unflatten(-1,(2,2))
    q = pair[:,:,0]+.5*torch.atan2((pair[:,:,1]-pair[:,:,0]).sin(), (pair[:,:,1]-pair[:,:,0]).cos())
    q = torch.atan2(q.sin(),q.cos())
    select = torch.tensor(active & (times>=6.) & (clear>.15)) & (obs[:,9]<-.9) & (q[:,0]<-2.) & (q[:,1]>2.)
    if select.any():
        target_obs.append(obs[select])
        target_sources.append(dict(name=lane['name'], actual_family=lane['post_settle_family'], frames=int(select.sum())))
anchor = torch.cat(anchors)
target = torch.cat(target_obs)
with torch.inference_mode():
    A = torch.cat([features(x) for x in anchor.split(512)]).double()
    T = torch.cat([features(x) for x in target.split(512)]).double()
    original = torch.cat([common_component(x) for x in target.split(512)]).double()
desired = .7-original
train_anchor = torch.arange(len(A))%3 != 0
train_target = torch.arange(len(T))%3 != 0
Xa, Xt, yt = A[train_anchor], T[train_target], desired[train_target]
# One predeclared solve; no search over native results or regularization values.
anchor_weight, ridge = 100., 1e-5
gram = anchor_weight*(Xa.T@Xa)/len(Xa)+(Xt.T@Xt)/len(Xt)+ridge*torch.eye(129,dtype=torch.float64)
vector = torch.linalg.solve(gram, Xt.T@yt/len(Xt))
anchor_change = A@vector
target_output = original+T@vector
metrics = dict(anchor_frames=len(A), target_frames=len(T),
    heldout_anchor_frames=int((~train_anchor).sum()), heldout_target_frames=int((~train_target).sum()),
    anchor_common_delta_rms=float(anchor_change.square().mean().sqrt()),
    anchor_common_delta_max=float(anchor_change.abs().max()),
    target_common_mean=float(target_output.mean()), target_common_min=float(target_output.min()),
    heldout_target_common_mean=float(target_output[~train_target].mean()),
    heldout_target_common_min=float(target_output[~train_target].min()),
    edit_vector_l2=float(vector.norm()), edit_vector_max=float(vector.abs().max()))
checks = dict(anchor_rms_at_most_005=metrics['anchor_common_delta_rms']<=.05,
    anchor_max_at_most_020=metrics['anchor_common_delta_max']<=.2,
    target_common_mean_at_least_050=metrics['target_common_mean']>=.5,
    heldout_target_common_min_at_least_035=metrics['heldout_target_common_min']>=.35,
    finite=bool(torch.isfinite(vector).all()))
out = dict(schema='entry-common-linear-edit-feasibility-v1', checkpoint=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    target_common_logit=.7, anchor_weight=anchor_weight, ridge=ridge, metrics=metrics,
    checks=checks, target_sources=target_sources, anchor_sources=anchor_sources,
    decision='eligible_for_weight_edit_preflight' if all(checks.values()) else 'reject_fit_without_native_or_PPO',
    validation_scope='Interleaved recorded-state function checks, not independent behavioral holdout.',
    edit_vector=vector.tolist(), source_weights_modified=False, optimizer_updates=0,
    new_PPO=0, actual_experiment_consumed=2603, qualification=None, hardware_started=False, created_unix=time.time())
(R/'entry_common_edit_feasibility001.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('edit_vector','anchor_sources')},indent=2))

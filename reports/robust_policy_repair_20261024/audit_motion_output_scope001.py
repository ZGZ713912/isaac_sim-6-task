"""Recorded direct output sensitivity of existing motion/fallen output layers."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest, write_json
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.robust_getup import tilt_gate

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
cp = B/'motion_damping080002/model_probe.pt'
mean, _ = load_mean(cp)
mean.requires_grad_(True)
groups = dict(motion_output=list(mean.arc[-1].parameters()), fallen_output=list(mean.recovery[-1].parameters()))
rows = []


def describe(name, obs):
    original = obs
    if len(obs)>48:
        obs = obs[np.linspace(0, len(obs)-1, 48).astype(int)]
    x = torch.from_numpy(obs.copy())
    action = mean(x)
    objective = (action*x.new_tensor([1., 1., 1., 1., .1, .1])).sum()
    norms = {}
    for group, parameters in groups.items():
        gradients = torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
        norms[group] = float(torch.stack([g.square().sum() for g in gradients if g is not None]).sum().sqrt())
    rows.append(dict(name=name, eligible_frames=len(original), audited_frames=len(obs),
        entry_gate_peak=float(handoff_entry_gate(x).max()),
        fallen_gate_peak=float(tilt_gate(x, True, True).max()),
        output_parameter_gradient_L2=norms))


traces = []
run = B/'diagnosis025_motion_damping080_fixed/evaluation'
report = json.loads((run/'report.json').read_text())
path = run/'trajectory.npz'
assert report['trajectory_sha256']==digest(path)
z = np.load(path)
quiet = []
for lane in report['lanes']:
    i = lane['lane']
    active = z['alive'][:, i]&~z['passive'][:, i]
    if lane['name'].startswith(('left_symmetric__', 'right_symmetric__', 'inverted_symmetric__')) and lane['recovered']:
        mask = active&(z['time'][:, i]>=4.)
        describe(lane['name']+'/late_motion', z['observations'][mask, i])
    if lane['behavior_case'].startswith('stand'):
        mask = active&(z['time'][:, i]>=2.)
        quiet.append(z['observations'][mask, i])
describe('standing_all_heights_control', np.concatenate(quiet))
traces.append(dict(path=str(path), sha256=digest(path)))
run = B/'diagnosis026_motion_damping080_nominal_pair/robust'
report = json.loads((run/'report.json').read_text())
path = run/'trajectory.npz'
assert report['trajectory_sha256']==digest(path)
z = np.load(path)
for lane in report['lanes']:
    if not lane.get('in_motion_fall'):
        continue
    i = lane['lane']
    ids = np.flatnonzero(z['in_motion_fall'][:, i]&z['alive'][:, i])
    start = z['time'][ids[0], i]
    ids = ids[z['time'][ids, i]<=start+.100001]
    describe(lane['name']+'/first100ms_fall', z['observations'][ids, i])
traces.append(dict(path=str(path), sha256=digest(path)))
checks = dict(late_moving_frames_present=len(rows)>=5,
    late_motion_output_sensitive=all(r['output_parameter_gradient_L2']['motion_output']>0. for r in rows if r['name'].endswith('/late_motion')),
    standing_motion_output_exact_zero=next(r for r in rows if r['name']=='standing_all_heights_control')['output_parameter_gradient_L2']['motion_output']==0.,
    actual_fall_cases_present=any(r['name'].endswith('/first100ms_fall') for r in rows))
out = dict(schema='recorded-motion-output-direct-scope-v1', checks=checks, rows=rows,
    checkpoint=str(cp), checkpoint_sha256=digest(cp), traces=traces,
    motion_output_parameters=sum(p.numel() for p in groups['motion_output']),
    objective='sum legs plus.1sum wheels, existing final output layer only; direct action sensitivity, not native PPO credit or closed-loop benefit',
    interpretation='A zero direct gradient excludes immediate action changes at these recorded states only. Earlier action changes can alter later states. No new controller/network or optimization is introduced.',
    new_PPO=0, optimizer_calls=0, actual_experiment_consumed=2638, repair_optimizer_updates=258,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R/'motion_output_scope001.json', out)
print(json.dumps(out, indent=2))
assert all(checks.values()), checks

"""Recheck the zero extension against the completed1296 weights and native inputs."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
torch.set_num_threads(2)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
seal_path = HERE.parent / 'static_teacher0_1296_completion001.json'
queue_path = RUN / 'static_teacher0_1296_queue001.json'
seal, queue = read(seal_path), read(queue_path)
out = HERE / 'actual1296_cpu002.json'
if out.exists():
    raise SystemExit('Preserve the completed exact-parent extension proof')
if (not seal['passed'] or seal['queue_sha256'] != sha(queue_path) or queue['status'] != 'complete'
        or any(Path('/proc/' + str(queue[k])).exists() for k in ('pid', 'worker_pid'))):
    raise SystemExit('Wait for the actual1296 native trial and all replay owners to exit')
path = RUN / 'pilot026/model_001296.pt'
if sha(path) != seal['checkpoint_sha256']:
    raise SystemExit('Use the exact sealed1296 parent')
proof = read(HERE / 'integration_cpu002.json')
helper = HERE / 'src/wheeled_algo/wheel_authority.py'
if not proof['passed'] or sha(helper) != proof['candidate_source_sha256']['src/wheeled_algo/wheel_authority.py']:
    raise SystemExit('The CPU-reviewed candidate helper changed')
spec = importlib.util.spec_from_file_location('actual1296_wheel_authority', helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parent = torch.load(path, map_location='cpu', weights_only=True)
mean = mean_from_checkpoint(parent, ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx')
child = module.WheelAuthorityActor(mean)


class Wrapped(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.distribution = nn.Module()
        self.distribution.register_parameter('std_param', nn.Parameter(
            parent['actor_state_dict']['distribution.std_param'].clone(), requires_grad=False))
        self.mlp = value


old_actor, actor = Wrapped(mean), Wrapped(child)
critic = nn.Module()
critic.mlp = nn.Sequential(nn.Linear(83, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                          nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
critic.load_state_dict(parent['critic_state_dict'])
optimizer = torch.optim.Adam([dict(params=actor.parameters(), lr=.0003), dict(params=critic.parameters(), lr=.0005)])
rng = torch.get_rng_state().clone()
transformed, extension = module.extend_wheel_authority_parent(parent, actor, critic, optimizer,
                                                             list(dict(old_actor.named_parameters())))
old_names, new_names = list(dict(old_actor.named_parameters())), list(dict(actor.named_parameters()))
old_ids = dict(zip(old_names, parent['optimizer_state_dict']['param_groups'][0]['params'], strict=True))
new_ids = dict(zip(new_names, transformed['optimizer_state_dict']['param_groups'][0]['params'], strict=True))


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


checks = dict(extension_preserves_RNG=torch.equal(rng, torch.get_rng_state()),
    all_old_actor_state_exact=all(torch.equal(v, transformed['actor_state_dict'][n])
                                  for n, v in parent['actor_state_dict'].items()),
    all_other_parent_fields_exact=all(exact(v, transformed[k]) for k, v in parent.items()
                                     if k not in ('actor_state_dict', 'optimizer_state_dict')),
    all_old_actor_Adam_exact=all(exact(parent['optimizer_state_dict']['state'].get(i),
        transformed['optimizer_state_dict']['state'].get(new_ids[n])) for n, i in old_ids.items()),
    critic_Adam_exact=all(exact(parent['optimizer_state_dict']['state'].get(a),
        transformed['optimizer_state_dict']['state'].get(b)) for a, b in zip(
        parent['optimizer_state_dict']['param_groups'][1]['params'],
        transformed['optimizer_state_dict']['param_groups'][1]['params'], strict=True)))
traces = {}
for cohort in seal['cohorts']:
    p = RUN / cohort / 'traces.npz'
    trace = np.load(p)
    x = torch.from_numpy(trace['observations'][trace['active'].astype(bool)])
    static = np.isin(trace['groups'], ('stand_230mm', 'stand_305mm', 'stand_430mm', 'height_scan'))
    stationary = torch.from_numpy(trace['observations'][:, static][trace['active'][:, static].astype(bool)])
    with torch.inference_mode():
        identity = torch.equal(child(x), mean(x))
        zero = bool((child.authority_gate(stationary) == 0.).all())
    checks[cohort + '_all_active_initial_function_exact'] = identity
    checks[cohort + '_static_gate_zero'] = zero
    traces[cohort] = dict(sha256=sha(p), active_samples=len(x), static_samples=len(stationary))
    trace.close()
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
out.write_text(json.dumps(dict(schema='flat-wheel-authority-actual1296-zero-extension-CPU-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
    parent_checkpoint_sha256=sha(path), completion_sha256=sha(seal_path), helper_sha256=sha(helper),
    trace_proofs=traces, extension_contract=extension, adopted=False, new_optimizer_updates=0,
    new_native_steps=0, behavior_qualified=False), indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=True, checks=len(checks), new_optimizer_updates=0), indent=2))

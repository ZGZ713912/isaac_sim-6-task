"""Verify actual-parent Adam rate migration and full4096 static training contract."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / 'wheel_rate_candidate001'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.flat_optimizer_rates import apply_flat_optimizer_rates
from wheeled_tasks.chassis.flat_base_continuation import read_plan, write_bundle
from wheeled_tasks.chassis.preflight import design_preflight

torch.set_num_threads(1)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
out = CANDIDATE / 'integration_cpu002.json'
if out.exists() or (CANDIDATE / 'config4096_002').exists():
    raise SystemExit('Preserve completed CPU integration outputs')
candidate_path = CANDIDATE / 'cpu001.json'
candidate = read(candidate_path)
plan = read_plan(ROOT, candidate['plan_file'])
parent_path = ROOT / candidate['parent_checkpoint']
parent = torch.load(parent_path, map_location='cpu', weights_only=True)
source = ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx'
mean = mean_from_checkpoint(parent, source)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.distribution = nn.Module()
        self.distribution.register_parameter('std_param', nn.Parameter(
            parent['actor_state_dict']['distribution.std_param'].clone()))
        self.mlp = mean


actor = Actor()
critic = nn.Module()
critic.mlp = nn.Sequential(nn.Linear(83, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
    nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
critic.load_state_dict(parent['critic_state_dict'])
optimizer = torch.optim.Adam([dict(params=actor.parameters(), name='actor', lr=.001),
    dict(params=critic.parameters(), name='critic', lr=.0005)])
optimizer.load_state_dict(parent['optimizer_state_dict'])


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


restored = exact(optimizer.state_dict(), parent['optimizer_state_dict'])
expected = deepcopy(parent['optimizer_state_dict'])
expected['param_groups'][0]['lr'] = .001
rate = apply_flat_optimizer_rates(optimizer, .001, .0005, parent_rates=(.0003, .0005))
worker, seed = write_bundle(ROOT, CANDIDATE / 'config4096_002', 4096, plan_file=candidate['plan_file'])
preflight = design_preflight(worker, read(ROOT / plan['control_file']), ROOT,
    stage='speed', role='train', seed=seed, num_envs=4096)
old_worker = read(ROOT / 'runs/flat_base_20261019/pilot027_from1280/config/contract.json')
worker_changes = sorted(k for k in set(old_worker) | set(worker) if old_worker.get(k) != worker.get(k))
checks = dict(
    authenticated_parent=sha(parent_path) == candidate['parent_checkpoint_sha256'],
    full_real_Adam_restore_exact=restored,
    only_actor_lr_changes_in_real_Adam=exact(optimizer.state_dict(), expected),
    actor_state_exact=exact(actor.state_dict(), parent['actor_state_dict']),
    critic_state_exact=exact(critic.state_dict(), parent['critic_state_dict']),
    production_static_preflight=preflight['passed'],
    only_rate_and_plan_marker_change_worker=worker_changes == ['flat_base_continuation', 'learning_rate'],
    all_execution_sources_exact=all(sha(ROOT / n) == v for n, v in candidate['source_sha256'].items()),
    new_plan_exact=sha(ROOT / candidate['plan_file']) == candidate['plan_sha256'],
)
if not all(checks.values()):
    raise SystemExit(json.dumps(dict(checks=checks, worker_changes=worker_changes), indent=2))
out.write_text(json.dumps(dict(schema='flat-wheel-rate-real-parent-integration-CPU-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
    candidate_sha256=sha(candidate_path), parent_checkpoint_sha256=sha(parent_path),
    plan_sha256=candidate['plan_sha256'], verifier_sha256=sha(Path(__file__)),
    preflight=preflight, changed_worker_fields=worker_changes, actual_rate_contract=rate,
    source_sha256=candidate['source_sha256'], new_optimizer_updates=0, new_native_steps=0,
    behavior_qualified=False), indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=True, checks=len(checks), changed_worker_fields=worker_changes), indent=2))

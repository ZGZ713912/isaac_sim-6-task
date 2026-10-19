"""Exercise staged v6 contract, loader and actual saved actor/critic Adam mapping."""
from copy import deepcopy
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
torch.set_num_threads(2)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


command = module('wheeled_algo.command_residual', HERE / 'src/wheeled_algo/command_residual.py')
command.__file__ = str(ROOT / 'src/wheeled_algo/command_residual.py')
wheel = module('wheeled_algo.wheel_authority', HERE / 'src/wheeled_algo/wheel_authority.py')
flat = module('wheeled_tasks.chassis._wheel_candidate', HERE / 'src/wheeled_tasks/chassis/flat_base_continuation.py')
plan_path = HERE / 'plan_wheel.json'
plan = flat.read_plan(ROOT, plan_path)
worker, seed = flat.write_bundle(ROOT, HERE / 'config4096', 4096, plan_file=plan_path)
from wheeled_tasks.chassis.preflight import design_preflight
preflight = design_preflight(worker, json.loads((ROOT / plan['control_file']).read_text()), ROOT,
                             stage='speed', role='train', seed=seed, num_envs=4096)
source = ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx'
parent_path = ROOT / 'runs/flat_base_20261019/pilot026/model_001281.pt'
parent = torch.load(parent_path, map_location='cpu', weights_only=True)
mean = command.mean_from_checkpoint(parent, source)
child = wheel.WheelAuthorityActor(mean)


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
transformed, extension = wheel.extend_wheel_authority_parent(parent, actor, critic, optimizer,
                                                           list(dict(old_actor.named_parameters())))
actor.load_state_dict(transformed['actor_state_dict'])
optimizer.load_state_dict(transformed['optimizer_state_dict'])
proof = wheel.configure_wheel_authority_training(actor, True)
transformed['infos']['actor_adaptation'] = wheel.KIND
loaded = command.mean_from_checkpoint(transformed, source)
x = torch.zeros(256, 35)
x[:, 3], x[:, 9], x[:, 28] = 1.525, -1., 1.
x[:64, 0], x[64:128, 0], x[128:192, 2] = 3., -3., 4.
x[192:, 20:22] = torch.tensor([3., -3.])
with torch.inference_mode():
    checks = dict(independent4096_contract_passed=preflight['status'] == 'static_passed',
        exactly994_trainable=proof['trainable_parameters'] == 994,
        only_new_head_trainable=all(n.startswith('mlp.wheel_authority.') for n in proof['trainable_names']),
        exact_parent_function_at_entry=torch.equal(child(x), mean(x)),
        exact_checkpoint_loader=torch.equal(loaded(x), child(x)),
        old_actor_state_exact=all(torch.equal(v, transformed['actor_state_dict'][n])
                                  for n, v in parent['actor_state_dict'].items()))
bad = deepcopy(transformed)
bad['actor_state_dict']['mlp.authority_bound'][0] *= 2
try:
    command.mean_from_checkpoint(bad, source)
except ValueError:
    checks['changed_authority_bound_rejected'] = True
else:
    checks['changed_authority_bound_rejected'] = False
bad = deepcopy(transformed)
bad['actor_state_dict']['mlp.static_feedback.2.bias'][0] += .01
try:
    command.mean_from_checkpoint(bad, source)
except ValueError:
    checks['changed_static_head_rejected'] = True
else:
    checks['changed_static_head_rejected'] = False
wheel.configure_wheel_authority_training(actor, False)
checks['critic_only_mode_freezes_entire_actor'] = not any(p.requires_grad for p in actor.parameters())
old_tree = ast.parse((ROOT / 'scripts/train_flat_base.py').read_text())
new_tree = ast.parse((HERE / 'scripts/train_flat_base.py').read_text())
for name in ('step', 'update', 'save_checkpoint'):
    old = [n for n in ast.walk(old_tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    new = [n for n in ast.walk(new_tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    checks[name + '_AST_exact'] = len(old) == len(new) == 1 and ast.dump(old[0]) == ast.dump(new[0])
old_plan = json.loads((ROOT / parent['infos']['plan_file']).read_text())
checks['only_actor_kind_changed_in_plan'] = sorted(k for k in old_plan.keys() | plan.keys()
    if old_plan.get(k) != plan.get(k)) == ['actor_adaptation']
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
out = HERE / 'integration_cpu001.json'
if out.exists():
    raise SystemExit('Preserve completed integration proof')
out.write_text(json.dumps(dict(schema='flat-wheel-authority-candidate-integration-CPU-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
    parent_checkpoint_sha256=sha(parent_path), candidate_plan_sha256=sha(plan_path),
    candidate_source_sha256={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py') if 'parent_source' not in p.parts},
    static_preflight=preflight, extension_contract=extension,
    adopted=False, new_native_steps=0, new_optimizer_updates=0, behavior_qualified=False), indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=True, checks=len(checks), adopted=False), indent=2))

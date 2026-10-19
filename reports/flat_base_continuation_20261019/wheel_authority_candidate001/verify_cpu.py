"""Check zero extension, frozen low commands and actual Adam ownership on CPU."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
spec = importlib.util.spec_from_file_location('wheel_authority_candidate', HERE / 'wheel_authority.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(2)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
path = ROOT / 'runs/flat_base_20261019/pilot026/initial_state.pt'
parent = torch.load(path, map_location='cpu', weights_only=True)
source = ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx'
mean = mean_from_checkpoint(parent, source)
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


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


old_names = list(dict(old_actor.named_parameters()))
new_names = list(dict(actor.named_parameters()))
old_ids = dict(zip(old_names, parent['optimizer_state_dict']['param_groups'][0]['params'], strict=True))
new_ids = dict(zip(new_names, transformed['optimizer_state_dict']['param_groups'][0]['params'], strict=True))
old_slots, new_slots = parent['optimizer_state_dict']['state'], transformed['optimizer_state_dict']['state']
checks = dict(extension_preserves_RNG=torch.equal(rng, torch.get_rng_state()),
    all_old_actor_state_exact=all(torch.equal(v, transformed['actor_state_dict'][n])
                                  for n, v in parent['actor_state_dict'].items()),
    all_old_actor_Adam_exact=all(exact(old_slots.get(i), new_slots.get(new_ids[n])) for n, i in old_ids.items()),
    critic_Adam_exact=all(exact(old_slots.get(a), new_slots.get(b)) for a, b in zip(
        parent['optimizer_state_dict']['param_groups'][1]['params'],
        transformed['optimizer_state_dict']['param_groups'][1]['params'], strict=True)),
    new_head_no_Adam=all(new_ids[n] not in new_slots for n in new_names if n not in old_ids),
    all_other_parent_fields_exact=all(exact(v, transformed[k]) for k, v in parent.items()
                                     if k not in ('actor_state_dict', 'optimizer_state_dict')),
    exactly994_trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad) == 994)
samples, statics, high_reverse = [], [], []
for cohort in ('static_recall1280_basic001', 'static_recall1280_default_envelope001',
               'static_recall1280_maximum_envelope001'):
    trace = np.load(ROOT / 'runs/flat_base_20261019' / cohort / 'traces.npz')
    active = trace['active'].astype(bool)
    obs = trace['observations']
    samples.append(obs[active][::17].copy())
    stationary = np.isin(trace['groups'], ('stand_230mm', 'stand_305mm', 'stand_430mm', 'height_scan'))
    statics.append(obs[:, stationary][active[:, stationary]].copy())
    if cohort.endswith('maximum_envelope001'):
        lanes = trace['groups'] == 'backward_3'
        high_reverse.append(obs[:, lanes][active[:, lanes]][1000:1800].copy())
    trace.close()
x = torch.from_numpy(np.concatenate(samples))
stationary = torch.from_numpy(np.concatenate(statics))
with torch.inference_mode():
    checks['zero_extension_bitexact_on_recorded_states'] = torch.equal(child(x), mean(x))
    checks['recorded_static_and_height_scan_all_gate_zero'] = bool((child.authority_gate(stationary) == 0.).all())
    stationary_before = child(stationary).clone()
moving = torch.from_numpy(np.concatenate(high_reverse))
braking = moving.clone()
braking[:, :3] = 0.
braking[:, 20:22] = torch.tensor([2., -2.])
low = moving.clone()
low[:, 0], low[:, 2] = -.5, 0.
checks['low_moving_gate_zero_even_with_high_encoders'] = bool((child.authority_gate(low) == 0.).all())
checks['braking_has_visible_gate'] = bool((child.authority_gate(braking) == 1.).all())
for frame in (moving, braking):
    child.zero_grad(set_to_none=True)
    (child(frame)[:, 4:6] - mean(frame)[:, 4:6].detach() - .03).square().mean().backward()
    checks['moving_head_gradient' if frame is moving else 'braking_head_gradient'] = any(
        p.grad is not None and p.grad.count_nonzero() for p in child.wheel_authority.parameters())
    checks['all_existing_heads_frozen_' + ('moving' if frame is moving else 'braking')] = all(
        p.grad is None and not p.requires_grad for n, p in child.named_parameters() if not n.startswith('wheel_authority.'))
with torch.no_grad():
    child.wheel_authority[2].bias.add_(torch.tensor([-.3, .3]))
    child.wheel_authority[2].weight.normal_(std=.02)
    checks['static_outputs_bitexact_after_new_head_change'] = torch.equal(child(stationary), stationary_before)
    checks['correction_bounded_per_wheel'] = bool((child.authority_actions(x)[:, 4:6].abs() <= 6.).all())
    checks['new_head_keeps_all_leg_outputs_exact'] = torch.equal(child(x)[:, :4], mean(x)[:, :4])
graph = child.export_graph(source)
options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
session = ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
with torch.inference_mode():
    expected = child(x).numpy()
error = float(np.max(np.abs(session.run(None, {session.get_inputs()[0].name:x.numpy()})[0] - expected)))
checks['ONNX_matches_diverged_cpu_head'] = error < 5e-6
original = onnx.load(source)
checks['original_ONNX_initializers_byte_exact'] = all(a.SerializeToString() == b.SerializeToString()
    for a, b in zip(original.graph.initializer, graph.graph.initializer))
if not all(bool(v) for v in checks.values()):
    raise SystemExit(json.dumps({k:bool(v) for k,v in checks.items()}, indent=2))
out = HERE / 'cpu001.json'
if out.exists():
    raise SystemExit('Preserve prior candidate proof')
out.write_text(json.dumps(dict(schema='flat-wheel-authority-zero-extension-CPU-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks={k:bool(v) for k,v in checks.items()},
    helper_sha256=sha(HERE / 'wheel_authority.py'), parent_checkpoint_sha256=sha(path),
    all_recorded_state_samples=len(x), zero_gate_static_and_height_samples=len(stationary),
    CPU_ONNX_max_abs_error=error, extension_contract=extension,
    adopted=False, new_optimizer_updates=0, new_native_steps=0,
    evaluation_traces_used_for_optimization=False,
    scope='Function/ownership diagnostics only. Virtual output perturbations were discarded; no model exported or saved. Observed static invariance is conditional on this visible gate, not all future trajectories.',
    behavior_qualified=False), indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=True, checks=len(checks), CPU_ONNX_max_abs_error=error,
                     zero_gate_static_samples=len(stationary)), indent=2))

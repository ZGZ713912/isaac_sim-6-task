"""Bind one low-friction static-only pilot to the sealed1328 full state."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.static_grip import StaticGripActor, KIND
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.flat_base_continuation import digest, encode, read_plan, write_bundle
from wheeled_tasks.chassis.preflight import design_preflight

torch.set_num_threads(1)
OUT = Path(__file__).with_name('static_grip_candidate001')
if OUT.exists():
    raise SystemExit('Preserve existing static grip candidate')
parent_path = ROOT / 'runs/flat_base_20261019/pilot028_wheel_rate1328/model_001328.pt'
assert digest(parent_path) == '5e2320c50b8fd7982b278dfb4d21cc4763cf12be3cf10193bbf630c648bf672b'
checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
info = checkpoint['infos']
assert info['successful_updates_total'] == 1328 and info['critic_warmup_remaining'] == 0
old = read_plan(ROOT, info['plan_file'])
assert digest(ROOT / info['plan_file']) == info['plan_sha256']
plan = deepcopy(read_plan(ROOT, 'contracts/flat_base_finetune_low_friction_v1.json'))
plan.update(actor_adaptation=KIND, learning_rate=.0001, additional_critic_warmup_updates=2,
    training_scope='low_grip_static_feedback_only_pilot_preserve_all_moving_heads')
source = ROOT / plan['actor_bundle'] / 'source.onnx'
parent = mean_from_checkpoint(checkpoint, source)
child = StaticGripActor(parent)
assert list(parent.state_dict()) == list(child.state_dict())
assert list(dict(parent.named_parameters())) == list(dict(child.named_parameters()))
assert all(torch.equal(v, child.state_dict()[n]) for n, v in parent.state_dict().items())
trace_path = ROOT / 'runs/flat_base_20261019/low_friction1328_basic001/traces.npz'
traces = np.load(trace_path)
obs = torch.from_numpy(traces['observations'].reshape(-1, 35).copy())
active = torch.from_numpy(traces['active'].reshape(-1).copy()).bool()
obs = obs[active]
with torch.no_grad():
    assert torch.equal(parent(obs), child(obs))
static = child.moving_mix(obs).squeeze(-1) == 0
assert int(static.sum()) >= 6000
loss = child(obs[static][:6000]).square().mean()
loss.backward()
gradient = {name: float(p.grad.norm()) for name, p in child.named_parameters() if p.requires_grad}
assert len(gradient) == 4 and all(value > 0 for value in gradient.values())
assert all(p.grad is None for name, p in child.named_parameters() if not name.startswith('static_feedback.'))
OUT.mkdir()
plan_path = OUT / 'plan.json'
plan_path.write_bytes(encode(plan))
assert read_plan(ROOT, str(plan_path.relative_to(ROOT))) == plan
config, seed = write_bundle(ROOT, OUT / 'config4096', 4096, plan_file=str(plan_path.relative_to(ROOT)))
static_preflight = design_preflight(config, load_chassis_control(ROOT / config['control_math_source']),
    ROOT, stage='speed', role='train', num_envs=4096, seed=seed)
sources = set(info['source_sha256'])
sources.update(str(path.relative_to(ROOT)) for path in (ROOT / 'src/wheeled_tasks/chassis').glob('*.py'))
sources.add('src/wheeled_algo/static_grip.py')
source_hashes = {name: digest(ROOT / name) for name in sorted(sources)}
changed = sorted(key for key in old.keys() | plan.keys() if old.get(key) != plan.get(key))
revision = dict(schema='flat-training-plan-revision-v1', parent_checkpoint_sha256=digest(parent_path),
    old_plan_file=info['plan_file'], old_plan_sha256=info['plan_sha256'], new_plan_sha256=digest(plan_path),
    old_source_sha256=info['source_sha256'], new_source_sha256=source_hashes, changed_plan_fields=changed)
revision_path = OUT / 'revision.json'
revision_path.write_bytes(encode(revision))
receipt = dict(schema='flat-static-grip-preparation-v1', recorded_at=datetime.now(timezone.utc).isoformat(),
    passed=True, parent_checkpoint=str(parent_path.relative_to(ROOT)), parent_checkpoint_sha256=digest(parent_path),
    parent_successful_updates=1328, parent_actor_updates=info['actor_updates_total'],
    plan_file=str(plan_path.relative_to(ROOT)), plan_sha256=digest(plan_path),
    revision_sha256=digest(revision_path), source_sha256=source_hashes,
    changed_plan_fields=changed, static_preflight=static_preflight,
    actual_input_probe=dict(trace_sha256=digest(trace_path), active_frames=len(obs), static_frames=int(static.sum()),
        full_parent_function_equal=True, unchanged_parameter_layout=True, static_gradient_norms=gradient,
        scope='Identity and trainability only; squared-output diagnostic is not a training loss or behavior proof'),
    pilot=dict(critic_warmup_updates=2, actor_updates=8, total_optimizer_updates=10,
        population=4096, policy_hz=50, physics_hz=1000, actor_learning_rate=.0001,
        expected_final_successful_updates=1338, expected_final_actor_updates=info['actor_updates_total'] + 8,
        inherited_optimizer_updates_consumed=info['consumed_optimizer_updates_total'],
        frozen='original, command, moving feedback, fast feedback, wheel authority, std and their Adam',
        admitted_scope='one bounded pilot only; fixed low-material basic evaluation follows; no long run'),
    preparer_sha256=digest(Path(__file__)), new_optimizer_updates=0, qualified=False,
    hardware_motion_started=False)
(OUT / 'preparation001.json').write_bytes(encode(receipt))
for name, value in source_hashes.items():
    path = OUT / 'source' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((ROOT / name).read_bytes())
print(json.dumps(dict(passed=True, changed=changed, static_gradient_norms=gradient,
    trainable_parameters=sum(p.numel() for p in child.parameters() if p.requires_grad))))

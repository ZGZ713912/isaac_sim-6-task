"""Bind one motion-only low-friction pilot to the sealed1338 full state."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.motion_grip import MotionGripActor, KIND
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.flat_base_continuation import digest, encode, read_plan, write_bundle
from wheeled_tasks.chassis.preflight import design_preflight

torch.set_num_threads(1)
OUT = HERE / 'motion_grip_candidate001'
plan_path = OUT / 'plan.json'
if plan_path.exists() or (OUT / 'preparation001.json').exists():
    raise SystemExit('Preserve existing native motion preparation')
envelope_path = HERE / 'static_grip_candidate002/envelopes_completion001.json'
envelopes = json.loads(envelope_path.read_text())
if (not envelopes['execution_verified'] or not all(envelopes['checks'].values())
        or not envelopes['no_lost_parent_checks']):
    raise SystemExit('Require all four sealed1338 replay cohorts')
queue = json.loads((ROOT / 'runs/flat_base_20261019/static_grip1338_envelopes_queue001.json').read_text())
if queue['status'] != 'complete' or any(Path('/proc/' + str(queue[key])).exists() for key in ('pid', 'worker_pid')):
    raise SystemExit('Wait for all old replay owners to exit')
parent_path = ROOT / 'runs/flat_base_20261019/pilot029_static_grip1338/model_001338.pt'
assert digest(parent_path) == '39f7f2d364d82905741478fd254c7fdde3d9c101a613cfba1ba79d14a483fd73'
checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
info = checkpoint['infos']
assert info['successful_updates_total'] == 1338 and info['critic_warmup_remaining'] == 0
old = read_plan(ROOT, info['plan_file'])
assert digest(ROOT / info['plan_file']) == info['plan_sha256']
plan = deepcopy(old)
plan.update(actor_adaptation=KIND, additional_critic_warmup_updates=2,
    motion_grip_parent=dict(checkpoint=str(parent_path.relative_to(ROOT)), sha256=digest(parent_path)),
    training_scope='low_grip_motion_feedback_and_wheel_adaptation_preserve_adapted_static')
assert plan == json.loads((OUT / 'proposed_plan.json').read_text())
parent = mean_from_checkpoint(checkpoint, ROOT / plan['actor_bundle'] / 'source.onnx')
child = MotionGripActor(parent, digest(parent_path))
assert list(parent.state_dict()) == list(child.state_dict())
assert list(dict(parent.named_parameters())) == list(dict(child.named_parameters()))
assert all(torch.equal(value, child.state_dict()[name]) for name, value in parent.state_dict().items())
baseline_path = ROOT / 'runs/flat_base_20261019/ground025_static1338_basic001/evaluation.json'
baseline = json.loads(baseline_path.read_text())
assert baseline['status'] == 'evaluated' and baseline['checkpoint_sha256'] == digest(parent_path)
assert all(row['passed'] for row in baseline['result']['cases'].values())
probe_trace = ROOT / 'runs/flat_base_20261019/ground025_static1338_default_envelope001/traces.npz'
with np.load(probe_trace) as trace:
    obs = trace['observations'][trace['active'].astype(bool)]
    selected = torch.from_numpy(obs[np.linspace(0, len(obs)-1, min(4096, len(obs)), dtype=int)].copy())
with torch.no_grad():
    assert torch.equal(parent(selected), child(selected))
loss = (child(selected) - child(selected).detach() - .01).square().mean()
loss.backward()
gradients = {name:float(p.grad.norm()) for name,p in child.named_parameters() if p.requires_grad}
assert len(gradients) == 12 and all(value > 0 for value in gradients.values())
plan_path.write_bytes(encode(plan))
assert read_plan(ROOT, str(plan_path.relative_to(ROOT))) == plan
config, seed = write_bundle(ROOT, OUT / 'config4096', 4096, plan_file=str(plan_path.relative_to(ROOT)))
preflight = design_preflight(config, load_chassis_control(ROOT / config['control_math_source']),
    ROOT, stage='speed', role='train', num_envs=4096, seed=seed)
sources = set(info['source_sha256'])
sources.update(str(path.relative_to(ROOT)) for path in (ROOT / 'src/wheeled_tasks/chassis').glob('*.py'))
sources.add('src/wheeled_algo/motion_grip.py')
hashes = {name:digest(ROOT / name) for name in sorted(sources)}
changed = sorted(key for key in old.keys() | plan.keys() if old.get(key) != plan.get(key))
revision = dict(schema='flat-training-plan-revision-v1', parent_checkpoint_sha256=digest(parent_path),
    old_plan_file=info['plan_file'], old_plan_sha256=info['plan_sha256'], new_plan_sha256=digest(plan_path),
    old_source_sha256=info['source_sha256'], new_source_sha256=hashes, changed_plan_fields=changed)
revision_path = OUT / 'revision.json'
revision_path.write_bytes(encode(revision))
(OUT / 'preparation001.json').write_bytes(encode(dict(schema='flat-motion-grip-preparation-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True,
    parent_checkpoint=str(parent_path.relative_to(ROOT)), parent_checkpoint_sha256=digest(parent_path),
    parent_successful_updates=1338, parent_actor_updates=info['actor_updates_total'],
    baseline_file=str(baseline_path.relative_to(ROOT)), baseline_sha256=digest(baseline_path),
    full_parent_replay_seal_sha256=digest(envelope_path),
    plan_file=str(plan_path.relative_to(ROOT)), plan_sha256=digest(plan_path), revision_sha256=digest(revision_path),
    source_sha256=hashes, changed_plan_fields=changed, static_preflight=preflight,
    actual_input_probe=dict(entry_function_equal=True, parameter_order_equal=True, frames=len(selected),
        trace_sha256=digest(probe_trace), gradient_norms=gradients,
        scope='Squared-output CPU gradient probe only; not a training loss or behavior proof'),
    pilot=dict(critic_warmup_updates=2, actor_updates=8, total_optimizer_updates=10,
        population=4096, policy_hz=50, physics_hz=1000, actor_learning_rate=.0001,
        expected_final_successful_updates=1348, expected_final_actor_updates=info['actor_updates_total'] + 8,
        inherited_optimizer_updates_consumed=info['consumed_optimizer_updates_total'],
        frozen='original, command, adapted static feedback, std and their Adam',
        admitted_scope='one bounded motion pilot only; complete fixed basic replay follows; no long run'),
    preparer_sha256=digest(Path(__file__)), new_optimizer_updates=0, qualified=False,
    hardware_motion_started=False)))
(OUT / 'prepare_motion_grip.py').write_bytes(Path(__file__).read_bytes())
for name, sha in hashes.items():
    path = OUT / 'source' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((ROOT / name).read_bytes())
print(json.dumps(dict(passed=True, changed=changed, trainable_parameters=3246)))

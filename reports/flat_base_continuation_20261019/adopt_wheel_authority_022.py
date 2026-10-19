"""Adopt only the reviewed zero extension after the1296 trial is fully sealed."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
CANDIDATE = HERE / 'wheel_authority_candidate001'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
out = HERE / 'wheel_authority_plan_revision022.json'
backup = CANDIDATE / 'adoption_parent_source'
if out.exists() or backup.exists() or (ROOT / 'src/wheeled_algo/wheel_authority.py').exists():
    raise SystemExit('Preserve earlier or partial adoption and inspect it explicitly')
seal_path = HERE / 'static_teacher0_1296_completion001.json'
queue_path = RUN / 'static_teacher0_1296_queue001.json'
seal, queue = read(seal_path), read(queue_path)
if (not seal['passed'] or seal['queue_sha256'] != sha(queue_path) or queue['status'] != 'complete'
        or any(Path('/proc/' + str(queue[k])).exists() for k in ('pid', 'worker_pid'))):
    raise SystemExit('Wait for all prior native owners and replay sealing')
if any(sha(ROOT / name) != value for name, value in queue['source_sha256'].items()):
    raise SystemExit('The previously frozen execution source changed before adoption')
cpu_path = CANDIDATE / 'integration_cpu002.json'
cpu = read(cpu_path)
actual_path = CANDIDATE / 'actual1296_cpu002.json'
actual = read(actual_path)
parent_path = RUN / 'pilot026/model_001296.pt'
parent = torch.load(parent_path, map_location='cpu', weights_only=True)
old = parent['infos']
plan_path = CANDIDATE / 'plan_wheel.json'
old_plan, plan = read(ROOT / old['plan_file']), read(plan_path)
changed = sorted(k for k in plan.keys() | old_plan.keys() if plan.get(k) != old_plan.get(k))
if (not cpu['passed'] or not actual['passed'] or actual['completion_sha256'] != sha(seal_path)
        or actual['parent_checkpoint_sha256'] != sha(parent_path) or seal['checkpoint_sha256'] != sha(parent_path)
        or cpu['candidate_plan_sha256'] != sha(plan_path) or changed != ['actor_adaptation']):
    raise SystemExit('Exact-parent CPU proof or actor-only plan migration is missing')
files = [name for name in cpu['candidate_source_sha256'] if name.startswith(('scripts/', 'src/'))]
if len(files) != 6:
    raise SystemExit('Unexpected candidate integration files')
for name in files:
    if sha(CANDIDATE / name) != cpu['candidate_source_sha256'][name]:
        raise SystemExit('Reviewed candidate changed: ' + name)
for name, value in old['source_sha256'].items():
    if sha(ROOT / name) != value:
        raise SystemExit('Parent execution source changed: ' + name)
for name in files:
    destination = ROOT / name
    if destination.exists():
        saved = backup / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination, saved)
for name in files:
    destination = ROOT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((CANDIDATE / name).read_bytes())
new_sources = dict(old['source_sha256'])
for name in ('scripts/train_flat_base.py', 'src/wheeled_algo/command_residual.py',
             'src/wheeled_algo/wheel_authority.py', 'src/wheeled_tasks/chassis/flat_base_continuation.py'):
    new_sources[name] = sha(ROOT / name)
out.write_text(json.dumps(dict(schema='flat-training-plan-revision-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), parent_checkpoint_sha256=sha(parent_path),
    old_plan_file=old['plan_file'], old_plan_sha256=old['plan_sha256'],
    new_plan_file=str(plan_path.relative_to(ROOT)), new_plan_sha256=sha(plan_path),
    old_source_sha256=old['source_sha256'], new_source_sha256=new_sources, changed_plan_fields=changed,
    parent_completion_sha256=sha(seal_path), integration_CPU_sha256=sha(cpu_path),
    actual_parent_CPU_sha256=sha(actual_path),
    integration_source_sha256={name:sha(ROOT / name) for name in files},
    declared_change='Append zero-output high-demand/braking wheel head; train only994 new parameters; freeze every existing actor branch and its Adam.',
    unchanged='Exact initial parent function/weights/std/critic/Adam/RNG/counters;4096 population,reward,groups,clocks,rates,teacher0 and all fixed thresholds.',
    bounded_optimizer_updates=16, additional_critic_warmup_updates=0,
    new_native_steps=0, new_optimizer_updates=0, behavior_qualified=False,
    long_training_started=False, hardware_motion_started=False), indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(adopted=True, receipt_sha256=sha(out), changed_plan_fields=changed,
                     new_optimizer_updates=0), indent=2))

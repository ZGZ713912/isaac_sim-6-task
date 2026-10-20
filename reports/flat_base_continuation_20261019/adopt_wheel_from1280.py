"""Adopt the reviewed v6 sources and bind an offline1280 composition to them."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
CANDIDATE = HERE / 'wheel_authority_candidate001'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
out = HERE / 'wheel_authority_from1280_adoption023.json'
backup = CANDIDATE / 'adoption_from1280_parent_source'
new_parent = RUN / 'pilot025/model_001280_wheel_authority_composed002.pt'
if out.exists() or backup.exists() or new_parent.exists() or (ROOT / 'src/wheeled_algo/wheel_authority.py').exists():
    raise SystemExit('Preserve prior or partial adoption')
queue_path = RUN / 'static_teacher0_1296_queue001.json'
seal_path = HERE / 'static_teacher0_1296_completion001.json'
queue, seal = read(queue_path), read(seal_path)
if (queue['status'] != 'complete' or not seal['passed'] or seal['queue_sha256'] != sha(queue_path)
        or any(Path('/proc/' + str(queue[k])).exists() for k in ('pid', 'worker_pid'))):
    raise SystemExit('All prior owners and replay sealing must finish')
if any(sha(ROOT / n) != value for n, value in queue['source_sha256'].items()):
    raise SystemExit('Prior frozen sources changed')
cpu_path = CANDIDATE / 'integration_cpu003.json'
compose_path = CANDIDATE / 'compose_from1280_cpu001.json'
cpu, proof = read(cpu_path), read(compose_path)
raw_path = RUN / 'pilot025/model_001280_wheel_authority_composed.pt'
origin_path = RUN / 'pilot025/model_001280.pt'
plan_path = CANDIDATE / 'plan_wheel_from1280.json'
if (not cpu['passed'] or not proof['passed'] or proof['composed_checkpoint_sha256'] != sha(raw_path)
        or proof['parent_checkpoint_sha256'] != sha(origin_path) or proof['plan_sha256'] != sha(plan_path)):
    raise SystemExit('Exact composition and current staged-source proof are required')
files = [n for n in cpu['candidate_source_sha256'] if n.startswith(('scripts/', 'src/'))]
if len(files) != 6 or any(sha(CANDIDATE / n) != cpu['candidate_source_sha256'][n] for n in files):
    raise SystemExit('Reviewed integration source changed')
prior = load(RUN / 'pilot026/model_001296.pt')
old_sources = prior['infos']['source_sha256']
if any(sha(ROOT / n) != value for n, value in old_sources.items()):
    raise SystemExit('Current production differs from its last exact source receipt')
for n in files:
    p = ROOT / n
    if p.exists():
        saved = backup / n
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, saved)
for n in files:
    p = ROOT / n
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((CANDIDATE / n).read_bytes())
new_sources = dict(old_sources)
for n in ('scripts/train_flat_base.py', 'src/wheeled_algo/command_residual.py',
          'src/wheeled_algo/wheel_authority.py', 'src/wheeled_tasks/chassis/flat_base_continuation.py'):
    new_sources[n] = sha(ROOT / n)
raw, origin = load(raw_path), load(origin_path)
composed = deepcopy(raw)
plan = read(plan_path)
info = composed['infos']
info.update(schema='flat-base-offline-composition-v1', status='offline_composed', pid=None,
    source_sha256=new_sources, teacher_anchor=deepcopy(plan['teacher_anchor']),
    teacher_anchor_runtime={**info['teacher_anchor_runtime'], 'coefficient':0.},
    new_native_steps=0, new_optimizer_updates=0,
    initialization='offline1280_static_recall_and_zero_wheel_head; no_PPO_or_native_steps',
    origin_checkpoint=str(origin_path.relative_to(ROOT)), origin_checkpoint_sha256=sha(origin_path),
    origin_execution_source_sha256=origin['infos']['source_sha256'],
    composition_artifact_sha256=sha(raw_path),
    source_binding='Execution sources for this explicitly composed artifact; inherited PPO provenance remains in origin_checkpoint.',
    training_release=False, hardware_deployment_ready=False, behavior_qualified=False)
for key in ('plan_revision_sha256', 'changed_plan_fields', 'full_restore_verification',
            'restore_verification_reference', 'entry_revision_sha256'):
    info.pop(key, None)
torch.save(composed, new_parent)
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
source = ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx'
old_mean, new_mean = mean_from_checkpoint(origin, source), mean_from_checkpoint(composed, source)
old_names = ['distribution.std_param'] + ['mlp.' + n for n, _ in old_mean.named_parameters()]
new_names = ['distribution.std_param'] + ['mlp.' + n for n, _ in new_mean.named_parameters()]
old_opt, new_opt = origin['optimizer_state_dict'], composed['optimizer_state_dict']
old_ids = dict(zip(old_names, old_opt['param_groups'][0]['params'], strict=True))
new_ids = dict(zip(new_names, new_opt['param_groups'][0]['params'], strict=True))

def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b

checks = dict(
    existing_Adam_exact=all(exact(old_opt['state'].get(i), new_opt['state'].get(new_ids[n])) for n, i in old_ids.items()),
    critic_Adam_exact=all(exact(old_opt['state'].get(a), new_opt['state'].get(b)) for a, b in zip(
        old_opt['param_groups'][1]['params'], new_opt['param_groups'][1]['params'], strict=True)),
    new_static_and_wheel_heads_have_no_Adam=all(new_ids[n] not in new_opt['state'] for n in new_names if n not in old_ids),
    metadata_binding_only=all(exact(v, composed[k]) for k, v in raw.items() if k != 'infos'),
    exact_execution_sources=all(sha(ROOT / n) == value for n, value in new_sources.items()),
    teacher_coefficient_zero=info['teacher_anchor_runtime']['coefficient'] == plan['teacher_anchor']['coefficient'] == 0.,
    exact_plan=info['plan_sha256'] == sha(plan_path),
    exact1280_origin=info['origin_checkpoint_sha256'] == proof['parent_checkpoint_sha256'],
    no_extra_learning=info['successful_updates_total'] == 1280 and info['new_optimizer_updates'] == 0,
    prior1296_rejected_for_long_training=not seal['long_training_eligible'])
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
out.write_text(json.dumps(dict(schema='flat-offline-composition-source-adoption-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
    original1280_checkpoint_sha256=sha(origin_path), raw_composition_sha256=sha(raw_path),
    parent_checkpoint=str(new_parent.relative_to(ROOT)), parent_checkpoint_sha256=sha(new_parent),
    plan_file=str(plan_path.relative_to(ROOT)), plan_sha256=sha(plan_path),
    old_production_source_sha256=old_sources, new_source_sha256=new_sources,
    integration_source_sha256={n:sha(ROOT / n) for n in files},
    integration_CPU_sha256=sha(cpu_path), composition_CPU_sha256=sha(compose_path),
    prior1296_completion_sha256=sha(seal_path), prior1296_used_as_learning_parent=False,
    changes='Offline1280 composition binds v5 static recall plus zero v6 wheel head to the reviewed sources; teacher anchor0. Only994 appended wheel parameters train in the next bounded block.',
    unchanged='All original1280 actor/critic/std/Adam/RNG/counters retained; static1248 reference exact; reward/population/clocks/rates/thresholds unchanged.',
    new_optimizer_updates=0, new_native_steps=0, bounded_next_updates=16,
    behavior_qualified=False, long_training_started=False, hardware_motion_started=False), indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(passed=True, checks=len(checks), receipt_sha256=sha(out), parent_checkpoint_sha256=sha(new_parent)), indent=2))

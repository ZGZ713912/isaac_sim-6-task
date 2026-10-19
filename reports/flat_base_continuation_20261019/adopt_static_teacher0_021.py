"""Adopt the reviewed static composition only after both native diagnostics exit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
CANDIDATE = HERE / 'static_feedback_v5_candidate001'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
out = HERE / 'static_teacher0_plan_revision021.json'
backup = CANDIDATE / 'adoption_parent_source'
if out.exists() or backup.exists() or (ROOT / 'src/wheeled_algo/static_feedback.py').exists():
    raise SystemExit('Preserve earlier or partial adoption; inspect it explicitly')
reviews = {}
for suffix in ('envelope', 'maximum'):
    queue_path = RUN / f'static_recall1280_{suffix}_queue001.json'
    review_path = HERE / f'static_recall1280_{suffix}_review001.json'
    queue, review = read(queue_path), read(review_path)
    if (queue['status'] != 'complete' or not review['passed'] or review['queue_sha256'] != sha(queue_path)
            or not review['comparisons']['parent1280']['no_lost_baseline_checks']
            or any(Path('/proc/' + str(queue[k])).exists() for k in ('pid', 'worker_pid'))):
        raise SystemExit('Both unchanged-horizon native diagnostics must be sealed with all owners out')
    if any(sha(ROOT / name) != value for name, value in queue['source_sha256'].items()):
        raise SystemExit('Frozen diagnostic execution source changed before adoption')
    reviews[suffix] = sha(review_path)
basic_path = HERE / 'static_recall1280_diagnostic_review001.json'
basic = read(basic_path)
if not basic['passed'] or basic['passed_cases'] != basic['requested_cases']:
    raise SystemExit('The separate basic diagnostic must pass')
parent_path = RUN / 'pilot025/model_001280.pt'
parent = torch.load(parent_path, map_location='cpu', weights_only=True)
old = parent['infos']
old_plan_path = ROOT / old['plan_file']
plan_path = CANDIDATE / 'plan_static_teacher0.json'
plan, old_plan = read(plan_path), read(old_plan_path)
changed = sorted(k for k in plan.keys() | old_plan.keys() if plan.get(k) != old_plan.get(k))
if changed != ['actor_adaptation', 'static_feedback_recall', 'teacher_anchor']:
    raise SystemExit('Unexpected plan intervention')
integration_path = CANDIDATE / 'integration_cpu001.json'
cpu, plan_cpu = read(integration_path), read(CANDIDATE / 'teacher0_plan_cpu001.json')
ast = read(CANDIDATE / 'learning_path_ast001.json')
if (not all(x['passed'] for x in (cpu, plan_cpu, ast))
        or cpu['parent_checkpoint_sha256'] != sha(parent_path)
        or plan_cpu['candidate_plan_sha256'] != sha(plan_path)
        or sha(old_plan_path) != old['plan_sha256']):
    raise SystemExit('CPU evidence or plans differ from this exact parent')
files = [name for name in cpu['candidate_source_sha256'] if name.startswith(('scripts/', 'src/'))]
if len(files) != 6:
    raise SystemExit('Unexpected candidate integration files')
for name in files:
    if sha(CANDIDATE / name) != cpu['candidate_source_sha256'][name]:
        raise SystemExit('Reviewed candidate changed: ' + name)
for name, value in old['source_sha256'].items():
    if sha(ROOT / name) != value:
        raise SystemExit('Parent training source changed before the declared migration: ' + name)
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
             'src/wheeled_algo/static_feedback.py', 'src/wheeled_tasks/chassis/flat_base_continuation.py'):
    new_sources[name] = sha(ROOT / name)
if any(sha(ROOT / name) != value for name, value in new_sources.items()):
    raise SystemExit('Post-adoption source verification failed')
receipt = dict(schema='flat-training-plan-revision-v1', recorded_at=datetime.now(timezone.utc).isoformat(),
    parent_checkpoint_sha256=sha(parent_path), old_plan_file=old['plan_file'], old_plan_sha256=old['plan_sha256'],
    new_plan_file=str(plan_path.relative_to(ROOT)), new_plan_sha256=sha(plan_path),
    old_source_sha256=old['source_sha256'], new_source_sha256=new_sources,
    changed_plan_fields=changed, native_diagnostic_review_sha256=reviews,
    basic_review_sha256=sha(basic_path), integration_CPU_sha256=sha(integration_path),
    plan_CPU_sha256=sha(CANDIDATE / 'teacher0_plan_cpu001.json'),
    learning_path_CPU_sha256=sha(CANDIDATE / 'learning_path_ast001.json'),
    integration_source_sha256={name: sha(ROOT / name) for name in files},
    bounded_optimizer_updates=16, additional_critic_warmup_updates=0,
    unchanged='Existing moving actor/std/critic/Adam/RNG/lifetime consumption;4096 population,reward,groups,clocks,rates and original acceptance thresholds.',
    declared_function_change='Append frozen1248 static feedback with a continuous command transition; zero commands use it, moving endpoint uses exact1280 v4 feedback.',
    learning_change='Set teacher coefficient .02 to0; all mask/scales/source stay declared. Static function is frozen during the trial.',
    causal_scope='Bounded moving-feedback optimization after separately validated static composition; no isolated paired teacher-effect claim.',
    new_native_steps=0, new_optimizer_updates=0, behavior_qualified=False,
    long_training_started=False, hardware_motion_started=False)
out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(adopted=True, receipt_sha256=sha(out), source_files=len(new_sources),
    changed_plan_fields=changed, new_optimizer_updates=0), indent=2))

"""Explicit, reversible source adoption after the old native queue is sealed."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
write = lambda p,x: p.write_text(json.dumps(x,indent=2,allow_nan=False) + '\n')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage',choices=('install','seal'))
args = parser.parse_args()
queue = read(RUN / 'feedback1024_queue001.json')
completion_path = HERE / 'feedback1024_queue_completion001.json'
if (queue['status'] != 'complete' or not completion_path.exists()
        or any(Path('/proc/' + str(queue[k])).exists() for k in ('pid','worker_pid'))):
    raise SystemExit('The1024 queue and last native worker must finish, exit and be sealed first')
completion = read(completion_path)
if not completion['passed'] or completion['queue_sha256'] != sha(RUN / 'feedback1024_queue001.json'):
    raise SystemExit('The1024 completion receipt changed or failed')
candidate_path = HERE / 'clock_candidate_v15_source002.json'
candidate = read(candidate_path)
tree = HERE / 'clock_v15_source_candidate'
if any(sha(tree / n) != v for n,v in candidate['candidate_source_sha256'].items()):
    raise SystemExit('The verified isolated clock source changed')
prerequisites = ['clock_candidate_v15_cpu002.json','clock_candidate_v15_tests001.json',
    'clock_candidate_v15_export_cpu001.json','clock_candidate_v15_reference_cpu001.json']
if not all(read(HERE / n)['passed'] for n in prerequisites):
    raise SystemExit('A candidate CPU review failed')
support = read(HERE / 'clock_candidate_v15_support001.json')
if any(sha(ROOT / n) != v for n,v in support['unchanged_dependency_sha256'].items()):
    raise SystemExit('A byte-identical pure-math support dependency changed')
plan_target = ROOT / 'contracts/flat_base_finetune_v15.json'
if args.stage == 'install':
    if (HERE / 'clock_v15_install001.json').exists() or plan_target.exists():
        raise SystemExit('Preserve the previous clock source installation')
    if any(sha(ROOT / n) != v for n,v in queue['source_sha256'].items()):
        raise SystemExit('The completed123-source freeze changed before installation')
    baseline = read(HERE / 'clock_candidate_v15_base_freeze001.json')
    if baseline['candidate_sha256'] != sha(candidate_path):
        raise SystemExit('Candidate changed after baseline authentication')
    for n,value in baseline['source_sha256'].items():
        path = ROOT / n
        if (value is None and path.exists()) or (value is not None and (not path.exists() or sha(path) != value)):
            raise SystemExit('A baseline source changed before installation: ' + n)
    backup = HERE / 'clock_v15_pre_adoption_source'
    backup.mkdir()
    for n in candidate['changed_files']:
        target = ROOT / n
        if target.exists():
            saved = backup / n
            saved.parent.mkdir(parents=True,exist_ok=True)
            saved.write_bytes(target.read_bytes())
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((tree / n).read_bytes())
    plan_target.write_bytes((HERE / 'clock_candidate_v15.json').read_bytes())
    assert sha(plan_target) == candidate['candidate_plan_sha256']
    write(HERE / 'clock_v15_install001.json',dict(schema='flat-clock-v15-source-install-v1',
        recorded_at=datetime.now(timezone.utc).isoformat(),candidate_sha256=sha(candidate_path),
        candidate_source_sha256=candidate['candidate_source_sha256'],plan_sha256=sha(plan_target),
        preceding_completion_sha256=sha(completion_path),source_adopted=True,
        training_started=False,native_environment_steps=0,optimizer_updates_performed=0,behavior_qualified=False))
    print('Installed five reviewed clock/export source changes and v15 plan; no training started')
else:
    import torch
    sys.path.insert(0,str(ROOT / 'src'))
    from wheeled_tasks.chassis.flat_base_continuation import read_plan
    if (HERE / 'training_plan_revision017.json').exists():
        raise SystemExit('Preserve the existing revision017')
    production = read(HERE / 'clock_v15_production_cpu001.json')
    if not production['passed']:
        raise SystemExit('Installed source verification failed')
    assert all(sha(ROOT / n) == v for n,v in candidate['candidate_source_sha256'].items())
    parent_path = RUN / 'pilot019/model_001024.pt'
    parent = torch.load(parent_path,map_location='cpu',weights_only=True)
    old = read_plan(ROOT,'contracts/flat_base_finetune_v14.json')
    new = read_plan(ROOT,'contracts/flat_base_finetune_v15.json')
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    assert changed == ['control_file','control_sha256','physics_dt','schema','training_scope']
    old_sources = parent['infos']['source_sha256']
    names = list(old_sources) + ['src/wheeled_tasks/chassis/flat_clock_transition.py']
    sources = {n:sha(ROOT / n) for n in names}
    delta = sorted(n for n in names if old_sources.get(n) != sources[n])
    assert len(sources) == 118 and delta == ['scripts/train_flat_base.py',
        'src/wheeled_tasks/chassis/flat_base_continuation.py','src/wheeled_tasks/chassis/flat_clock_transition.py',
        'src/wheeled_tasks/chassis/flat_command_reference.py']
    snapshot = HERE / 'training_plan_revision017_source'
    snapshot.mkdir()
    for n in names:
        target = snapshot / n
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT / n).read_bytes())
    required = ['feedback1024_queue_completion001.json','feedback1024_behavior_review001.json',
        'feedback1024_native_milestones.json','feedback1024_training_event_review001.json',
        'static960_1024_window_audit001.json','static_sampling960_native_review001.json',
        'clock_candidate_v15_source002.json','clock_candidate_v15_base_freeze001.json',
        *prerequisites,'clock_candidate_v15_support001.json','clock_v15_install001.json','clock_v15_production_cpu001.json']
    revision = dict(schema='flat-training-plan-revision-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
        parent_checkpoint_sha256=sha(parent_path),old_plan_file='contracts/flat_base_finetune_v14.json',
        new_plan_file='contracts/flat_base_finetune_v15.json',old_plan_sha256=parent['infos']['plan_sha256'],
        new_plan_sha256=sha(plan_target),changed_plan_fields=changed,changed_source_files=delta,
        old_source_sha256=old_sources,new_source_sha256=sources,
        semantics='Exact1024 actor/std/critic/Adam/RNG/counters. Physics and fresh-feedback PD200→1000Hz; policy remains50Hz with unchanged gamma/GAE/rollout/rewards/sampling/rates. Fresh physical reset under the new timing contract. Ten critic-only adaptation updates then22 actor updates; no evaluation data fits weights.',
        additional_critic_warmup_updates=10,maximum_additional_optimizer_updates=32,
        prior_consumed_optimizer_updates=1025,behavior_qualified=False,
        prerequisite_sha256={n:sha(HERE / n) for n in required})
    assert sha(ROOT / revision['old_plan_file']) == revision['old_plan_sha256']
    write(HERE / 'training_plan_revision017.json',revision)
    write(HERE / 'clock_v15_adoption001.json',dict(schema='flat-clock-v15-adoption-v1',
        plan_sha256=sha(plan_target),revision_sha256=sha(HERE/'training_plan_revision017.json'),
        source_count=118,changed_plan_fields=changed,changed_source_files=delta,source_adopted=True,
        training_started=False,optimizer_updates_performed=0,behavior_qualified=False))
    print('Sealed revision017; maximum32 native updates; no training started')

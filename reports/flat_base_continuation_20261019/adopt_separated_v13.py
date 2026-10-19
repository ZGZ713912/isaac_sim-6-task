"""Install the verified candidate only after the old native queue has exited."""
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
write = lambda p, d: p.write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage', choices=('install', 'seal'))
args = parser.parse_args()
previous = read(RUN/'feedback832_queue001.json')
if previous['status'] != 'complete' or any(Path('/proc/'+str(previous[k])).exists() for k in ('pid','worker_pid')):
    raise SystemExit('The old controller and native worker must both finish and exit')
completion = read(HERE/'feedback832_queue_completion001.json')
if not completion['passed'] or completion['queue_sha256'] != sha(RUN/'feedback832_queue001.json'):
    raise SystemExit('The completed832 freeze is missing or changed')
candidate = read(HERE/'separated_candidate_v13_source001.json')
source_tree = HERE/'separated_v13_source_candidate'
if any(sha(source_tree/n) != v for n,v in candidate['candidate_source_sha256'].items()):
    raise SystemExit('Verified isolated source changed')
for name in ('separated_v13_source_actual832_cpu002.json', 'separated_identity_cpp_candidate001.json'):
    if not read(HERE/name)['passed']:
        raise SystemExit('Candidate verification failed: '+name)
plan_target = ROOT/'contracts/flat_base_finetune_v13.json'
if args.stage == 'install':
    if (HERE/'separated_v13_source_install001.json').exists() or plan_target.exists():
        raise SystemExit('Preserve the existing source installation')
    if any(sha(ROOT/n) != v for n,v in previous['source_sha256'].items()):
        raise SystemExit('The previous121 frozen files changed before source installation')
    backup = HERE/'separated_v13_pre_adoption_source'
    backup.mkdir()
    for name in candidate['changed_files']:
        target = ROOT/name
        if target.exists():
            old = backup/name; old.parent.mkdir(parents=True, exist_ok=True)
            old.write_bytes(target.read_bytes())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_tree/name).read_bytes())
    plan_target.write_bytes((HERE/'separated_candidate_v13.json').read_bytes())
    assert sha(plan_target) == candidate['candidate_plan_sha256']
    assert all(sha(ROOT/n) == v for n,v in candidate['candidate_source_sha256'].items())
    write(HERE/'separated_v13_source_install001.json', dict(
        schema='flat-separated-v13-source-install-v1', recorded_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=candidate['candidate_source_sha256'], plan_sha256=sha(plan_target),
        previous_completion_sha256=sha(HERE/'feedback832_queue_completion001.json'),
        source_adopted=True, training_started=False, native_optimizer_updates=0,
        behavior_qualified=False, hardware_motion_started=False))
    print('Installed six verified files and the v13 plan; native training has not started')
else:
    import torch
    sys.path.insert(0, str(ROOT/'src'))
    from wheeled_tasks.chassis.flat_base_continuation import read_plan
    if (HERE/'training_plan_revision015.json').exists():
        raise SystemExit('Preserve the existing revision015')
    test = read(HERE/'separated_v13_production_tests001.json')
    cpu = read(HERE/'separated_v13_production_actual832_cpu001.json')
    assert test['passed'] and test['passed_tests'] == 95 and cpu['passed']
    assert all(sha(ROOT/n) == v for n,v in candidate['candidate_source_sha256'].items())
    parent_path = RUN/'pilot016/model_000832.pt'
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    old = read_plan(ROOT, 'contracts/flat_base_finetune_v12.json')
    new = read_plan(ROOT, 'contracts/flat_base_finetune_v13.json')
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    assert changed == ['actor_adaptation', 'schema', 'training_scope']
    old_sources = parent['infos']['source_sha256']
    assert len(old_sources) == 116
    new_names = list(old_sources) + ['src/wheeled_algo/separated_feedback.py']
    new_sources = {n:sha(ROOT/n) for n in new_names}
    changed_sources = sorted(n for n in new_names if old_sources.get(n) != new_sources[n])
    assert changed_sources == ['scripts/train_flat_base.py', 'src/wheeled_algo/command_residual.py',
        'src/wheeled_algo/separated_feedback.py', 'src/wheeled_tasks/chassis/flat_base_continuation.py']
    snapshot = HERE/'training_plan_revision015_source'
    snapshot.mkdir()
    for name in new_names:
        target = snapshot/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    prerequisites = ['feedback832_queue_completion001.json', 'feedback832_behavior_review001.json',
        'static768_832_window_audit001.json', 'teacher_static_coupling_cpu001.json',
        'separated_v13_source_actual832_cpu002.json', 'separated_identity_cpp_candidate001.json',
        'separated_v13_source_install001.json', 'separated_v13_production_tests001.json',
        'separated_v13_production_actual832_cpu001.json']
    revision = dict(schema='flat-training-plan-revision-v1', recorded_at=datetime.now(timezone.utc).isoformat(),
        parent_checkpoint_sha256=sha(parent_path), old_plan_file='contracts/flat_base_finetune_v12.json',
        new_plan_file='contracts/flat_base_finetune_v13.json',
        old_plan_sha256=parent['infos']['plan_sha256'], new_plan_sha256=sha(plan_target),
        changed_plan_fields=changed, changed_source_files=changed_sources,
        old_source_sha256=old_sources, new_source_sha256=new_sources,
        semantics='Exact832 mean; existing actor/std/critic/Adam/RNG/counters retained; fast feedback and its Adam including step are cloned from shared feedback. Low/fast feedback separate by command demand with continuous interpolation and the unchanged bounded tracking gate. Native joint actor gradient clipping, teacher, PPO, sampling, rewards, rates, physics and acceptance unchanged. No evaluation data used for optimization.',
        additional_critic_warmup_updates=0, maximum_additional_optimizer_updates=64,
        prior_consumed_optimizer_updates=parent['infos']['consumed_optimizer_updates_total'],
        inherited_shared_feedback_updates=304, post_split_actor_updates=0, behavior_qualified=False,
        native_restore_and_learning_verification='pending actual833/865/896; low/fast Adam clone and both feedback branches must learn; frozen tensors and all counters retained',
        prerequisite_sha256={n:sha(HERE/n) for n in prerequisites})
    assert revision['old_plan_sha256'] == sha(ROOT/revision['old_plan_file'])
    write(HERE/'training_plan_revision015.json', revision)
    write(HERE/'separated_v13_adoption001.json', dict(
        schema='flat-separated-v13-source-adoption-v1', recorded_at=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(plan_target), revision_sha256=sha(HERE/'training_plan_revision015.json'),
        checkpoint_sha256=sha(parent_path), source_adopted=True, source_count=117,
        changed_plan_fields=changed, changed_source_files=changed_sources,
        unit_tests_passed=95, actual832_synthetic_sensor_checks_passed=len(cpu['checks']),
        initial_identity_cpp_checks_passed=5, training_started=False,
        optimizer_updates_performed=0, behavior_qualified=False, hardware_motion_started=False))
    print('Sealed revision015 with117 training files; native training has not started')

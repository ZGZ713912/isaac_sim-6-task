"""One bounded v14 cost-scale experiment, then four serial fixed replays."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/flat_base_20261019'
HERE = Path(__file__).resolve().parent
STATE = RUN / 'feedback1024_queue001.json'
PYTHON = '/home/noir/miniconda3/envs/isaacsim6/bin/python'
phases = ('pilot019', 'feedback1024_native_verification001', 'feedback1024_basic001',
    'feedback1024_envelope001', 'rmcs1024_envelope001', 'rmcs_default1024_envelope001')
if STATE.exists() or any((RUN / n).exists() or (RUN / (n + '.log')).exists() for n in phases):
    raise SystemExit('Preserve every existing continuation output and receipt')
lock = (RUN / '.feedback1024_queue001.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
revision_path = HERE / 'training_plan_revision016.json'
revision = read(revision_path)
hashes = dict(revision['new_source_sha256'])
for n in ('scripts/evaluate_flat_base.py', 'scripts/compare_flat_base.py', 'contracts/flat_base_finetune_v14.json',
          'reports/flat_base_continuation_20261019/verify_feedback1024_native.py',
          'scripts/flat_default_height_profile.py', 'scripts/flat_maximum_height_profile.py'):
    hashes[n] = sha(ROOT / n)
state = dict(schema='flat-feedback1024-cost-scale-serial-queue-v1', pid=os.getpid(), status='starting',
    started_at=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
    parent_checkpoint_sha256=revision['parent_checkpoint_sha256'], plan_revision_sha256=sha(revision_path),
    maximum_additional_optimizer_updates=64, expected_critic_only_updates=10, expected_actor_updates=54,
    new_optimizer_updates=0, subsequent_training_queued=False, hardware_motion_started=False, history=[])


def save():
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(STATE)


def run(phase, command):
    if any(sha(ROOT / n) != v for n,v in hashes.items()):
        raise ValueError('Frozen experiment source changed')
    if phase not in phases or (RUN / (phase + '.log')).exists():
        raise ValueError('Unexpected phase or preexisting output log')
    state.update(status='running', phase=phase)
    with (RUN / (phase + '.log')).open('x') as log:
        worker = subprocess.Popen([PYTHON, *command], cwd=ROOT,
            env={**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES'}, stdout=log, stderr=subprocess.STDOUT)
        state['worker_pid'] = worker.pid
        save()
        code = worker.wait()
    state['history'].append(dict(phase=phase, exit_code=code))
    progress = RUN / 'pilot019/progress.json'
    if progress.exists():
        state['new_optimizer_updates'] = read(progress)['successful_updates_total'] - 960
    save()
    if code:
        raise RuntimeError('Native experiment child failed: ' + phase)


try:
    save()
    previous = read(RUN / 'feedback960_queue001.json')
    complete = read(HERE / 'feedback960_queue_completion001.json')
    if (previous['status'] != 'complete' or not complete['passed']
            or complete['queue_sha256'] != sha(RUN / 'feedback960_queue001.json')
            or any(Path('/proc/' + str(previous[k])).exists() for k in ('pid','worker_pid'))):
        raise ValueError('Previous queue must be complete, exited and sealed')
    for n in ('static_mean_sampling960_native002', 'static_mean_sampling960_1k_native001'):
        diag = read(RUN / n / 'audit.json')
        if diag['status'] != 'audited' or Path('/proc/' + str(diag['pid'])).exists():
            raise ValueError('Previous diagnostic must have completed and exited')
    for n,v in revision['prerequisite_sha256'].items():
        if sha(HERE / n) != v:
            raise ValueError('Frozen prerequisite changed: ' + n)
    parent = RUN / 'pilot018/model_000960.pt'
    if sha(parent) != revision['parent_checkpoint_sha256']:
        raise ValueError('Authenticated parent checkpoint changed')
    run('pilot019', ['scripts/train_flat_base.py', '--plan', 'contracts/flat_base_finetune_v14.json',
        '--output', str(RUN / 'pilot019'), '--resume', str(parent),
        '--resume-plan-revision', str(revision_path), '--num-envs', '512', '--updates', '64'])
    run('feedback1024_native_verification001', [str(HERE / 'verify_feedback1024_native.py')])
    proof = read(HERE / 'feedback1024_native_milestones.json')
    if not proof['passed'] or not proof['all_required_milestones_present']:
        raise ValueError('Native restoration, warmup or learning verification failed')
    checkpoint = RUN / 'pilot019/model_001024.pt'
    state['checkpoint_sha256'] = sha(checkpoint)
    replay_specs = [
        ('feedback1024_basic001', [], 'baseline_native002', 'feedback960_basic001'),
        ('feedback1024_envelope001', ['--envelope'], 'envelope_source002', 'feedback960_envelope001'),
        ('rmcs1024_envelope001', ['--deployment-pd','--rmcs-command-reference','--envelope'],
            'rmcs_source_envelope001', 'rmcs960_envelope001'),
        ('rmcs_default1024_envelope001', ['--deployment-pd','--rmcs-command-reference','--packaged-default-height','--envelope'],
            'rmcs_default_source_envelope001', 'rmcs_default960_envelope001')]
    sys.path.insert(0, str(ROOT / 'scripts'))
    from compare_flat_base import compare
    for phase, flags, source, previous in replay_specs:
        run(phase, ['scripts/evaluate_flat_base.py', '--checkpoint', str(checkpoint),
            '--output', str(RUN / phase), *flags])
        result_path = RUN / phase / 'evaluation.json'
        result = read(result_path)
        if result['status'] != 'evaluated' or result['checkpoint_sha256'] != state['checkpoint_sha256']:
            raise ValueError('Fixed replay did not complete against the final1024 checkpoint')
        for label, ref in (('source', source), ('parent960', previous)):
            refpath = RUN / ref / 'evaluation.json'
            comparison = compare(read(refpath), result)
            comparison.update(baseline_sha256=sha(refpath), candidate_sha256=sha(result_path))
            (HERE / ('comparison_' + phase + '_' + label + '.json')).write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
    state.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat())
    save()
except Exception:
    state.update(status='failed', error=traceback.format_exc())
    save()
    raise

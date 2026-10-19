"""One bounded moving-feedback trial after both static-recall diagnostics are sealed."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
STATE = RUN / 'static_teacher0_1296_queue001.json'
CONTRACT = HERE / 'static_teacher0_1296_contract001.json'
PHASES = ('pilot026', 'static_teacher0_native_final', 'deployment1296_basic001',
          'rmcs_default1296_envelope001', 'rmcs_maximum1296_envelope001')
if STATE.exists() or CONTRACT.exists() or any((RUN / p).exists() or (RUN / (p + '.log')).exists() for p in PHASES):
    raise SystemExit('Preserve all completed and partial trial outputs')
proofs = {}
for suffix, review_name in (('envelope', 'envelope'), ('maximum', 'maximum')):
    queue_path = RUN / f'static_recall1280_{suffix}_queue001.json'
    review_path = HERE / f'static_recall1280_{review_name}_review001.json'
    queue, review = read(queue_path), read(review_path)
    if (queue['status'] != 'complete' or not review['passed'] or review['queue_sha256'] != sha(queue_path)
            or not review['comparisons']['parent1280']['no_lost_baseline_checks']
            or any(Path('/proc/' + str(queue[k])).exists() for k in ('pid', 'worker_pid'))):
        raise SystemExit('Wait for sealed fixed-weight diagnostics and all owners to exit')
    proofs[suffix] = sha(review_path)
parent = RUN / 'pilot025/model_001280.pt'
candidate = HERE / 'static_feedback_v5_candidate001'
plan = candidate / 'plan_static_teacher0.json'
revision_path = HERE / 'static_teacher0_plan_revision021.json'
revision = read(revision_path)
cpu_path = candidate / 'integration_cpu001.json'
plan_cpu_path = candidate / 'teacher0_plan_cpu001.json'
ast_path = candidate / 'learning_path_ast001.json'
cpu, plan_cpu, ast = read(cpu_path), read(plan_cpu_path), read(ast_path)
if (not all(p['passed'] for p in (cpu, plan_cpu, ast))
        or plan_cpu['candidate_plan_sha256'] != sha(plan)
        or revision['parent_checkpoint_sha256'] != sha(parent)
        or revision['new_plan_sha256'] != sha(plan)):
    raise SystemExit('The adopted plan must match actual parent and reviewed candidate')
for name, value in cpu['candidate_source_sha256'].items():
    if name.startswith(('scripts/', 'src/')) and sha(ROOT / name) != value:
        raise SystemExit('Reviewed candidate was not adopted exactly: ' + name)
verifier = HERE / 'verify_static_teacher0_1296_native.py'
hashes = dict(revision['new_source_sha256'])
for path in (ROOT / 'scripts/evaluate_flat_base.py', ROOT / 'scripts/export_flat_base.py',
             Path(__file__), verifier, plan, revision_path):
    hashes[str(path.relative_to(ROOT))] = sha(path)
if any(sha(ROOT / name) != value for name, value in hashes.items()):
    raise SystemExit('Frozen execution sources changed')
lock = (RUN / '.static_teacher0_1296.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
contract = dict(schema='flat-static-feedback-teacher0-bounded-contract-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), parent_checkpoint_sha256=sha(parent),
    fixed_weight_native_reviews=proofs, source_sha256=hashes,
    plan_sha256=sha(plan), plan_revision_sha256=sha(revision_path),
    CPU_proofs=dict(integration=sha(cpu_path), plan=sha(plan_cpu_path), learning_path=sha(ast_path)),
    optimizer_updates=16, actor_updates=16, extra_critic_warmup=0,
    expected_final_successful_updates=1296, expected_final_consumption=1297,
    inherited_successful_frames=73924608, additional_successful_frames=6291456,
    expected_final_successful_frames=80216064,
    changed='Declared frozen1248 static head; v4 moving feedback retained; teacher coefficient .02 to0.',
    unchanged='Parent moving tensors/std/critic/Adam/RNG/history; reward,4096 population, groups, clocks, rates and original full-horizon thresholds.',
    evaluation='Serial basic, RMCS23cm full22, RMCS42cm full22. No subsequent PPO is queued.',
    causal_scope='Static composition was separately diagnosed; this small trial tests moving feedback without teacher anchoring. No paired teacher causal estimate.',
    behavior_qualified=False, subsequent_training_queued=False, long_training_started=False,
    heldout_seed_used=False, hardware_motion_started=False)
CONTRACT.write_text(json.dumps(contract, indent=2, allow_nan=False) + '\n')
state = dict(schema='flat-static-feedback-teacher0-1296-serial-queue-v1', status='starting',
    pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
    parent_checkpoint_sha256=sha(parent), contract_sha256=sha(CONTRACT), source_sha256=hashes,
    history=[], additional_optimizer_updates=16, subsequent_training_queued=False,
    heldout_seed_used=False, hardware_motion_started=False)


def save():
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    temporary.replace(STATE)


def frozen():
    if any(sha(ROOT / n) != value for n, value in hashes.items()):
        raise ValueError('Frozen trial source changed')


def child(name, command, watch=False):
    frozen()
    state.update(status='running', phase=name)
    with (RUN / (name + '.log')).open('x') as log:
        worker = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES'},
                                  stdout=log, stderr=subprocess.STDOUT)
        state['worker_pid'] = worker.pid
        save()
        checked = False
        while worker.poll() is None:
            if watch and not checked and (RUN / 'pilot026/model_001281.pt').exists():
                with (RUN / 'static_teacher0_native_first.log').open('x') as first_log:
                    code = subprocess.run([sys.executable, str(verifier), '--first-only'], cwd=ROOT,
                                          stdout=first_log, stderr=subprocess.STDOUT).returncode
                state['first_native_verification_exit_code'] = code
                save()
                if code:
                    worker.terminate()
                    worker.wait(timeout=60)
                    raise RuntimeError('First native state audit failed; stopped only this worker')
                checked = True
            time.sleep(2)
        state['history'].append(dict(phase=name, exit_code=worker.returncode))
        save()
    if worker.returncode:
        raise RuntimeError('Child failed; no later work launched: ' + name)
    frozen()


try:
    save()
    child('pilot026', [sys.executable, str(ROOT / 'scripts/train_flat_base.py'), '--plan', str(plan.relative_to(ROOT)),
        '--output', str(RUN / 'pilot026'), '--updates', '16', '--num-envs', '4096',
        '--resume', str(parent), '--resume-plan-revision', str(revision_path)], watch=True)
    final = RUN / 'pilot026/model_001296.pt'
    state['checkpoint_sha256'] = sha(final)
    child('static_teacher0_native_final', [sys.executable, str(verifier)])
    for name, flags in (('deployment1296_basic001', []),
                        ('rmcs_default1296_envelope001', ['--rmcs-command-reference', '--packaged-default-height', '--envelope']),
                        ('rmcs_maximum1296_envelope001', ['--rmcs-command-reference', '--packaged-maximum-height', '--envelope'])):
        child(name, [sys.executable, str(ROOT / 'scripts/evaluate_flat_base.py'), '--seed', '191029',
            '--checkpoint', str(final), '--deployment-pd', *flags, '--output', str(RUN / name)])
        evaluation = read(RUN / name / 'evaluation.json')
        cases = evaluation['result']['cases']
        state['history'][-1].update(evaluation_sha256=sha(RUN / name / 'evaluation.json'),
            passed_cases=sum(c['passed'] for c in cases.values()), requested_cases=len(cases),
            full_horizon_episodes=sum(c['full_horizon_episodes'] for c in cases.values()),
            requested_episodes=sum(c['requested_episodes'] for c in cases.values()))
        save()
    state.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat())
except Exception:
    state.update(status='failed', error=traceback.format_exc(), finished_at=datetime.now(timezone.utc).isoformat())
    raise
finally:
    save()

"""One bounded zero wheel-head trial after the sealed1296 continuation."""
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
STATE = RUN / 'wheel_authority1312_queue001.json'
CONTRACT = HERE / 'wheel_authority1312_contract001.json'
PHASES = ('pilot027', 'wheel_authority_native_final', 'deployment1312_basic001',
          'rmcs_default1312_envelope001', 'rmcs_maximum1312_envelope001')
if STATE.exists() or CONTRACT.exists() or any((RUN / p).exists() or (RUN / (p + '.log')).exists() for p in PHASES):
    raise SystemExit('Preserve all completed and partial trial outputs')
prior_path = RUN / 'static_teacher0_1296_queue001.json'
seal_path = HERE / 'static_teacher0_1296_completion001.json'
prior, seal = read(prior_path), read(seal_path)
if (prior['status'] != 'complete' or not seal['passed'] or seal['queue_sha256'] != sha(prior_path)
        or any(Path('/proc/' + str(prior[k])).exists() for k in ('pid', 'worker_pid'))):
    raise SystemExit('Wait for the preceding trial, all three native replays and every owner to exit')
parent = RUN / 'pilot026/model_001296.pt'
candidate = HERE / 'wheel_authority_candidate001'
plan = candidate / 'plan_wheel.json'
revision_path = HERE / 'wheel_authority_plan_revision022.json'
revision = read(revision_path)
cpu_path = candidate / 'integration_cpu002.json'
actual_path = candidate / 'actual1296_cpu002.json'
cpu, actual = read(cpu_path), read(actual_path)
unit_path = candidate / 'invariance_unit001.json'
unit = read(unit_path)
if (not all(p['passed'] for p in (cpu, actual, unit))
        or cpu['candidate_plan_sha256'] != sha(plan)
        or actual['completion_sha256'] != sha(seal_path)
        or actual['parent_checkpoint_sha256'] != sha(parent)
        or revision['parent_checkpoint_sha256'] != sha(parent)
        or revision['new_plan_sha256'] != sha(plan)):
    raise SystemExit('The adopted zero extension must match actual parent and reviewed candidate')
for name, value in cpu['candidate_source_sha256'].items():
    if name.startswith(('scripts/', 'src/')) and sha(ROOT / name) != value:
        raise SystemExit('Reviewed candidate was not adopted exactly: ' + name)
verifier = HERE / 'verify_wheel_authority1312_native.py'
hashes = dict(revision['new_source_sha256'])
for path in (ROOT / 'scripts/evaluate_flat_base.py', ROOT / 'scripts/export_flat_base.py',
             Path(__file__), verifier, plan, revision_path):
    hashes[str(path.relative_to(ROOT))] = sha(path)
if any(sha(ROOT / name) != value for name, value in hashes.items()):
    raise SystemExit('Frozen execution sources changed')
lock = (RUN / '.wheel_authority1312.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
contract = dict(schema='flat-wheel-authority-bounded-contract-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), parent_checkpoint_sha256=sha(parent),
    parent_completion_sha256=sha(seal_path), source_sha256=hashes,
    plan_sha256=sha(plan), plan_revision_sha256=sha(revision_path),
    CPU_proofs=dict(integration=sha(cpu_path), actual_parent=sha(actual_path), synthetic_invariance=sha(unit_path)),
    optimizer_updates=16, actor_updates=16, extra_critic_warmup=0,
    expected_final_successful_updates=1312, expected_final_consumption=1313,
    inherited_successful_frames=80216064, additional_successful_frames=6291456,
    expected_final_successful_frames=86507520,
    changed='Zero-initialized high-demand/braking wheel head; only994 appended parameters train; all existing actor parameters and Adam freeze.',
    unchanged='Exact entry function and parent actor/std/critic/Adam/RNG/history; reward,4096 population,groups,clocks,rates,teacher0 and original full-horizon thresholds.',
    evaluation='Serial basic, RMCS23cm full22, RMCS42cm full22. No subsequent PPO is queued.',
    causal_scope='Bounded additional wheel authority with every existing actor branch frozen. No global convergence or hardware claim.',
    behavior_qualified=False, subsequent_training_queued=False, long_training_started=False,
    heldout_seed_used=False, hardware_motion_started=False)
CONTRACT.write_text(json.dumps(contract, indent=2, allow_nan=False) + '\n')
state = dict(schema='flat-wheel-authority1312-serial-queue-v1', status='starting',
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
            if watch and not checked and (RUN / 'pilot027/model_001297.pt').exists():
                with (RUN / 'wheel_authority_native_first.log').open('x') as first_log:
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
    child('pilot027', [sys.executable, str(ROOT / 'scripts/train_flat_base.py'), '--plan', str(plan.relative_to(ROOT)),
        '--output', str(RUN / 'pilot027'), '--updates', '16', '--num-envs', '4096',
        '--resume', str(parent), '--resume-plan-revision', str(revision_path)], watch=True)
    final = RUN / 'pilot027/model_001312.pt'
    state['checkpoint_sha256'] = sha(final)
    child('wheel_authority_native_final', [sys.executable, str(verifier)])
    for name, flags in (('deployment1312_basic001', []),
                        ('rmcs_default1312_envelope001', ['--rmcs-command-reference', '--packaged-default-height', '--envelope']),
                        ('rmcs_maximum1312_envelope001', ['--rmcs-command-reference', '--packaged-maximum-height', '--envelope'])):
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

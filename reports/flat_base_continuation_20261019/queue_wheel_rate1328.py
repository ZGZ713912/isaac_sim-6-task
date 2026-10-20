"""Run one32-update isolated wheel-rate trial followed by three full native replays."""
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
CANDIDATE = HERE / 'wheel_rate_candidate001'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
STATE = RUN / 'wheel_rate1328_queue001.json'
CONTRACT = HERE / 'wheel_rate1328_contract001.json'
PHASES = ('pilot028_wheel_rate1328', 'wheel_rate1328_native_final', 'wheel_rate1328_basic001',
    'wheel_rate1328_default001', 'wheel_rate1328_maximum001')
if STATE.exists() or CONTRACT.exists() or any((RUN / p).exists() or (RUN / (p + '.log')).exists() for p in PHASES):
    raise SystemExit('Preserve all previous trial outputs')
candidate_path = CANDIDATE / 'cpu001.json'
integration_path = CANDIDATE / 'integration_cpu003.json'
candidate, integration = read(candidate_path), read(integration_path)
parent, plan = ROOT / candidate['parent_checkpoint'], ROOT / candidate['plan_file']
revision = CANDIDATE / 'revision.json'
previous_path = RUN / 'wheel_from1280_queue001.json'
seal_path = HERE / 'wheel_from1280_completion001.json'
previous, seal = read(previous_path), read(seal_path)
if (previous['status'] != 'complete' or not seal['passed'] or seal['queue_sha256'] != sha(previous_path)
        or any(Path('/proc/' + str(previous[k])).exists() for k in ('pid', 'worker_pid'))):
    raise SystemExit('All previous owners must have exited and the trial must be sealed')
if (not candidate['passed'] or not all(candidate['checks'].values()) or not integration['passed']
        or not all(integration['checks'].values()) or integration['candidate_sha256'] != sha(candidate_path)
        or sha(parent) != candidate['parent_checkpoint_sha256'] or sha(plan) != candidate['plan_sha256']
        or sha(revision) != candidate['revision_sha256']):
    raise SystemExit('The exact actual-parent rate and integration proofs are required')
if any(not c['comparisons']['parent1280']['no_lost_checks']
        or c['comparisons']['parent1280']['relative_regressions'] for c in seal['cohorts'].values()):
    raise SystemExit('No new regression from1280 may be silently accepted')
clock_path = HERE / 'release_clock_adoption024.json'
clock = read(clock_path)
if not clock['passed'] or sha(ROOT / 'scripts/export_flat_base.py') != clock['new_export_sha256']:
    raise SystemExit('The authenticated clock-reader adoption changed')
verifier = HERE / 'verify_wheel_rate_native.py'
hashes = dict(candidate['source_sha256'])
for p in (ROOT / 'scripts/evaluate_flat_base.py', ROOT / 'scripts/export_flat_base.py', Path(__file__),
        verifier, plan, revision, candidate_path, integration_path, clock_path,
        HERE / 'verify_wheel_rate_candidate003.py', HERE / 'prepare_wheel_rate_trial.py'):
    hashes[str(p.relative_to(ROOT))] = sha(p)
if any(sha(ROOT / n) != value for n, value in hashes.items()):
    raise SystemExit('Execution sources do not match reviewed parent')
lock = (RUN / '.wheel_rate1328.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
contract = dict(schema='flat-wheel-rate1328-bounded-contract-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), parent_checkpoint_sha256=sha(parent),
    candidate_sha256=sha(candidate_path), integration_CPU_sha256=sha(integration_path),
    plan_sha256=sha(plan), revision_sha256=sha(revision), source_sha256=hashes,
    optimizer_updates=32, actor_updates=32, additional_critic_warmup=0,
    expected_final_successful_updates=1328, expected_final_lineage_consumption=1329,
    prior_rejected_sibling_consumption_retained=16,
    origin1280_and_all_subsequent_trial_consumption_at_exit=1345,
    inherited_successful_training_frames=80216064, additional_training_frames=12582912,
    expected_final_successful_training_frames=92798976,
    changed_plan_fields=['learning_rate'], actor_rate_before=.0003, actor_rate_after=.001,
    critic_rate=.0005, trainable_actor_parameters=994,
    frozen='All existing actor branches/std/Adam; old wheel-head Adam retained. Reward,4096 population,50/1000Hz and thresholds retained.',
    required_replays='Original basic and RMCS23cm/42cm full22,seed191029; no subsequent PPO queued.',
    behavior_qualified=False, long_training_started=False, heldout_seed_used=False, hardware_motion_started=False)
CONTRACT.write_text(json.dumps(contract, indent=2, allow_nan=False) + '\n')
state = dict(schema='flat-wheel-rate1328-serial-queue-v1', status='starting', pid=os.getpid(),
    started_at=datetime.now(timezone.utc).isoformat(), contract_sha256=sha(CONTRACT), source_sha256=hashes,
    parent_checkpoint_sha256=sha(parent), history=[], additional_optimizer_updates=32,
    prior_sibling_consumption_retained=True, subsequent_training_queued=False,
    heldout_seed_used=False, hardware_motion_started=False)


def save():
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    tmp.replace(STATE)


def frozen():
    if any(sha(ROOT / n) != value for n, value in hashes.items()):
        raise ValueError('Frozen rate-trial execution sources changed')


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
            if watch and not checked and (RUN / 'pilot028_wheel_rate1328/model_001297.pt').exists():
                with (RUN / 'wheel_rate1328_native_first.log').open('x') as first:
                    code = subprocess.run([sys.executable, str(verifier), '--first-only'], cwd=ROOT,
                        stdout=first, stderr=subprocess.STDOUT).returncode
                state['first_native_verification_exit_code'] = code
                save()
                if code:
                    worker.terminate()
                    worker.wait(timeout=60)
                    raise RuntimeError('First state audit failed; stopped this worker only')
                checked = True
            time.sleep(2)
        state['history'].append(dict(phase=name, exit_code=worker.returncode))
        save()
    if worker.returncode:
        raise RuntimeError('Child failed; no later work launched: ' + name)
    if watch and state.get('first_native_verification_exit_code') != 0:
        raise RuntimeError('First state audit is missing')
    frozen()


try:
    save()
    child('pilot028_wheel_rate1328', [sys.executable, str(ROOT / 'scripts/train_flat_base.py'),
        '--plan', str(plan.relative_to(ROOT)), '--output', str(RUN / 'pilot028_wheel_rate1328'),
        '--updates', '32', '--num-envs', '4096', '--resume', str(parent),
        '--resume-plan-revision', str(revision)], watch=True)
    final = RUN / 'pilot028_wheel_rate1328/model_001328.pt'
    state['checkpoint_sha256'] = sha(final)
    child('wheel_rate1328_native_final', [sys.executable, str(verifier)])
    for name, flags in (('wheel_rate1328_basic001', []),
            ('wheel_rate1328_default001', ['--rmcs-command-reference', '--packaged-default-height', '--envelope']),
            ('wheel_rate1328_maximum001', ['--rmcs-command-reference', '--packaged-maximum-height', '--envelope'])):
        child(name, [sys.executable, str(ROOT / 'scripts/evaluate_flat_base.py'), '--seed', '191029',
            '--checkpoint', str(final), '--deployment-pd', *flags, '--output', str(RUN / name)])
        data = read(RUN / name / 'evaluation.json')
        cases = data['result']['cases']
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

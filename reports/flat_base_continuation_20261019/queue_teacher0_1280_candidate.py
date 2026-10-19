"""Bounded teacher-zero trial; adopt only after the completed1264 diagnostic review."""
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
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
STATE = RUN / 'teacher0_1280_queue001.json'
CONTRACT = HERE / 'teacher0_1280_contract001.json'
phases = ('pilot025', 'teacher0_native_final', 'deployment1280_basic001')
if STATE.exists() or CONTRACT.exists() or any((RUN / name).exists() or (RUN / (name + '.log')).exists() for name in phases):
    raise SystemExit('Preserve previous trials and all partial outputs')
prior_path = RUN / 'feedback1264_population_queue001.json'
prior = read(prior_path)
seal_path = HERE / 'feedback1264_population_completion001.json'
seal = read(seal_path)
diag_path = RUN / 'low_gate1264_queue001.json'
diag = read(diag_path)
diag_review_path = HERE / 'low_gate1264_diagnostic_review001.json'
diag_review = read(diag_review_path)
if (prior['status'] != 'complete' or not seal['passed'] or seal['queue_sha256'] != sha(prior_path)
        or diag['status'] != 'complete' or not diag_review['passed']
        or diag['review_sha256'] != sha(diag_review_path)
        or any(Path('/proc/' + str(q[k])).exists() for q in (prior, diag) for k in ('pid', 'worker_pid'))):
    raise SystemExit('Wait for1264 source/event sealing and the serial gate diagnostic to exit')
parent = RUN / 'pilot024/model_001264.pt'
if sha(parent) != seal['checkpoint_sha256'] or sha(parent) != diag['checkpoint_sha256']:
    raise SystemExit('The same complete1264 parent is required')
plan = HERE / 'teacher_ablation_candidate001/plan_teacher0.json'
revision_path = HERE / 'teacher0_plan_revision020.json'
revision = read(revision_path)  # Written only after an explicit evidence-based adoption decision.
verifier = HERE / 'verify_teacher0_1280_native.py'
cpu_path = plan.with_name('cpu001.json')
cpu = read(cpu_path)
cause_cpu_path = HERE / 'termination_metrics_candidate001/cpu003.json'
cause_cpu = read(cause_cpu_path)
if (not cpu['passed'] or cpu['candidate_plan_sha256'] != sha(plan) or not cause_cpu['passed']
        or revision['parent_checkpoint_sha256'] != sha(parent)
        or revision['new_plan_sha256'] != sha(plan)
        or revision['new_source_sha256']['scripts/train_flat_base.py'] != sha(ROOT / 'scripts/train_flat_base.py')
        or sha(ROOT / 'scripts/train_flat_base.py') != 'fb09bf77cb0bba0df968695aa456304cc84694866a7fde158da83607c6331e3c'):
    raise SystemExit('Adopt the reviewed zero-teacher plan and cause-only logger before launching')
hashes = dict(prior['source_sha256'])
hashes['scripts/train_flat_base.py'] = sha(ROOT / 'scripts/train_flat_base.py')
for p in (plan, verifier):
    hashes[str(p.relative_to(ROOT))] = sha(p)
if any(sha(ROOT / n) != value for n, value in hashes.items()):
    raise SystemExit('The reviewed frozen sources changed')
lock = (RUN / '.teacher0_1280_queue001.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
contract = dict(schema='flat-teacher0-bounded-continuation-contract-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), parent_checkpoint_sha256=sha(parent),
    parent_completion_sha256=sha(seal_path), gate_diagnostic_review_sha256=sha(diag_review_path),
    plan_sha256=sha(plan), plan_revision_sha256=sha(revision_path), source_sha256=hashes,
    candidate_CPU_proofs=dict(plan=sha(cpu_path), cause_logger=sha(cause_cpu_path)),
    optimizer_updates=16, actor_updates=16, extra_critic_warmup=0,
    expected_final_successful_updates=1280, expected_final_consumption=1281,
    inherited_successful_frames=67633152, additional_successful_frames=6291456,
    expected_final_successful_frames=73924608,
    changed='Only teacher coefficient .02 to0 and diagnostic cause logging; no extra warmup.',
    unchanged='All learned parent tensors, Adam/std/RNG at entry; architecture, reward, groups,4096 population,50Hz/1000Hz clocks, learning rates,12 Adam steps/update and fixed thresholds.',
    evaluation='Basic fixed development replay first; full release cohorts remain required if this screen passes.',
    scope='Serial learning experiment, not a matched same-parent teacher A/B or a causal population comparison.',
    subsequent_training_queued=False, long_training_started=False, heldout_seed_used=False,
    behavior_qualified=False, hardware_motion_started=False)
CONTRACT.write_text(json.dumps(contract, indent=2) + '\n')
state = dict(schema='flat-teacher0-1280-serial-queue-v1', status='starting', pid=os.getpid(),
    started_at=datetime.now(timezone.utc).isoformat(), parent_checkpoint_sha256=sha(parent),
    contract_sha256=sha(CONTRACT), source_sha256=hashes, history=[],
    additional_optimizer_updates=16, subsequent_training_queued=False,
    heldout_seed_used=False, hardware_motion_started=False)


def save():
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
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
            if watch and not checked and (RUN / 'pilot025/model_001265.pt').exists():
                with (RUN / 'teacher0_native_first.log').open('x') as first_log:
                    code = subprocess.run([sys.executable, str(verifier), '--first-only'], cwd=ROOT,
                                          stdout=first_log, stderr=subprocess.STDOUT).returncode
                state['first_native_verification_exit_code'] = code
                save()
                if code:
                    worker.terminate()
                    worker.wait(timeout=60)
                    raise RuntimeError('First native continuation proof failed; stopped only this worker')
                checked = True
            time.sleep(2)
        state['history'].append(dict(phase=name, exit_code=worker.returncode))
        save()
    if worker.returncode:
        raise RuntimeError('Child failed; no later work launched: ' + name)
    frozen()


try:
    save()
    child('pilot025', [sys.executable, str(ROOT / 'scripts/train_flat_base.py'), '--plan', str(plan.relative_to(ROOT)),
        '--output', str(RUN / 'pilot025'), '--updates', '16', '--num-envs', '4096', '--resume', str(parent),
        '--resume-plan-revision', str(revision_path)], watch=True)
    final = RUN / 'pilot025/model_001280.pt'
    state['checkpoint_sha256'] = sha(final)
    child('teacher0_native_final', [sys.executable, str(verifier)])
    child('deployment1280_basic001', [sys.executable, str(ROOT / 'scripts/evaluate_flat_base.py'),
        '--seed', '191029', '--checkpoint', str(final), '--deployment-pd',
        '--output', str(RUN / 'deployment1280_basic001')])
    sys.path.insert(0, str(ROOT / 'scripts'))
    from compare_flat_base import compare
    evaluation_path = RUN / 'deployment1280_basic001/evaluation.json'
    evaluation = read(evaluation_path)
    comparisons = {}
    for label, reference in (('source', 'deployment_source001'), ('parent1264', 'deployment1264_basic001'),
                             ('parent1248', 'deployment1248_basic001')):
        output = HERE / ('comparison_deployment1280_basic001_' + label + '.json')
        result = compare(read(RUN / reference / 'evaluation.json'), evaluation)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        comparisons[label] = dict(path=str(output.relative_to(ROOT)), sha256=sha(output),
                                  lost_checks=result['lost_baseline_checks'])
    state['history'][-1].update(evaluation_sha256=sha(evaluation_path),
        passed_cases=sum(c['passed'] for c in evaluation['result']['cases'].values()),
        requested_cases=len(evaluation['result']['cases']), comparisons=comparisons)
    state.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat())
except Exception:
    state.update(status='failed', error=traceback.format_exc(), finished_at=datetime.now(timezone.utc).isoformat())
    raise
finally:
    save()

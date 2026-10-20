"""Run exactly2 critic +8 static actor updates, audit them, and replay basic behavior."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_tasks.chassis.flat_base_continuation import digest, encode

CANDIDATE = HERE / 'static_grip_candidate002'
STATE = RUN / 'static_grip1338_queue001.json'
CONTRACT = CANDIDATE / 'execution001.json'
TRIAL = RUN / 'pilot029_static_grip1338'
EVALUATION = RUN / 'ground025_static1338_basic001'
if STATE.exists() or CONTRACT.exists() or TRIAL.exists() or EVALUATION.exists():
    raise SystemExit('Preserve existing static grip pilot outputs')
prep_path = CANDIDATE / 'preparation001.json'
prep = json.loads(prep_path.read_text())
baseline_path = ROOT / prep['baseline_file']
baseline = json.loads(baseline_path.read_text())
parent = ROOT / prep['parent_checkpoint']
plan = ROOT / prep['plan_file']
revision = CANDIDATE / 'revision.json'
if not prep['passed'] or digest(parent) != prep['parent_checkpoint_sha256']:
    raise SystemExit('Require the exact sealed1328 parent')
if (digest(baseline_path) != prep['baseline_sha256'] or baseline['status'] != 'evaluated'
        or baseline['startup']['contact_material']['ground_static_range'] != [.2, .5]):
    raise SystemExit('Require the completed baseline using the latest0.2–0.5 range')
if digest(plan) != prep['plan_sha256'] or digest(revision) != prep['revision_sha256']:
    raise SystemExit('The declared plan or migration changed')
tests = ET.parse(CANDIDATE / 'cpu001.xml').getroot()
if (sum(int(s.attrib['tests']) for s in tests) != 97
        or any(int(s.attrib.get('failures', 0)) or int(s.attrib.get('errors', 0)) for s in tests)):
    raise SystemExit('Candidate CPU checks failed')
hashes = dict(prep['source_sha256'])
for path in (Path(__file__), HERE / 'verify_static_grip_native.py', ROOT / 'scripts/evaluate_flat_base.py',
        ROOT / 'scripts/export_flat_base.py', prep_path, plan, revision, CANDIDATE / 'cpu001.xml', baseline_path):
    hashes[str(path.relative_to(ROOT))] = digest(path)
if any(digest(ROOT / name) != value for name, value in hashes.items()):
    raise SystemExit('Reviewed execution sources changed')
lock = (RUN / '.static_grip1338.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
CONTRACT.write_bytes(encode(dict(schema='flat-static-grip-bounded-pilot-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), parent_checkpoint_sha256=digest(parent),
    plan_sha256=digest(plan), baseline_sha256=digest(baseline_path), source_sha256=hashes,
    optimizer_updates=10, critic_only_updates=2, static_actor_updates=8,
    population=4096, policy_hz=50, physics_hz=1000, ground_material_range=[.2, .5], wheel_material=.1,
    frozen='all moving heads, original actor, command, wheel authority, std and their Adam',
    required='native restore/ownership audits at1329,1331,1338 then complete basic replay',
    long_training_started=False, subsequent_training_queued=False, behavior_qualified=False)))
state = dict(schema='flat-static-grip-serial-queue-v1', status='starting', pid=os.getpid(),
    started_at=datetime.now(timezone.utc).isoformat(), source_sha256=hashes, history=[],
    contract_sha256=digest(CONTRACT), additional_optimizer_updates=10,
    prior_rejected_sibling_consumption_retained=True, subsequent_training_queued=False,
    hardware_motion_started=False)


def save():
    path = STATE.with_suffix('.tmp')
    path.write_bytes(encode(state))
    path.replace(STATE)


def frozen():
    if any(digest(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError('Frozen static pilot sources changed')


def verify(count):
    path = RUN / f'static_grip1338_native{count}.log'
    with path.open('x') as log:
        result = subprocess.run([sys.executable, str(HERE / 'verify_static_grip_native.py'),
            '--count', str(count)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Native full-state/ownership audit failed at{count}')


def child(name, command, watch=False):
    frozen()
    state.update(status='running', phase=name)
    with (RUN / (name + '.log')).open('x') as log:
        worker = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES'},
            stdout=log, stderr=subprocess.STDOUT)
        state['worker_pid'] = worker.pid
        state['worker_start_ticks'] = Path(f'/proc/{worker.pid}/stat').read_text().split(') ', 1)[1].split()[19]
        save()
        checked = set()
        try:
            while worker.poll() is None:
                for count in ((1329, 1331) if watch else ()):
                    if count not in checked and (TRIAL / f'model_{count:06d}.pt').exists():
                        verify(count)
                        checked.add(count)
                        state['verified_training_milestones'] = sorted(checked)
                        save()
                time.sleep(2)
        except BaseException:
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=60)
            raise
    state['history'].append(dict(phase=name, exit_code=worker.returncode))
    save()
    if worker.returncode:
        raise RuntimeError('Child failed; no subsequent phase launched: ' + name)
    if watch:
        for count in (1329, 1331):
            if count not in checked:
                verify(count)
    frozen()


try:
    save()
    child('pilot029_static_grip1338', [sys.executable, str(ROOT / 'scripts/train_flat_base.py'),
        '--plan', str(plan.relative_to(ROOT)), '--resume', str(parent), '--resume-plan-revision', str(revision),
        '--output', str(TRIAL), '--updates', '10', '--num-envs', '4096'], watch=True)
    verify(1338)
    final = TRIAL / 'model_001338.pt'
    state['checkpoint_sha256'] = digest(final)
    child(EVALUATION.name, [sys.executable, str(ROOT / 'scripts/evaluate_flat_base.py'),
        '--plan', str(plan.relative_to(ROOT)), '--checkpoint', str(final), '--output', str(EVALUATION)])
    report_path = EVALUATION / 'evaluation.json'
    report = json.loads(report_path.read_text())
    if report['status'] != 'evaluated' or report['checkpoint_sha256'] != digest(final):
        raise RuntimeError('The complete matching basic evaluation is missing')
    cases = report['result']['cases']
    state.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
        evaluation_sha256=digest(report_path), passed_cases=sum(row['passed'] for row in cases.values()),
        requested_cases=len(cases), full_horizon_episodes=sum(row['full_horizon_episodes'] for row in cases.values()),
        requested_episodes=sum(row['requested_episodes'] for row in cases.values()),
        long_training_started=False, behavior_qualified=False)
except BaseException:
    state.update(status='failed', error=traceback.format_exc(), finished_at=datetime.now(timezone.utc).isoformat())
    raise
finally:
    save()

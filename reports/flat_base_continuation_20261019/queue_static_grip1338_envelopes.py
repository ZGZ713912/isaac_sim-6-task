"""Replay1328/1338 at both packaged RMCS heights under the current ground range."""
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import json
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from wheeled_tasks.chassis.flat_base_continuation import digest, encode
from compare_flat_base import compare

CANDIDATE = HERE / 'static_grip_candidate002'
STATE = RUN / 'static_grip1338_envelopes_queue001.json'
CONTRACT = CANDIDATE / 'envelopes_execution001.json'
if STATE.exists() or CONTRACT.exists():
    raise SystemExit('Preserve existing full-envelope replay outputs')
completion_path = CANDIDATE / 'completion001.json'
completion = json.loads(completion_path.read_text())
if (not completion['execution_verified'] or not all(completion['checks'].values())
        or not completion['basic_behavior_passed'] or completion['comparison']['lost_baseline_checks']):
    raise SystemExit('Require the sealed, passing eight-case static-only pilot')
old_queue = json.loads((RUN / 'static_grip1338_queue001.json').read_text())
if old_queue['status'] != 'complete' or any(Path('/proc/' + str(old_queue[key])).exists() for key in ('pid', 'worker_pid')):
    raise SystemExit('Wait for the previous bounded queue owners to exit')
checkpoints = dict(parent1328=RUN / 'pilot028_wheel_rate1328/model_001328.pt',
    static1338=RUN / 'pilot029_static_grip1338/model_001338.pt')
plan = CANDIDATE / 'plan.json'
phases = [f'ground025_{kind}_{height}_envelope001' for height in ('default', 'maximum')
    for kind in checkpoints]
if any((RUN / name).exists() or (RUN / (name + '.log')).exists() for name in phases):
    raise SystemExit('Preserve all previous envelope artifacts')
export_path = ROOT / 'models/flat_static_grip1338_20261020/verification.json'
export = json.loads(export_path.read_text())
if not export['deployment_interface_verified'] or export['checkpoint_sha256'] != digest(checkpoints['static1338']):
    raise SystemExit('Require the numerically verified exact static1338 ONNX candidate')
sources = list((ROOT / 'src/wheeled_tasks/chassis').glob('*.py')) + list((ROOT / 'src/wheeled_algo').glob('*.py'))
sources += [Path(__file__), ROOT / 'scripts/evaluate_flat_base.py', ROOT / 'scripts/compare_flat_base.py',
    ROOT / 'scripts/flat_rmcs_command_reference.py', ROOT / 'scripts/flat_default_height_profile.py',
    ROOT / 'scripts/flat_maximum_height_profile.py', ROOT / 'scripts/export_flat_base.py',
    plan, completion_path, export_path, *checkpoints.values()]
hashes = {str(path.relative_to(ROOT)):digest(path) for path in sources}
lock = (RUN / '.static_grip1338_envelopes.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
CONTRACT.write_bytes(encode(dict(schema='flat-static-grip-paired-envelope-replay-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
    cohorts=phases, seed=191029, episodes_per_case=4, cases_per_cohort=22,
    ground_material_range=[.2, .5], wheel_material=.1, policy_hz=50, physics_hz=1000,
    new_optimizer_updates=0, subsequent_training_queued=False, hardware_motion_started=False,
    qualification_scope='Development paired RMCS-height replay only; independent seed and self-righting remain separate')))
state = dict(schema='flat-static-grip-paired-envelope-serial-queue-v1', status='starting',
    pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
    contract_sha256=digest(CONTRACT), history=[], new_optimizer_updates=0,
    subsequent_training_queued=False, hardware_motion_started=False, heldout_seed_used=False)


def save():
    path = STATE.with_suffix('.tmp')
    path.write_bytes(encode(state))
    path.replace(STATE)


def frozen():
    if any(digest(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError('Full-envelope replay sources changed')


try:
    save()
    comparisons = {}
    for height in ('default', 'maximum'):
        cohort = {}
        for kind, checkpoint in checkpoints.items():
            name = f'ground025_{kind}_{height}_envelope001'
            frozen()
            state.update(status='running', phase=name)
            command = [sys.executable, str(ROOT / 'scripts/evaluate_flat_base.py'), '--plan', str(plan.relative_to(ROOT)),
                '--checkpoint', str(checkpoint), '--deployment-pd', '--rmcs-command-reference', '--envelope',
                '--packaged-' + height + '-height', '--output', str(RUN / name), '--seed', '191029']
            with (RUN / (name + '.log')).open('x') as log:
                worker = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES'},
                    stdout=log, stderr=subprocess.STDOUT)
                state['worker_pid'] = worker.pid
                state['worker_start_ticks'] = Path(f'/proc/{worker.pid}/stat').read_text().split(') ', 1)[1].split()[19]
                save()
                try:
                    code = worker.wait()
                except BaseException:
                    if worker.poll() is None:
                        worker.terminate()
                        worker.wait(timeout=60)
                    raise
            state['history'].append(dict(phase=name, exit_code=code))
            save()
            if code:
                raise RuntimeError('Native replay failed; no further phase launched: ' + name)
            report_path = RUN / name / 'evaluation.json'
            report = json.loads(report_path.read_text())
            if (report['status'] != 'evaluated' or report['checkpoint_sha256'] != digest(checkpoint)
                    or report['startup']['contact_material']['ground_static_range'] != [.2, .5]):
                raise RuntimeError('The exact current-material cohort is missing')
            cases = report['result']['cases']
            state['history'][-1].update(evaluation_sha256=digest(report_path),
                passed_cases=sum(row['passed'] for row in cases.values()), requested_cases=len(cases),
                full_horizon_episodes=sum(row['full_horizon_episodes'] for row in cases.values()),
                requested_episodes=sum(row['requested_episodes'] for row in cases.values()))
            cohort[kind] = report
            frozen()
            save()
        comparisons[height] = compare(cohort['parent1328'], cohort['static1338'])
        state['comparisons'] = comparisons
        save()
    state.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(), behavior_qualified=False)
except BaseException:
    state.update(status='failed', error=traceback.format_exc(), finished_at=datetime.now(timezone.utc).isoformat())
    raise
finally:
    save()

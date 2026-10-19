"""Serial development coverage after the bounded1280 trial has been sealed."""
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
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
state_path = RUN / 'low_gate1280_height_queue001.json'
contract_path = HERE / 'low_gate1280_height_contract001.json'
specs = (
    ('rmcs_default1280_envelope001', '--packaged-default-height',
     'rmcs_default_source_envelope001', 'rmcs_default1248_envelope001', 'rmcs_default1264_envelope001'),
    ('rmcs_maximum1280_envelope001', '--packaged-maximum-height',
     'rmcs_maximum_source_envelope001', 'rmcs_maximum1248_envelope001', 'rmcs_maximum1264_envelope001'))
if state_path.exists() or contract_path.exists() or any(
        (RUN/name).exists() or (RUN/(name+'.log')).exists() for name, *_ in specs):
    raise SystemExit('Preserve all completed and partial development replays')
prior_path = RUN / 'low_gate_1280_queue001.json'
seal_path = HERE / 'low_gate1280_completion001.json'
prior, seal = read(prior_path), read(seal_path)
if (prior['status'] != 'complete' or not seal['passed'] or seal['queue_sha256'] != sha(prior_path)
        or any(Path('/proc/'+str(prior[k])).exists() for k in ('pid', 'worker_pid'))):
    raise SystemExit('Wait for the bounded trial, basic screen, native proof and every owner to exit')
checkpoint = RUN / 'pilot025/model_001280.pt'
if sha(checkpoint) != seal['checkpoint_sha256']:
    raise SystemExit('Use the same complete1280 checkpoint')
hashes = dict(prior['source_sha256'])
hashes[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
if any(sha(ROOT/name) != value for name, value in hashes.items()):
    raise SystemExit('Reviewed execution sources changed')
for name, flag, *references in specs:
    if any(read(RUN/reference/'evaluation.json')['status'] != 'evaluated' for reference in references):
        raise SystemExit('Every matching source and historical reference must be complete')
lock = (RUN/'.low_gate1280_height_queue001.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
contract = dict(schema='flat-low-gate1280-height-development-contract-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), checkpoint_sha256=sha(checkpoint),
    bounded_completion_sha256=sha(seal_path), source_sha256=hashes,
    cohorts=[name for name, *_ in specs], development_seed=191029,
    unchanged='Same1280 checkpoint; existing full22 cases, original thresholds/horizons, RMCS command references and50Hz/1000Hz clocks.',
    purpose='Locate retained and lost low/high-height skills after the gate trial; a failed basic screen remains failed.',
    optimizer_updates=0, subsequent_training_queued=False, heldout_seed_used=False,
    behavior_qualified=False, hardware_motion_started=False)
contract_path.write_text(json.dumps(contract, indent=2, allow_nan=False)+'\n')
state = dict(schema='flat-low-gate1280-height-development-queue-v1',
    pid=os.getpid(), status='starting', started_at=datetime.now(timezone.utc).isoformat(),
    checkpoint_sha256=sha(checkpoint), contract_sha256=sha(contract_path),
    source_sha256=hashes, history=[], optimizer_updates=0,
    subsequent_training_queued=False, heldout_seed_used=False, hardware_motion_started=False)


def save():
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
    temporary.replace(state_path)


def frozen():
    if any(sha(ROOT/name) != value for name, value in hashes.items()):
        raise ValueError('Frozen execution file changed')


sys.path.insert(0, str(ROOT/'scripts'))
from compare_flat_base import compare

try:
    save()
    for name, flag, *references in specs:
        frozen()
        state.update(status='running', phase=name)
        command = [sys.executable, str(ROOT/'scripts/evaluate_flat_base.py'),
            '--seed', '191029', '--checkpoint', str(checkpoint), '--deployment-pd',
            '--rmcs-command-reference', flag, '--envelope', '--output', str(RUN/name)]
        with (RUN/(name+'.log')).open('x') as log:
            worker = subprocess.Popen(command, cwd=ROOT,
                env={**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES'}, stdout=log, stderr=subprocess.STDOUT)
            state['worker_pid'] = worker.pid
            save()
            code = worker.wait()
        state['history'].append(dict(phase=name, exit_code=code))
        save()
        if code:
            raise RuntimeError(name+' failed; no later child launched')
        frozen()
        path = RUN/name/'evaluation.json'
        data = read(path)
        if data['status'] != 'evaluated' or data['checkpoint_sha256'] != sha(checkpoint):
            raise ValueError('Completed replay identity differs')
        comparisons = {}
        for label, reference in zip(('source', 'parent1248', 'parent1264'), references, strict=True):
            old = read(RUN/reference/'evaluation.json')
            result = compare(old, data)
            relative = []
            for case_name, case in data['result']['cases'].items():
                vx, yaw, _ = case['command']
                if abs(vx) >= .75 or abs(yaw) >= 1.5:
                    for metric, tolerance in (('vx_mae_m_s', .02), ('yaw_mae_rad_s', .05)):
                        before = old['result']['cases'][case_name][metric]
                        if case[metric] > before+tolerance:
                            relative.append(dict(case=case_name, metric=metric,
                                before=before, after=case[metric], tolerance=tolerance))
            output = HERE/f'comparison_{name}_{label}.json'
            if output.exists():
                raise ValueError('Preserve the earlier comparison')
            output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            comparisons[label] = dict(path=str(output.relative_to(ROOT)), sha256=sha(output),
                lost_checks=result['lost_baseline_checks'], relative_regressions=relative)
        cases = data['result']['cases']
        state['history'][-1].update(evaluation_sha256=sha(path),
            passed_cases=sum(c['passed'] for c in cases.values()), requested_cases=len(cases),
            full_horizon_episodes=sum(c['full_horizon_episodes'] for c in cases.values()),
            requested_episodes=sum(c['requested_episodes'] for c in cases.values()), comparisons=comparisons)
        save()
    state.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat())
except Exception:
    state.update(status='failed', error=traceback.format_exc(), finished_at=datetime.now(timezone.utc).isoformat())
    raise
finally:
    save()

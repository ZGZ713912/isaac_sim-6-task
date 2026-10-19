"""Four serial development replays after the unchanged-v16 pilot is sealed."""
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
STATE = RUN / 'feedback1248_height_coverage_queue001.json'
PYTHON = '/home/noir/miniconda3/envs/isaacsim6/bin/python'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
specs = (
    ('rmcs_maximum_source_envelope001', None,
        ['--deployment-pd', '--rmcs-command-reference', '--packaged-maximum-height', '--envelope'], None),
    ('deployment1248_envelope001', 'pilot023/model_001248.pt',
        ['--deployment-pd', '--envelope'], 'deployment_source_envelope001'),
    ('rmcs1248_envelope001', 'pilot023/model_001248.pt',
        ['--deployment-pd', '--rmcs-command-reference', '--envelope'], 'rmcs_source_envelope001'),
    ('rmcs_maximum1248_envelope001', 'pilot023/model_001248.pt',
        ['--deployment-pd', '--rmcs-command-reference', '--packaged-maximum-height', '--envelope'],
        'rmcs_maximum_source_envelope001'))
if STATE.exists() or any((RUN/name).exists() or (RUN/(name+'.log')).exists() for name, *_ in specs):
    raise SystemExit('Preserve every existing coverage output and log')
lock = (RUN/'.feedback1248_height_coverage_queue001.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
previous_path = RUN/'feedback1248_queue001.json'
previous = read(previous_path)
seal_path = HERE/'feedback1248_queue_completion001.json'
seal = read(seal_path)
checkpoint = RUN/'pilot023/model_001248.pt'
if (previous['status'] != 'complete' or not seal['passed']
        or seal['queue_sha256'] != sha(previous_path)
        or seal['checkpoint_sha256'] != sha(checkpoint)
        or any(Path('/proc/'+str(previous[k])).exists() for k in ('pid','worker_pid'))):
    raise SystemExit('The complete pilot and all its native owners must exit and be sealed first')
hashes = dict(previous['source_sha256'])
sys.path.insert(0, str(ROOT/'scripts'))
from compare_flat_base import compare
state = dict(schema='flat-feedback1248-height-development-coverage-v1', pid=os.getpid(), status='starting',
    started_at=datetime.now(timezone.utc).isoformat(), previous_queue_sha256=sha(previous_path),
    previous_completion_sha256=sha(seal_path), checkpoint_sha256=sha(checkpoint),
    source_sha256=hashes, queue_entry_sha256=sha(Path(__file__)), history=[],
    new_optimizer_updates=0, subsequent_training_queued=False, hardware_motion_started=False,
    seed=191029, heldout_seed_used=False,
    scope='Raw nominal, RMCS nominal and RMCS maximum-height 1000Hz full envelopes; original maximum-height baseline.')


def save():
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
    temporary.replace(STATE)


try:
    save()
    for name, relative_checkpoint, flags, baseline_name in specs:
        if any(sha(ROOT/n) != v for n, v in hashes.items()) or sha(checkpoint) != state['checkpoint_sha256']:
            raise ValueError('Sealed pilot source or checkpoint changed')
        command = [PYTHON, 'scripts/evaluate_flat_base.py', '--seed', '191029', '--output', str(RUN/name), *flags]
        if relative_checkpoint:
            command += ['--checkpoint', str(RUN/relative_checkpoint)]
        state.update(status='running', phase=name)
        with (RUN/(name+'.log')).open('x') as log:
            worker = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'OMNI_KIT_ACCEPT_EULA':'YES'},
                stdout=log, stderr=subprocess.STDOUT)
            state['worker_pid'] = worker.pid
            save()
            code = worker.wait()
        state['history'].append(dict(phase=name, exit_code=code))
        save()
        if code:
            raise RuntimeError('Development replay failed: '+name)
        path = RUN/name/'evaluation.json'
        data = read(path)
        if (data['status'] != 'evaluated' or data['new_optimizer_updates'] != 0
                or data['physics_hz'] != 1000 or data['policy_hz'] != 50 or data['seed'] != 191029
                or len(data['result']['cases']) != 22
                or data.get('checkpoint_sha256') != (state['checkpoint_sha256'] if relative_checkpoint else None)):
            raise ValueError('Completed replay identity differs: '+name)
        comparisons = {}
        if baseline_name:
            baseline_path = RUN/baseline_name/'evaluation.json'
            baseline = read(baseline_path)
            paired = compare(baseline, data)
            relative = []
            for case_name, case in data['result']['cases'].items():
                vx, yaw, _ = case['command']
                if abs(vx) < .75 and abs(yaw) < 1.5:
                    continue
                old = baseline['result']['cases'][case_name]
                for metric, tolerance in (('vx_mae_m_s', .02), ('yaw_mae_rad_s', .05)):
                    if case[metric] > old[metric]+tolerance:
                        relative.append(dict(case=case_name, metric=metric, before=old[metric],
                            after=case[metric], tolerance=tolerance))
            paired.update(baseline_sha256=sha(baseline_path), candidate_sha256=sha(path),
                no_relative_regressions=not relative, relative_regressions=relative)
            comparison_path = HERE/('comparison_'+name+'_source.json')
            if comparison_path.exists():
                raise ValueError('Preserve existing comparison')
            comparison_path.write_text(json.dumps(paired, indent=2, allow_nan=False)+'\n')
            comparisons = dict(path=str(comparison_path.relative_to(ROOT)), sha256=sha(comparison_path),
                lost_checks=paired['lost_baseline_checks'], relative_regressions=relative)
        cases = data['result']['cases']
        state['history'][-1].update(evaluation_sha256=sha(path),
            passed_cases=sum(c['passed'] for c in cases.values()), requested_cases=len(cases),
            full_horizon_episodes=sum(c['full_horizon_episodes'] for c in cases.values()),
            requested_episodes=sum(c['requested_episodes'] for c in cases.values()), comparison=comparisons)
        save()
    state.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
        behavior_qualified=False, heldout_and_remaining_release_checks_unassessed=True)
    save()
except Exception:
    state.update(status='failed', error=traceback.format_exc())
    save()
    raise

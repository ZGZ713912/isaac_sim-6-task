"""One zero-optimizer causal replay, followed by four identical metric readers."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NATIVE = OUT/'action_relaxation_native001'
CHECKPOINT = ROOT/'runs/flat_base_20261021/transient_dwell_pilot002/model_001418.pt'
CAMPAIGN = OUT/'campaign.json'
DESIGN = OUT/'action_relaxation_design001.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update(**values):
    state = json.loads(CAMPAIGN.read_text())
    state.update(last_observed_at=datetime.now(timezone.utc).isoformat(), **values)
    temporary = CAMPAIGN.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(CAMPAIGN)


source_files = list((ROOT/'src').rglob('*.py')) + [ROOT/name for name in (
    'scripts/probe_flat_action_relaxation.py', 'scripts/flat_action_relaxation_probe.py',
    'scripts/probe_flat_stop_heading.py', 'scripts/probe_flat_speed_prior.py',
    'scripts/flat_rmcs_command_reference.py', 'scripts/analyze_flat_transient.py',
    'scripts/flat_transient_metrics.py')]
sources = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
checkpoint_sha = digest(CHECKPOINT)
guard_file = OUT/'action_relaxation_guard001.json'
with guard_file.open('x') as stream:
    json.dump(dict(checkpoint_sha256=checkpoint_sha, source_sha256=sources,
                   design_sha256=digest(DESIGN), new_optimizer_updates=0), stream, indent=2)


def guard():
    if digest(CHECKPOINT) != checkpoint_sha:
        raise RuntimeError('Diagnostic checkpoint changed')
    for name, sha in sources.items():
        if digest(ROOT/name) != sha:
            raise RuntimeError('Diagnostic source changed: '+name)


def run(command, log, phase):
    guard()
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES', 'OMP_NUM_THREADS': '4',
                 'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'})
        stat = Path(f'/proc/{child.pid}/stat').read_text().split()
        update(status=phase, supervisor_pid=os.getpid(), worker_pid=child.pid,
               current_log=str(log), command=command,
               diagnostic_worker_identity=dict(pid=child.pid, start_ticks=int(stat[21]), argv=command))
        code = child.wait()
    if code:
        raise RuntimeError(f'{phase} exited {code}; zero optimizer consumption')


try:
    run([sys.executable, '-B', 'scripts/probe_flat_action_relaxation.py',
        '--output', str(NATIVE), '--mode', 'checkpoint_compare',
        '--parent-checkpoint', str(ROOT/'runs/flat_base_20261020/pilot034_directional200_1346/model_001346.pt'),
        '--candidate', str(CHECKPOINT), '--candidate-variants', *[str(CHECKPOINT)]*3,
        '--comparison-indices', '0', '2', '3', '4', '5', '--envelope', '--transient-response',
        '--policy-plan', str(OUT/'dwell512_001/plan.json'), '--seed', '191129'],
        OUT/'action_relaxation_native001.log', 'causal_zero_optimizer_diagnostic_running')
    summaries = {}
    for role in ('trained', 'variant_1', 'variant_2', 'variant_3'):
        target = NATIVE/f'response_{role}.json'
        run([sys.executable, '-B', 'scripts/analyze_flat_transient.py', '--input', str(NATIVE),
             '--output', str(target), '--target-role', role],
            OUT/f'action_relaxation_analysis001_{role}.log', 'causal_diagnostic_analysis_running')
        result = json.loads(target.read_text())
        summaries[role] = result['admission']
    update(status='causal_diagnostic_assessment_ready', worker_pid=None,
        diagnostic_assessments=summaries, diagnostic_native=str(NATIVE),
        next_action='Compare the same-weight control with all/leg/wheel relaxation; require causal native evidence before a new frozen bounded training recipe. Diagnostic overrides cannot admit 5000 or certify the raw checkpoint.')
except BaseException as error:
    update(status='causal_diagnostic_failed', worker_pid=None, error=str(error))
    raise

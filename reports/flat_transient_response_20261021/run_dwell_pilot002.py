"""One frozen 64-update causal iteration followed by native paired assessment."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PLAN = OUT/'dwell512_001'
RUN = ROOT/'runs/flat_base_20261021/transient_dwell_pilot002'
NATIVE = OUT/'paired_native002'
CAMPAIGN = OUT/'campaign.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update(**values):
    state = json.loads(CAMPAIGN.read_text())
    state.update(last_observed_at=datetime.now(timezone.utc).isoformat(), **values)
    temporary = CAMPAIGN.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(CAMPAIGN)


sources = json.loads((PLAN/'preflight/progress.json').read_text())['source_sha256']
assessment_sources = {name: digest(ROOT/name) for name in ('scripts/probe_flat_speed_prior.py',
    'scripts/analyze_flat_transient.py', 'scripts/flat_transient_metrics.py')}


def guard():
    for name, sha in {**sources, **assessment_sources}.items():
        if digest(ROOT/name) != sha:
            raise RuntimeError('Frozen source changed: '+name)


def run(command, log, phase):
    guard()
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES', 'OMP_NUM_THREADS': '4',
                 'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'})
        update(status=phase, supervisor_pid=os.getpid(), worker_pid=child.pid,
            current_log=str(log), command=command, current_run=str(RUN))
        code = child.wait()
    if code:
        raise RuntimeError(f'{phase} exited {code}; preserve outputs and consumption')


try:
    run([sys.executable, '-B', 'scripts/train_flat_base.py', '--plan', str(PLAN/'plan.json'),
        '--resume', str(ROOT/'runs/flat_base_20261021/transient_pilot001/model_001354.pt'),
        '--resume-plan-revision', str(PLAN/'revision.json'), '--output', str(RUN),
        '--updates', '64'], OUT/'dwell_pilot002.log', 'short_training_running')
    progress = json.loads((RUN/'progress.json').read_text())
    if progress['status'] != 'block_complete' or progress['successful_updates_total'] != 1418:
        raise RuntimeError('Unexpected short block endpoint')
    update(optimizer_updates_this_request=72, actor_updates_this_request=70,
        aggregate_successful_local_consumption=1447, branch_successful_updates=1418,
        branch_consumed_updates=progress['consumed_optimizer_updates_total'],
        training_transitions_total=progress['training_transitions_total'],
        latest_checkpoint=progress['final_checkpoint'], latest_checkpoint_sha256=progress['final_checkpoint_sha256'])
    run([sys.executable, '-B', 'scripts/probe_flat_speed_prior.py', '--output', str(NATIVE),
        '--mode', 'checkpoint_compare', '--parent-checkpoint',
        str(ROOT/'runs/flat_base_20261020/pilot034_directional200_1346/model_001346.pt'),
        '--candidate', progress['final_checkpoint'], '--envelope', '--transient-response',
        '--policy-plan', str(PLAN/'plan.json'), '--comparison-indices', '0', '2',
        '--seed', '191129'], OUT/'paired_native002.log', 'paired_evaluation_running')
    run([sys.executable, '-B', 'scripts/analyze_flat_transient.py', '--input', str(NATIVE),
        '--output', str(NATIVE/'response.json')], OUT/'paired_analysis002.log', 'response_analysis_running')
    response = json.loads((NATIVE/'response.json').read_text())
    update(status='short_assessment_ready', worker_pid=None, assessment=str(NATIVE/'response.json'),
        exploratory_admission=response['admission'],
        next_action='Two four-replica confirmation seeds before 5000 updates if admitted; otherwise inspect the failed phases and design the next bounded causal iteration')
except BaseException as error:
    update(status='pipeline_failed', worker_pid=None, error=str(error),
        next_action='Inspect the exact frozen log and preserve all attempted optimizer consumption before any restart')
    raise

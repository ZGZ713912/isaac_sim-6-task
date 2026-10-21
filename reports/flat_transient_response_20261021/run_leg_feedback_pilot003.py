"""One frozen 64-actor update block, then initialized/trained/control replay."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PLAN = OUT/'leg_feedback512_002'
RUN = ROOT/'runs/flat_base_20261021/transient_leg_feedback_pilot003'
NATIVE = OUT/'paired_native003'
PARENT = ROOT/'runs/flat_base_20261021/transient_dwell_pilot002/model_001418.pt'
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
    'scripts/analyze_flat_transient.py', 'scripts/flat_transient_metrics.py',
    'src/wheeled_algo/directional_runtime.py', 'scripts/export_flat_speed_preview.py')}
frozen = dict(source_sha256={**sources, **assessment_sources},
              parent_sha256=digest(PARENT), plan_sha256=digest(PLAN/'plan.json'),
              revision_sha256=digest(PLAN/'revision.json'))
with (OUT/'leg_feedback_guard003.json').open('x') as stream:
    json.dump(frozen, stream, indent=2)


def guard():
    for name, sha in frozen['source_sha256'].items():
        if digest(ROOT/name) != sha:
            raise RuntimeError('Frozen source changed: '+name)
    if digest(PARENT) != frozen['parent_sha256']:
        raise RuntimeError('Learning parent changed')
    if (digest(PLAN/'plan.json') != frozen['plan_sha256']
            or digest(PLAN/'revision.json') != frozen['revision_sha256']):
        raise RuntimeError('Frozen plan or migration changed')


def run(command, log, phase):
    guard()
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES', 'OMP_NUM_THREADS': '4',
                 'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'})
        fields = Path(f'/proc/{child.pid}/stat').read_text().split()
        identity = dict(pid=child.pid, start_ticks=int(fields[21]), argv=command)
        supervisor_fields = Path(f'/proc/{os.getpid()}/stat').read_text().split()
        update(status=phase, supervisor_pid=os.getpid(), worker_pid=child.pid,
            process_identity=dict(supervisor=dict(pid=os.getpid(), start_ticks=int(supervisor_fields[21]),
                argv=[sys.executable, '-B', str(Path(__file__).resolve())]), worker=identity),
            current_log=str(log), command=command, current_run=str(RUN),
            active_plan=str(PLAN/'plan.json'), error=None)
        code = child.wait()
    if code:
        raise RuntimeError(f'{phase} exited {code}; preserve all optimizer consumption')


try:
    run([sys.executable, '-B', 'scripts/train_flat_base.py', '--plan', str(PLAN/'plan.json'),
        '--resume', str(PARENT), '--resume-plan-revision', str(PLAN/'revision.json'),
        '--output', str(RUN), '--updates', '64'], OUT/'leg_feedback_pilot003.log', 'short_training_running')
    progress = json.loads((RUN/'progress.json').read_text())
    if progress['status'] != 'block_complete' or progress['successful_updates_total'] != 1482:
        raise RuntimeError('Unexpected leg-feedback short block endpoint')
    update(optimizer_updates_this_request=136, actor_updates_this_request=134,
        aggregate_successful_local_consumption=1511, branch_successful_updates=1482,
        branch_consumed_updates=progress['consumed_optimizer_updates_total'],
        training_transitions_total=progress['training_transitions_total'],
        optimizer_update_in_flight=False, latest_checkpoint=progress['final_checkpoint'],
        latest_checkpoint_sha256=progress['final_checkpoint_sha256'])
    run([sys.executable, '-B', 'scripts/probe_flat_speed_prior.py', '--output', str(NATIVE),
        '--mode', 'checkpoint_compare', '--parent-checkpoint',
        str(ROOT/'runs/flat_base_20261020/pilot034_directional200_1346/model_001346.pt'),
        '--candidate', progress['final_checkpoint'], '--candidate-variants', str(PARENT),
        '--envelope', '--transient-response', '--policy-plan', str(PLAN/'plan.json'),
        '--comparison-indices', '0', '1', '2', '3', '--seed', '191129'],
        OUT/'paired_native003.log', 'paired_evaluation_running')
    run([sys.executable, '-B', 'scripts/analyze_flat_transient.py', '--input', str(NATIVE),
        '--output', str(NATIVE/'response.json')], OUT/'paired_analysis003.log', 'response_analysis_running')
    response = json.loads((NATIVE/'response.json').read_text())
    update(status='short_assessment_ready', worker_pid=None,
        assessment=str(NATIVE/'response.json'), exploratory_admission=response['admission'],
        next_action='Compare initialization, trained1482 and same-weight1418 control. Preserve fast coordinated braking intent. If admitted, use exact same checkpoint for confirmation seeds191130/191131 with four replicas, then extended-dwell --parent-leg-feedback long5000. Otherwise inspect actual phases before any additional short block.')
except BaseException as error:
    update(status='pipeline_failed', worker_pid=None, error=str(error),
        next_action='Inspect the frozen failure, true returned update count and partial rollout progress before any retry; never refund consumed updates')
    raise

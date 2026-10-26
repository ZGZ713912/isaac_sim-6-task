"""One serial native rollout diagnostic, explicitly zero optimizer attempts."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from train_robust_getup import digest,write_json
from run_gap_long_training import verify_restore

R=Path(__file__).parent
plan=json.loads((R/'credit_rollout_plan001.json').read_text())
target=R/'credit_rollout_supervisor001.json'
if target.exists():raise ValueError('This diagnostic is single use')
if Path(plan['output']).exists():raise ValueError('Never reuse a diagnostic output')
state=dict(status='starting',pid=os.getpid(),new_PPO=0,active_job=None,created_unix=time.time())
child=None;stopped=False
def save():write_json(target,state)
def stop(number,frame):
    global stopped
    stopped=True
    if child is not None and child.poll() is None:child.send_signal(number)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def verify():
    for name,expected in plan['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Changed diagnostic input: '+name)
try:
    verify();save()
    with Path(plan['log']).open('x') as log:
        child=subprocess.Popen(plan['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',OPENBLAS_NUM_THREADS='1'))
        state.update(status='collecting_native_rollout',active_job=dict(pid=child.pid,command=plan['command']));save()
        code=child.wait();child=None
    if stopped:raise InterruptedError('Diagnostic interrupted')
    if code:raise RuntimeError('Diagnostic native process failed')
    run=Path(plan['output']);report=json.loads((run/'progress.json').read_text())
    receipt=json.loads((run/'rollout_diagnosis.json').read_text())
    checks=dict(no_optimizer_attempts=report['new_optimizer_attempts']==0,
        no_optimizer_updates=report['new_optimizer_updates']==0,source_clock=report['completed_updates']==1501,
        stopped_as_planned=report['status']=='stopped',captured=receipt['status']=='captured_before_optimizer_attempt',
        exact_initial_copy=digest(run/'model_initial.pt')==digest(run/'model_stopped.pt'),
        data_sha=digest(run/'native_rollout_and_GAE.npz')==receipt['data_sha256'])
    verify_restore(Path(plan['source']),run/'model_initial.pt',R/'credit_rollout_initial_restore001.json')
    checks['all_restore_checks']=all(json.loads((R/'credit_rollout_initial_restore001.json').read_text())['checks'].values())
    verify();checks['all_frozen_inputs_exact']=True
    state.update(status='completed_diagnosis' if all(checks.values()) else 'failed_validation',checks=checks,
        active_job=None,output=str(run),data_sha256=receipt['data_sha256'],
        frozen_inputs=len(plan['source_sha256']),finished_unix=time.time())
    save()
    if not all(checks.values()):raise ValueError('Zero-PPO diagnostic validation failed')
except BaseException:
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM);child.wait()
    state.update(status='stopped' if stopped else 'failed',error=traceback.format_exc(),active_job=None,finished_unix=time.time());save()
    raise

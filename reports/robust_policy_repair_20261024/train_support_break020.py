"""Single-use, source-pinned native contact-objective PPO supervisor."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
R = Path(__file__).resolve().parent
plan = json.loads((R / 'plan020_support_break001.json').read_text())
receipt = R / 'training020_status.json'
if receipt.exists():
    raise ValueError('A training supervisor is single-use')
state = dict(schema='native-strict-contact-training-v1', status='starting', pid=os.getpid(),
    new_PPO=0, attempted_PPO=0, prior_repair_updates=314, prior_experiment_consumed=2694,
    source_clock=1501, qualification=None, hardware_started=False)
child = None
stopping = False


def save():
    state['updated_unix'] = time.time()
    tmp = receipt.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(receipt)


def verify():
    changed = [p for p, h in plan['source_sha256'].items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    if changed:
        raise ValueError('Pinned training source/input changed: ' + repr(changed))


def stop(number, frame):
    global stopping
    stopping = True
    state['requested_stop_signal'] = number
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
save()
try:
    verify()
    if (ROOT / plan['output']).exists():
        raise ValueError('Never reuse a PPO output directory')
    with (ROOT / plan['log']).open('x') as stream:
        child = subprocess.Popen(plan['command'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', OPENBLAS_NUM_THREADS='1'))
        state.update(status='training', child_pid=child.pid, started_unix=time.time())
        save()
        code = child.wait()
        child = None
    result = json.loads((ROOT / plan['output'] / 'progress.json').read_text())
    state.update(new_PPO=result['new_optimizer_updates'], attempted_PPO=result['new_optimizer_attempts'],
        completed_branch_clock=result['completed_updates'], native_status=result['status'], returncode=code)
    verify()
    if code or result['status'] != 'completed':
        raise RuntimeError('Native trial did not complete: ' + result['status'])
    if state['new_PPO'] != 16 or state['attempted_PPO'] != 16 or state['completed_branch_clock'] != 1517:
        raise ValueError('Unexpected native PPO accounting')
    state.update(status='completed_unassessed', all_frozen_inputs_verified=True)
except BaseException as error:
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)
        child.wait()
    state.update(status='stopped' if stopping else 'failed', error=str(error))
    raise
finally:
    state['finished_unix'] = time.time()
    save()

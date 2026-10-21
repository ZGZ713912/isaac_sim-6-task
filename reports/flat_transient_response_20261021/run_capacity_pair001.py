"""One serial no-optimizer capacity pair, after the existing replay exits."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
state_path = OUT/'capacity_pair001.json'


def save(value):
    temp = state_path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(state_path)


state = dict(status='waiting_for_replay_close', pid=os.getpid(), new_optimizer_updates=0, stages=[])
save(state)
deadline = time.monotonic()+600
while (Path('/proc/1213120/status').exists()
       and '\nState:\tZ' not in Path('/proc/1213120/status').read_text()):
    if time.monotonic() > deadline:
        raise RuntimeError('Existing replay did not close; no concurrent simulator started')
    time.sleep(3)
if json.loads((OUT/'paired_native001/report.json').read_text())['status'] != 'evaluated':
    raise RuntimeError('Replay failed; preserve the failure before capacity tests')
for count in (512, 128):
    output = ROOT/f'runs/flat_base_20261021/transient_capacity{count}_001'
    plan = OUT/f'campaign{count}_001'
    cmd = [sys.executable, '-B', 'scripts/train_flat_base.py', '--plan', str(plan/'plan.json'),
        '--resume', str(ROOT/'runs/flat_base_20261021/transient_pilot001/model_001354.pt'),
        '--resume-plan-revision', str(plan/'revision.json'), '--output', str(output),
        '--updates', '1', '--capacity-probe']
    with (OUT/f'capacity{count}_001.log').open('xb') as log:
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES', 'OMP_NUM_THREADS': '4',
                 'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'})
        state.update(status='capacity_running', population=count, worker_pid=child.pid, command=cmd)
        save(state)
        code = child.wait()
    report = json.loads((output/'progress.json').read_text())
    if code or report['status'] != 'capacity_complete' or not report['capacity_probe']['passed']:
        state.update(status='failed', failing_population=count, return_code=code, error=report.get('error'))
        save(state)
        raise RuntimeError('Capacity failed; do not launch a learner')
    state['stages'].append(dict(population=count, report=str(output/'progress.json'),
                               capacity_probe=report['capacity_probe']))
    save(state)
state.update(status='capacity_pair_complete', worker_pid=None)
save(state)

"""Serial, explicit zero-PPO separation of sensor delay and nominal dynamics."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
R = Path(__file__).resolve().parent
PYTHON = '/home/noir/miniconda3/envs/isaacsim6/bin/python'
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', PYTHONPATH='src:scripts', OMNI_KIT_ACCEPT_EULA='YES')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


assert json.loads((R / 'diagnosis_queue073.json').read_text())['status'] == 'completed'
assert (R / 'diagnosis073_closed_source_seal001.json').exists()
assert not (R / 'diagnosis_plan075.json').exists()
plan = deepcopy(json.loads((R / 'diagnosis_plan073.json').read_text()))
template = plan['jobs'][0]
plan['jobs'] = []
for suffix in ('zero_sensor_delay', 'nominal_dynamics'):
    job = deepcopy(template)
    job.update(name=suffix, output='runs/robust_policy_repair_20261024/diagnosis075_' + suffix,
        log=str((R / ('diagnosis075_' + suffix + '.log')).relative_to(ROOT)))
    job['command'][job['command'].index('--output') + 1] = job['output']
    if suffix == 'zero_sensor_delay':
        job['command'][1] = str((R / 'evaluate_sensor_delay075.py').relative_to(ROOT))
    else:
        job['command'].append('--nominal-dynamics')
    plan['jobs'].append(job)
plan['purpose'] = 'Source019 additional height-matched21cases: only zero sensor transport delay with all other samples exact; separately nominal dynamics/noise while retaining explicit IMUprofiles. No acceptance substitution or model promotion.'
for path in (Path(__file__).resolve(), R / 'evaluate_sensor_delay075.py', R / 'contact073_height_matched019.json'):
    plan['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
queue = R / 'diagnosis_queue075.py'
queue.write_text((R / 'diagnosis_queue065.py').read_text().replace('065', '075'))
plan['source_sha256'][str(queue.relative_to(ROOT))] = sha(queue)
write(R / 'diagnosis_plan075.json', plan)
subprocess.run([PYTHON, str(queue)], cwd=ROOT, env=ENV, check=True)
assert all(sha(ROOT / path) == expected for path, expected in plan['source_sha256'].items())
write(R / 'diagnosis075_closed_source_seal001.json', dict(schema='closed-native-queue-source-seal-v1',
    queue='075', source_files=len(plan['source_sha256']), all_frozen_sources_verified=True,
    new_PPO=0, qualification=None, created_unix=time.time()))
for job in plan['jobs']:
    subprocess.run([PYTHON, str(R / 'analyze_strict_standing002.py'), '--run', job['output'],
        '--output', str(R / ('contact075_' + job['name'] + '.json'))], cwd=ROOT, env=ENV, check=True)
print('Sensor-delay/nominal-dynamics diagnoses finished; no policy change.', flush=True)

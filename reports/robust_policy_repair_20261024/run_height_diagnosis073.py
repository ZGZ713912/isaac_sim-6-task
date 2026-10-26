"""Single-use additional height-matched diagnostic after021 physical tests."""
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


assert (R / 'diagnosis071_closed_source_seal001.json').exists()
assert json.loads((R / 'diagnosis_queue071.json').read_text())['status'] == 'completed'
assert not (R / 'diagnosis_plan073.json').exists(), 'Do not reuse a diagnosis'
plan = deepcopy(json.loads((R / 'diagnosis_plan071.json').read_text()))
template = plan['jobs'][0]
job = dict(template, name='height_matched_source019',
    checkpoint='runs/robust_policy_repair_20261024/pilot019_quiet_full/model_final.pt',
    output='runs/robust_policy_repair_20261024/diagnosis073_height_matched019',
    log=str((R / 'diagnosis073_height_matched019.log').relative_to(ROOT)))
job['command'] = template['command'].copy()
job['command'][1] = str((R / 'evaluate_height_matched073.py').relative_to(ROOT))
for flag, value in (('--checkpoint', job['checkpoint']), ('--output', job['output']),
                   ('--bank', str((R / 'pose_bank_height_matched073.json').relative_to(ROOT)))):
    job['command'][job['command'].index(flag) + 1] = value
plan.update(jobs=[job], purpose='Separate height-transition initialization from continuing static oscillation. Source019 unchanged; additional three COM-balanced height-matched closed-chain birth poses. Original acceptance retained, no promotion.')
for path in (Path(__file__).resolve(), R / 'build_height_matched_starts073.py', R / 'evaluate_height_matched073.py',
             R / 'pose_bank_height_matched073.json', R / 'height_matched_starts073_geometry001.json'):
    plan['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
plan['source_sha256'].update(json.loads((R / 'height_matched_starts073_geometry001.json').read_text())['sources_sha256'])
queue = R / 'diagnosis_queue073.py'
queue.write_text((R / 'diagnosis_queue065.py').read_text().replace('065', '073'))
plan['source_sha256'][str(queue.relative_to(ROOT))] = sha(queue)
write(R / 'diagnosis_plan073.json', plan)
subprocess.run([PYTHON, str(queue)], cwd=ROOT, env=ENV, check=True)
assert all(sha(ROOT / path) == expected for path, expected in plan['source_sha256'].items())
write(R / 'diagnosis073_closed_source_seal001.json', dict(schema='closed-native-queue-source-seal-v1',
    queue='073', source_files=len(plan['source_sha256']), all_frozen_sources_verified=True,
    new_PPO=0, qualification=None, created_unix=time.time()))
subprocess.run([PYTHON, str(R / 'analyze_strict_standing002.py'), '--run', job['output'],
    '--output', str(R / 'contact073_height_matched019.json')], cwd=ROOT, env=ENV, check=True)
print('Additional height-matched diagnostic finished; original acceptance unchanged.', flush=True)

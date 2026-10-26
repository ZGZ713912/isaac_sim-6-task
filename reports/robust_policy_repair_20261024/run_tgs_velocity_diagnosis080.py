"""Run one labelled numerical ablation after078 closes; never launch PPO."""
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


def run(*args):
    subprocess.run([PYTHON, *args], cwd=ROOT, env=ENV, check=True)


assert not (R / 'diagnosis_plan080.json').exists(), 'Single-use ablation'
assert json.loads((R / 'diagnosis_queue078.json').read_text())['status'] == 'completed'
assert (R / 'diagnosis078_closed_source_seal001.json').exists()
prior = json.loads((R / 'diagnosis_plan078.json').read_text())
assert all(sha(ROOT / p) == h for p, h in prior['source_sha256'].items())
plan = deepcopy(json.loads((R / 'diagnosis_plan073.json').read_text()))
plan['source_sha256'] = prior['source_sha256'].copy()
job = plan['jobs'][0]
job.update(name='tgs_velocity1_height_matched019',
    output='runs/robust_policy_repair_20261024/diagnosis080_tgs_velocity1',
    log=str((R / 'diagnosis080_tgs_velocity1.log').relative_to(ROOT)))
job['command'][1] = str((R / 'evaluate_tgs_velocity080.py').relative_to(ROOT))
job['command'][job['command'].index('--output') + 1] = job['output']
queue = R / 'diagnosis_queue080.py'
queue.write_text((R / 'diagnosis_queue065.py').read_text().replace('065', '080'))
for path in (Path(__file__).resolve(), R / 'evaluate_tgs_velocity080.py', queue,
    R / 'diagnosis078_closed_source_seal001.json'):
    plan['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
plan.update(purpose='Numerical diagnostic only:019 height-matched073, solver64/32 ->64/1, no actor/reward/DR/dt/reset change. Original qualification remains64/32; no source or policy adoption.',
    new_PPO=0, planned_PPO_after_review=0, new_PPO_admitted=0,
    prior_completed_repair_PPO=378, prior_consumed_experiment=2758,
    reference_documentation='https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/simulation_control/simulation_control.html',
    qualification=None, created_unix=time.time())
write(R / 'diagnosis_plan080.json', plan)
run(str(queue))
assert all(sha(ROOT / p) == h for p, h in plan['source_sha256'].items())
write(R / 'diagnosis080_closed_source_seal001.json', dict(schema='closed-native-queue-source-seal-v1',
    queue='080', all_frozen_sources_verified=True, source_files=len(plan['source_sha256']),
    new_PPO=0, qualification=None, created_unix=time.time()))
analysis = R / 'contact080_tgs_velocity1.json'
comparison = R / 'comparison080_tgs_velocity1_vs073.json'
run(str(R / 'analyze_strict_standing002.py'), '--run', job['output'], '--output', str(analysis))
run(str(R / 'compare_strict_standing001.py'), '--before', str(R / 'contact073_height_matched019.json'),
    '--after', str(analysis), '--output', str(comparison))
result = json.loads(comparison.read_text())
result.update(physics_different='TGS articulation velocity iterations32->1 only; actual21native articulation readbacks saved.',
    interpretation='Numerical sensitivity diagnostic only; different physics is not a qualified policy improvement.', qualification=None)
write(comparison, result)
print('080 numerical ablation completed; no PPO or adoption.', flush=True)

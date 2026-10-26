"""After020 finishes, audit and run exactly one unchanged native standing comparison."""
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def run(*args):
    subprocess.run([PYTHON, *args], cwd=ROOT, env=ENV, check=True)


plan = json.loads((R / 'plan021_quiet_continuation001.json').read_text())
status = json.loads((R / 'training021_status.json').read_text())
assert status['status'] == 'completed_unassessed' and status['new_PPO'] == status['attempted_PPO'] == 32
assert not (R / 'diagnosis_plan071.json').exists(), 'Never admit the same evaluation twice'
run(str(R / 'audit_quiet_continuation_endpoint021.py'))
audit = json.loads((R / 'quiet_continuation_round021_endpoint001.json').read_text())
assert all(audit['checks'].values())
assert all(digest(ROOT / name) == expected for name, expected in plan['source_sha256'].items())
seal = R / 'training021_closed_source_seal001.json'
write(seal, dict(schema='closed-native-training-source-seal-v1', round='021',
    all_frozen_sources_verified=True, source_files=len(plan['source_sha256']),
    audit_sha256=digest(R / 'quiet_continuation_round021_endpoint001.json'), new_PPO=32,
    repair_completed_PPO=362, experiment_consumed=2742, qualification=None, created_unix=time.time()))
evaluation = json.loads((R / 'diagnosis_plan065.json').read_text())
evaluation['source_sha256'] = plan['source_sha256'].copy()
for path in (audit['checkpoint'], str(R / 'quiet_continuation_round021_endpoint001.json'),
             str(seal.relative_to(ROOT)), str(Path(__file__).relative_to(ROOT)),
             str((R / 'analyze_strict_standing002.py').relative_to(ROOT)),
             str((R / 'compare_strict_standing001.py').relative_to(ROOT))):
    evaluation['source_sha256'][path] = digest(ROOT / path)
import copy
import torch
mid = 'runs/robust_policy_repair_20261024/pilot021_quiet_continuation/model_1528.pt'
assert torch.load(ROOT / mid, map_location='cpu', weights_only=False)['infos']['completed_updates'] == 1529
template = evaluation['jobs'][0]
evaluation['jobs'] = []
for suffix, checkpoint in (('mid12', mid), ('final32', audit['checkpoint'])):
    evaluation['source_sha256'][checkpoint] = digest(ROOT / checkpoint)
    job = copy.deepcopy(template)
    job.update(name='quiet021_' + suffix, checkpoint=checkpoint,
        output='runs/robust_policy_repair_20261024/diagnosis071_quiet021_' + suffix,
        log=str((R / ('diagnosis071_quiet021_' + suffix + '.log')).relative_to(ROOT)))
    for flag, value in (('--checkpoint', checkpoint), ('--output', job['output'])):
        job['command'][job['command'].index(flag) + 1] = value
    evaluation['jobs'].append(job)
evaluation.update(purpose='Two actual021 checkpoints:12new/32new updates, both vs019 source, same21cases/8s/fullDR/native1ms/seed241201.',
    prior_completed_repair_PPO=362, prior_consumed_experiment=2742)
write(R / 'diagnosis_plan071.json', evaluation)
(R / 'diagnosis_queue071.py').write_text((R / 'diagnosis_queue065.py').read_text().replace('065', '071'))
run(str(R / 'diagnosis_queue071.py'))
assert all(digest(ROOT / name) == expected for name, expected in evaluation['source_sha256'].items())
write(R / 'diagnosis071_closed_source_seal001.json', dict(schema='closed-native-queue-source-seal-v1', queue='071',
    all_frozen_sources_verified=True, source_files=len(evaluation['source_sha256']), new_PPO=0,
    qualification=None, created_unix=time.time()))
for job in evaluation['jobs']:
    suffix = job['name'].removeprefix('quiet021_')
    analysis = str((R / ('contact071_quiet021_' + suffix + '.json')).relative_to(ROOT))
    run(str(R / 'analyze_strict_standing002.py'), '--run', job['output'], '--output', analysis)
    run(str(R / 'compare_strict_standing001.py'), '--before', str(R / 'contact065_strict_trained019.json'),
        '--after', analysis, '--output', str(R / ('comparison071_' + suffix + '_vs019.json')))
print('021 endpoint audit, two strict physical checkpoints and paired comparisons completed; no qualification.', flush=True)

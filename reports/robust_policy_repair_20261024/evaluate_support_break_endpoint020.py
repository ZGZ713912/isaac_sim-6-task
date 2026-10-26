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


plan = json.loads((R / 'plan020_support_break001.json').read_text())
status = json.loads((R / 'training020_status.json').read_text())
assert status['status'] == 'completed_unassessed' and status['new_PPO'] == status['attempted_PPO'] == 16
assert not (R / 'diagnosis_plan068.json').exists(), 'Never admit the same evaluation twice'
run(str(R / 'audit_support_break_endpoint020.py'))
audit = json.loads((R / 'support_break_round020_endpoint001.json').read_text())
assert all(audit['checks'].values())
assert all(digest(ROOT / name) == expected for name, expected in plan['source_sha256'].items())
seal = R / 'training020_closed_source_seal001.json'
write(seal, dict(schema='closed-native-training-source-seal-v1', round='020',
    all_frozen_sources_verified=True, source_files=len(plan['source_sha256']),
    audit_sha256=digest(R / 'support_break_round020_endpoint001.json'), new_PPO=16,
    repair_completed_PPO=330, experiment_consumed=2710, qualification=None, created_unix=time.time()))
evaluation = json.loads((R / 'diagnosis_plan065.json').read_text())
evaluation['source_sha256'] = plan['source_sha256'].copy()
for path in (audit['checkpoint'], str(R / 'support_break_round020_endpoint001.json'),
             str(seal.relative_to(ROOT)), str(Path(__file__).relative_to(ROOT)),
             str((R / 'analyze_strict_standing002.py').relative_to(ROOT)),
             str((R / 'compare_strict_standing001.py').relative_to(ROOT))):
    evaluation['source_sha256'][path] = digest(ROOT / path)
job = evaluation['jobs'][0]
job.update(name='trained_support_break020', checkpoint=audit['checkpoint'],
    output='runs/robust_policy_repair_20261024/diagnosis068_trained_support_break020',
    log=str((R / 'diagnosis068_trained_support_break020.log').relative_to(ROOT)))
for flag, value in (('--checkpoint', job['checkpoint']), ('--output', job['output'])):
    job['command'][job['command'].index(flag) + 1] = value
evaluation.update(purpose='Actual endpoint020 support-break objective vs019 and058; same21cases,8s,fullDR,seed241201,native1ms.',
    prior_completed_repair_PPO=330, prior_consumed_experiment=2710)
write(R / 'diagnosis_plan068.json', evaluation)
(R / 'diagnosis_queue068.py').write_text((R / 'diagnosis_queue065.py').read_text().replace('065', '068'))
run(str(R / 'diagnosis_queue068.py'))
assert all(digest(ROOT / name) == expected for name, expected in evaluation['source_sha256'].items())
write(R / 'diagnosis068_closed_source_seal001.json', dict(schema='closed-native-queue-source-seal-v1', queue='068',
    all_frozen_sources_verified=True, source_files=len(evaluation['source_sha256']), new_PPO=0,
    qualification=None, created_unix=time.time()))
analysis = 'reports/robust_policy_repair_20261024/contact068_strict_trained020.json'
run(str(R / 'analyze_strict_standing002.py'), '--run', job['output'], '--output', analysis)
for before, output in (('contact058_strict_grounded002.json', 'comparison068_trained020_vs_source.json'),
                       ('contact065_strict_trained019.json', 'comparison068_trained020_vs019.json')):
    run(str(R / 'compare_strict_standing001.py'), '--before', str(R / before), '--after', analysis,
        '--output', str(R / output))
print('020 native audit, support evaluation and paired comparisons completed; qualification remains null.', flush=True)

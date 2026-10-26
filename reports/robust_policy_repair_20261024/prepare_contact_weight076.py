"""Prepare a paired doubled-contact-weight preflight; do not launch native work."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src')]
from wheeled_tasks.chassis.standing_contact_reward import contract

R = Path(__file__).resolve().parent
assert not (R / 'diagnosis_plan076.json').exists()
plan = deepcopy(json.loads((R / 'diagnosis_plan075.json').read_text()))
job = deepcopy(json.loads((R / 'diagnosis_plan064.json').read_text())['jobs'][0])
job.update(name='contact_weight16_native_zero',
    output='runs/robust_policy_repair_20261024/diagnosis076_contact_weight16_zero',
    log=str((R / 'diagnosis076_contact_weight16_zero.log').relative_to(ROOT)))
job['command'][1] = str((R / 'collect_contact_weight076.py').relative_to(ROOT))
for flag, value in (('--output', job['output']), ('--standing-contact-weight', '16')):
    job['command'][job['command'].index(flag) + 1] = value
plan.update(jobs=[job], purpose='ZeroPPO paired reward-only preflight vs064/019 source. Dense contact weight8/s to16/s, existing bounds. Same root5350/seed/batch/physics/mean/conditionedstd/Adam/RNG. No event cost, no sampler or runtime change.',
    proposed_contact_contract=contract(16., True), prior_completed_repair_PPO=362, prior_consumed_experiment=2742,
    planned_PPO_after_review=16, new_PPO_admitted=0)
paths = [Path(__file__).resolve(), R / 'collect_contact_weight076.py',
    ROOT / 'runs/robust_policy_repair_20261024/diagnosis064_quiet_full_zero/rollout_diagnosis.json']
queue = R / 'diagnosis_queue076.py'
queue.write_text((R / 'diagnosis_queue065.py').read_text().replace('065', '076'))
paths.append(queue)
for path in paths:
    plan['source_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
(R / 'diagnosis_plan076.json').write_text(json.dumps(plan, indent=2) + '\n')
print('Prepared one zero-PPO native preflight only; do not overlap075.')

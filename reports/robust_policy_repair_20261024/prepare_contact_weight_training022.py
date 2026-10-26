"""Admit one sixteen-update reward-only comparison after actual076 verification."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
R = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not (R / 'plan022_contact_weight001.json').exists()
zero = json.loads((R / 'contact_weight_native_zero_audit076.json').read_text())
assert all(zero['checks'].values()) and zero['new_PPO'] == 0
preflight = json.loads((R / 'diagnosis_plan076.json').read_text())
assert all(sha(ROOT / path) == value for path, value in preflight['source_sha256'].items())
plan = json.loads((R / 'plan019_quiet_full001.json').read_text())
plan['command'][1] = str((R / 'run_contact_weight_pilot022.py').relative_to(ROOT))
plan.update(output='runs/robust_policy_repair_20261024/pilot022_contact_weight16',
    log=str((R / 'pilot022_contact_weight16.log').relative_to(ROOT)),
    source_sha256=preflight['source_sha256'].copy(),
    prior_repair_consumed=362, prior_experiment_consumed=2742,
    zero_PPO_audit=str((R / 'contact_weight_native_zero_audit076.json').relative_to(ROOT)),
    training_contact_contract=preflight['proposed_contact_contract'],
    hypothesis='Paired with019: same5350 V8 root, mean/std, fourRNGs, Adam, seed,256x512/2.56s, DR and physical clocks. Only dense contact weight8/s to16/s. No break event cost or runtime control change.',
    acceptance='Same21cases first: physical1ms support from firststep, geometry and leg vibration together; no promotion from reward/loss/survival. No automatic extension.',
    created_unix=time.time())
for flag, value in (('--zero-rollout-reference', 'runs/robust_policy_repair_20261024/diagnosis076_contact_weight16_zero/rollout_diagnosis.json'),
                    ('--output', plan['output']), ('--standing-contact-weight', '16')):
    plan['command'][plan['command'].index(flag) + 1] = value
for path in (R / 'run_contact_weight_pilot022.py', R / 'train_contact_weight022.py',
    R / 'audit_contact_weight_endpoint022.py', R / 'prepare_contact_weight_training022.py',
    R / 'audit_contact_weight_zero076.py', R / 'contact_weight_native_zero_audit076.json',
    R / 'diagnosis076_closed_source_seal001.json',
    ROOT / 'runs/robust_policy_repair_20261024/diagnosis076_contact_weight16_zero/rollout_diagnosis.json'):
    plan['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
assert '--standing-contact-break-cost' not in plan['command']
(R / 'plan022_contact_weight001.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(dict(admitted_once_PPO=16, prior_repair=362, prior_experiment=2742,
    frozen_inputs=len(plan['source_sha256']), changed_training_field='contact_weight8_to16', qualification=None)))

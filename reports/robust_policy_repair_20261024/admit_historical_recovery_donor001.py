"""Admit one paired nominal donor diagnosis after the preceding queue is sealed."""
from pathlib import Path
import json
import os
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
prior = json.loads((R / 'diagnosis_plan052.json').read_text())
assert json.loads((R / 'diagnosis_queue052.json').read_text())['status'] == 'completed'
assert json.loads((R / 'diagnosis052_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert all(digest(p) == s for p, s in prior['source_sha256'].items())
preflight = json.loads((R / 'historical_recovery_donor_preflight001.json').read_text())
assert all(preflight['checks'].values())
wrapper = R / 'evaluate_historical_recovery_donor001.py'
assert digest(wrapper) == preflight['wrapper_sha256']
assert digest('scripts/evaluate_robust_getup_batch.py') == preflight['batch_sha256']
assert digest(preflight['checkpoint']) == preflight['checkpoint_sha256']
alias_dir = B / 'historical_donor_reference_alias001'
assert not alias_dir.exists()
alias_dir.mkdir()
reference = alias_dir / 'model_reference.pt'
os.link(preflight['checkpoint'], reference)
assert digest(reference) == preflight['checkpoint_sha256']
write_json(alias_dir / 'alias_receipt.json', dict(schema='exact-checkpoint-reference-alias-v1',
    source=preflight['checkpoint'], alias=str(reference), sha256=digest(reference),
    unchanged_checkpoint_bytes=True, new_training=False))
template = json.loads((R / 'diagnosis_plan051.json').read_text())['jobs'][0]
job = dict(template)
output = B / 'diagnosis053_historical_recovery_donor'
assert not output.exists()
cmd = list(template['command'])
cmd[1] = str(wrapper)
for key, value in (('--checkpoint', preflight['checkpoint']),
                   ('--reference-checkpoint', str(reference)), ('--output', str(output))):
    cmd[cmd.index(key) + 1] = value
job.update(name='historical_recovery_donor', checkpoint=preflight['checkpoint'],
    output=str(output), log=str(R / 'diagnosis053_historical_recovery_donor.log'), command=cmd)
protocol = dict(schema='historical-recovery-neural-donor-protocol-v1',
    rationale='The pre-long-training096 policy recorded actual left moving-fall recovery under the original push. Current9a and output-only pilots fail that target. Test the old six learned fallen-network tensors in the current authenticated mean, retaining the current final action decoder. This tests learned-weight loss, not reproduction of historical096 behavior.',
    counterfactual='Only six existing recovery-network tensors; identical-feature basis verified. Old checkpoint is a hash-verified raw tensor donor whose incompatible historical decoder is never loaded.',
    control='Exact unchanged9a same-SHA hardlink reference, independently authenticated; original35cases,2replicas,seed211131,12s,nominal domain0 and original push.',
    target='At least one actual moving fall, recovery and resumed command; compare all recovery families, survival, strict followup, impact and native mechanical limits.',
    boundaries='Mean override is ineligible for saved-checkpoint qualification, PPO, export or adoption. Original qualification requirements remain unchanged.',
    new_PPO=0, actual_experiment_consumed=2654, repair_optimizer_updates=274,
    hardware_started=False, qualification=None, created_unix=time.time())
write_json(R / 'historical_recovery_donor_protocol001.json', protocol)
queue = R / 'diagnosis_queue053.py'
assert not queue.exists() and not (R / 'diagnosis_plan053.json').exists()
queue.write_text((R / 'diagnosis_queue052.py').read_text().replace('plan052', 'plan053').replace('queue052', 'queue053'))
frozen = dict(prior['source_sha256'])
extra = [wrapper, R / 'historical_recovery_donor_probe001.py', R / 'preflight_historical_recovery_donor001.py',
    R / 'historical_recovery_donor_preflight001.json', R / 'historical_recovery_donor_protocol001.json',
    R / 'review_quiet_common_reverse001.py', R / 'quiet_common_reverse_development_decision001.json',
    R / 'diagnosis052_closed_source_seal001.json', reference, alias_dir / 'alias_receipt.json',
    Path(preflight['runtime']['donor_checkpoint']), queue, Path(__file__)]
frozen.update({str(p): digest(p) for p in extra})
write_json(R / 'diagnosis_plan053.json', dict(schema='serial-native-diagnosis-v2', jobs=[job],
    source_sha256=frozen, protocol=str(R / 'historical_recovery_donor_protocol001.json'),
    new_PPO=0, actual_experiment_consumed=2654, repair_optimizer_updates=274,
    qualification=None, hardware_started=False, created_unix=time.time()))
progress = json.loads((R / 'progress.json').read_text())
progress.update(status='historical_recovery_donor_zero_PPO_diagnosis',
    active_evaluation=str(output), active_diagnosis_queue=str(R / 'diagnosis_queue053.json'),
    latest_completed_evaluation=str(B / 'diagnosis052_quiet_common_reverse_development'),
    latest_research_decision='052 reject:16/21 standing,one lost;050 yaw mechanism improves,17/21 still fails;051 original moving target0.',
    new_PPO_admitted=0, planned_PPO='No PPO admitted; six historical learned recovery tensors tested with current decoder.',
    updated_unix=time.time())
write_json(R / 'progress.json', progress)
print(json.dumps(dict(plan=str(R / 'diagnosis_plan053.json'), frozen_inputs=len(frozen), command=cmd)))

"""Diagnose the original moving-fall target, without overturning pilot rejection."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
last_plan = json.loads((R / 'diagnosis_plan046.json').read_text())
assert json.loads((R / 'diagnosis_queue046.json').read_text())['status'] == 'completed'
assert json.loads((R / 'diagnosis046_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert all(digest(p) == sha for p, sha in last_plan['source_sha256'].items())
negative = json.loads((R / 'motion_round015_fixed_decision001.json').read_text())
assert negative['decision'] == 'reject_extension_and_adoption'
comparison = json.loads((R / 'interim_practical_consistency001.json').read_text())
old, new = comparison['models'][0], comparison['models'][2]
assert old['label'] == 'best9a' and new['label'] == 'motion015'
assert new['groups']['fallen']['diagnostic_8s_tail95_success'] == 5
assert old['groups']['fallen']['diagnostic_8s_tail95_success'] == 4
assert new['groups']['entry']['diagnostic_8s_tail95_success'] == old['groups']['entry']['diagnostic_8s_tail95_success'] == 8
cp = negative['checkpoint']
assert digest(cp) == negative['checkpoint_sha256']
original_plan = json.loads((R / 'diagnosis_plan037.json').read_text())
template = next(x for x in original_plan['jobs'] if x['name'] == 'quiet_stance050_nominal_pair')
job = dict(template)
output = B / 'diagnosis047_motion015_nominal_target_diagnosis'
assert not output.exists()
job.update(name='motion015_original_moving_fall_diagnosis', checkpoint=cp, output=str(output),
    log=str(R / 'diagnosis047_motion015_nominal_target_diagnosis.log'))
job['command'] = [cp if x == template['checkpoint'] else str(output) if x == template['output'] else x
                  for x in template['command']]
queue = R / 'diagnosis_queue047.py'
assert not queue.exists() and not (R / 'diagnosis_plan047.json').exists()
queue.write_text((R / 'diagnosis_queue046.py').read_text().replace('plan046', 'plan047').replace('queue046', 'queue047'))
protocol = dict(schema='motion015-original-target-diagnostic-v1',
    rationale='The rejected015 retains the original8/12 diagnostic endpoint entry successes, increases fallen8s-tail95 successes4to5/6 and survives6/6 with0impacts. Its strict uninterrupted latch and leg-speed nonregression still fail. A bounded zero-PPO test of the original12s moving-fall target is needed to distinguish useful anticipatory motion from an isolated8s endpoint gain.',
    unchanged='Original35cases,2replicas,211131,12s,nominal domains,original push force/duration,all contacts/impulse/torque/closure guards. Exact saved015 mean, no counterfactual actor.',
    original_pilot_rejection_preserved=True, diagnostic_only=True,
    target='Actual in-episode moving fall followed by recovery and resumed command under the frozen practical endpoint definition; compare9a0 and all family/ordinary motion outcomes.',
    scope='No automatic candidate adoption, PPO admission/extension, export or qualification. Original final protocol requires two seeds,1346 parent vibration comparison,30s standing and fresh randomized holdouts.',
    new_PPO=0, actual_experiment_consumed=2646, repair_optimizer_updates=266,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'motion015_target_diagnostic_protocol001.json', protocol)
frozen = dict(last_plan['source_sha256'])
extras = [R / 'diagnosis046_closed_source_seal001.json', R / 'followup_credit_native_audit001.json',
    R / 'interim_practical_consistency001.json', R / 'audit_interim_practical_consistency001.py',
    R / 'motion015_target_diagnostic_protocol001.json', Path(cp), queue, Path(__file__)]
frozen.update({str(p): digest(p) for p in extras})
plan = dict(schema='serial-native-diagnosis-v2', jobs=[job], source_sha256=frozen,
    protocol=str(R / 'motion015_target_diagnostic_protocol001.json'), new_PPO=0,
    actual_experiment_consumed=2646, repair_optimizer_updates=266,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'diagnosis_plan047.json', plan)
p = json.loads((R / 'progress.json').read_text())
p.update(status='rejected_motion015_original_target_diagnosis', active_diagnosis_queue=str(R / 'diagnosis_queue047.json'),
    active_evaluation=str(output), active_training=None, new_PPO_admitted=0,
    planned_PPO='No new PPO; original12s diagnostic only,015 rejection preserved.', updated_unix=time.time())
write_json(R / 'progress.json', p)
print(json.dumps(dict(plan=str(R / 'diagnosis_plan047.json'), frozen_inputs=len(frozen), new_PPO=0)))

"""One original12s target diagnostic of the earlier legs-only update,0PPO."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
prior = json.loads((R / 'diagnosis_plan050.json').read_text())
assert json.loads((R / 'diagnosis_queue050.json').read_text())['status'] == 'completed'
assert json.loads((R / 'diagnosis050_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert all(digest(p) == s for p, s in prior['source_sha256'].items())
preflight = json.loads((R / 'fallen_legs_batch_preflight001.json').read_text())
assert all(preflight['checks'].values())
wrapper = R / 'evaluate_fallen_legs_original_target001.py'
assert digest(wrapper) == preflight['wrapper_sha256']
assert digest(preflight['checkpoint']) == preflight['checkpoint_sha256']
assert digest('scripts/evaluate_robust_getup_batch.py') == preflight['batch_sha256']
old = json.loads((R / 'fallen_partition_observed_decision001.json').read_text())['results']['legs']
assert old['metrics']['entry']['followup_success'] == 8 and old['metrics']['fallen']['recovered'] == 5
assert old['metrics']['fallen']['followup_success'] == 2 and old['impacts'] == 0
assert not old['checks']['fallen_speed']
template = json.loads((R / 'diagnosis_plan047.json').read_text())['jobs'][0]
job = dict(template)
output = B / 'diagnosis051_fallen_legs_original_target'
assert not output.exists()
cp = preflight['checkpoint']
job.update(name='fallen_legs_original_target', checkpoint=cp, output=str(output),
    log=str(R / 'diagnosis051_fallen_legs_original_target.log'))
job['command'] = [str(wrapper) if x == 'scripts/evaluate_robust_getup_batch.py'
    else cp if x == template['checkpoint'] else str(output) if x == template['output']
    else x for x in template['command']]
protocol = dict(schema='fallen-legs-original-target-protocol-v1',
    rationale='The existing014 legs-only diagnostic preserves8 entry followups, improves fallen recovery4to5 with2strict followups and0impacts. Its24.14862rad/s leg peak fails the prior24.11665 nonregression gate, whose rejection is preserved. Unlike the motion-only branch, this existing output is sensitive during the first100ms of actual moving falls. Test the unmeasured original12s moving-fall target once, without admitting any training or adoption.',
    unchanged='Original35cases,2replicas,seed211131,12s,nominal domains,original push,3mm closure,20Ns impulse and other guards. Same3a4f reference for descriptive comparison; this is not the1346-parent formal confirmation.',
    intervention='Retain only the actual014 learned leg rows of the existing fallen output; restore its wheel rows from its exact1501 source. Runtime layer hash must equal040. No other state/function/sensor/physics changes.',
    target='At least one actual in-episode moving fall followed by recovery and resumed command under the frozen practical endpoint; also report all families and ordinary survival.',
    original_nonregression_rejection_preserved=True,
    decision_scope='Explicit mean override, not saved-checkpoint qualification. No automatic adoption, PPO, export or further native run. Original complete qualification remains required.',
    new_PPO=0, actual_experiment_consumed=2654, repair_optimizer_updates=274,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'fallen_legs_original_target_protocol001.json', protocol)
queue = R / 'diagnosis_queue051.py'
assert not queue.exists() and not (R / 'diagnosis_plan051.json').exists()
queue.write_text((R / 'diagnosis_queue050.py').read_text().replace('plan050', 'plan051').replace('queue050', 'queue051'))
frozen = dict(prior['source_sha256'])
extra = [wrapper, R / 'preflight_fallen_legs_batch001.py', R / 'fallen_legs_batch_preflight001.json',
    R / 'fallen_legs_original_target_protocol001.json', R / 'review_quiet_yaw_restore001.py',
    R / 'quiet_yaw_restore_development_decision001.json', R / 'diagnosis050_closed_source_seal001.json',
    queue, Path(__file__)]
frozen.update({str(p): digest(p) for p in extra})
write_json(R / 'diagnosis_plan051.json', dict(schema='serial-native-diagnosis-v2', jobs=[job],
    source_sha256=frozen, protocol=str(R / 'fallen_legs_original_target_protocol001.json'), new_PPO=0,
    actual_experiment_consumed=2654, repair_optimizer_updates=274, qualification=None,
    hardware_started=False, created_unix=time.time()))
p = json.loads((R / 'progress.json').read_text())
p.update(status='fallen_legs_original_target_zero_PPO_diagnosis', active_diagnosis_queue=str(R / 'diagnosis_queue051.json'),
    active_evaluation=str(output), new_PPO_admitted=0, planned_PPO='No PPO admitted; original12s legs-only mean diagnosis.',
    updated_unix=time.time())
write_json(R / 'progress.json', p)
print(json.dumps(dict(plan=str(R / 'diagnosis_plan051.json'), frozen_inputs=len(frozen), new_PPO=0)))

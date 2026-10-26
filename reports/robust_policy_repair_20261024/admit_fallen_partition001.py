"""One serial zero-PPO row-partition test after the failed eight-update pilot."""
import json
import sys
import time
from pathlib import Path

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
prior = json.loads((R / 'plan014_fallen_output001.json').read_text())
decision = json.loads((R / 'fallen_round014_fixed_decision001.json').read_text())
test = json.loads((R / 'fallen_partition_preflight001.json').read_text())
assert decision['decision'] == 'reject_extension_and_adoption'
assert json.loads((R / 'round014_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert all(test['checks'].values())
wrapper = R / 'evaluate_fallen_partition_probe001.py'
probe = R / 'fallen_output_partition_probe001.py'
assert digest(wrapper) == test['wrapper_sha256'] and digest(probe) == test['probe_sha256']
cp = decision['checkpoint']
assert digest(cp) == decision['checkpoint_sha256'] == test['checkpoint_sha256']
assert all(digest(Path(n)) == h for n, h in prior['source_sha256'].items())
protocol = dict(schema='fallen-output-partition-native-diagnosis-v1', checkpoint=cp,
    checkpoint_sha256=digest(cp), source=prior['source_checkpoint'], source_sha256=prior['source_checkpoint_sha256'],
    question='Which part of the rejected learned output changes the native leg-speed peak and interrupts strict continuous followup: four leg output rows or two wheel output rows?',
    interventions=['Revert four leg output rows to1501 and retain learned wheel rows',
        'Revert two wheel output rows to1501 and retain learned leg rows'],
    unchanged='Native51 fixed cases,241201,8s,physics/sensors/materials/guard predicates; only explicit runtime final layer row partition.',
    gate=prior['review_gate'],
    decision_scope='Diagnostic effects and pairing only; runtime function differs from saved checkpoint mean. No checkpoint qualification, adoption or PPO admission.',
    new_PPO=0, actual_experiment_consumed=2638, repair_optimizer_updates=258,
    qualification=None, hardware_started=False, created_unix=time.time())
protocol_path = R / 'fallen_partition_protocol001.json'
assert not protocol_path.exists()
write_json(protocol_path, protocol)
queue = R / 'diagnosis_queue040.py'
assert not queue.exists() and not (R / 'diagnosis_plan040.json').exists()
queue.write_text((R / 'diagnosis_queue039.py').read_text().replace('plan039', 'plan040').replace('queue039', 'queue040'))
jobs = []
for mode in ('wheels', 'legs'):
    out = str(B / ('diagnosis040_fallen014_' + mode + '_only'))
    assert not Path(out).exists()
    jobs.append(dict(name='fallen014_' + mode + '_only', checkpoint=cp, output=out,
        log=str(R / ('diagnosis040_fallen014_' + mode + '_only.log')),
        command=['/home/noir/miniconda3/envs/isaacsim6/bin/python', str(wrapper),
            '--retained-fallen-output-rows', mode, '--checkpoint', cp, '--output', out,
            '--bank', str(R / 'pose_bank_rmcs003.json'), '--actuator-timing', 'native',
            '--seconds', '8', '--seed', '241201']))
frozen = dict(prior['source_sha256'])
extra = [Path(cp), wrapper, probe, R / 'preflight_fallen_partition001.py', R / 'fallen_partition_preflight001.json',
    protocol_path, queue, Path(__file__), R / 'plan014_fallen_output001.json',
    R / 'fallen_round014_endpoint001.json', R / 'fallen_round014_fixed_decision001.json',
    R / 'round014_closed_source_seal001.json', R / 'fallen_round014_followup_audit001.json',
    R / 'audit_fallen_followup001.py', R / 'audit_fallen_round014_endpoint001.py',
    R / 'review_fallen_round014_fixed001.py']
frozen.update({str(p): digest(p) for p in extra})
plan = dict(schema='serial-native-diagnosis-v2', jobs=jobs, source_sha256=frozen,
    protocol=str(protocol_path), new_PPO=0, actual_experiment_consumed=2638,
    repair_optimizer_updates=258, qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'diagnosis_plan040.json', plan)
write_json(R / 'fallen_partition_admission001.json', dict(plan=str(R / 'diagnosis_plan040.json'),
    plan_sha256=digest(R / 'diagnosis_plan040.json'), frozen_inputs=len(frozen),
    new_PPO=0, no_automatic_PPO_or_adoption=True, qualification=None, hardware_started=False))
print(json.dumps(dict(plan=str(R / 'diagnosis_plan040.json'), frozen_inputs=len(frozen),
    jobs=[x['name'] for x in jobs], new_PPO=0), indent=2))

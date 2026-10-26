"""One zero-PPO sensor-scoped wheel counterfactual after the failed row partitions."""
import json
import sys
import time
from pathlib import Path

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
prior = json.loads((R / 'diagnosis_plan040.json').read_text())
decision = json.loads((R / 'fallen_partition_observed_decision001.json').read_text())
assert json.loads((R / 'diagnosis040_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert all(digest(Path(n)) == h for n, h in prior['source_sha256'].items())
assert decision['results']['wheels']['checks']['entry_followup'] is False
assert decision['results']['legs']['checks']['fallen_speed'] is False
test = json.loads((R / 'fallen_wheel_scope_preflight001.json').read_text())
assert all(test['checks'].values())
wrapper = R / 'evaluate_fallen_wheel_scope001.py'
probe = R / 'fallen_wheel_entry_scope_probe001.py'
assert digest(wrapper) == test['wrapper_sha256'] and digest(probe) == test['probe_sha256']
cp = str(B / 'pilot014_fallen_output/model_final.pt')
assert digest(cp) == test['trained_sha256']
gate = json.loads((R / 'plan014_fallen_output001.json').read_text())['review_gate']
protocol = dict(schema='entry-protected-fallen-wheel-native-probe-v1', trained_checkpoint=cp,
    trained_sha256=digest(cp), source_checkpoint=str(B / 'fallen_output_root001/model_initial.pt'),
    source_sha256=test['source_sha256'],
    hypothesis='The wheel-row update improved fallen recovery and speed but harmed entry followup. Suppressing its action difference while the existing sensor-only entry authority is present may preserve that benefit without entry loss.',
    function='Source legs always; source wheels plus clamp(1-entry_gate/.05,0,1) times learned wheel-only output difference. Exact endpoints enforced.',
    unchanged='All native51 cases/241201/8s, physics, sensor domains, friction, torque/impact/closure/contact/time predicates, source/checkpoint files and sampling std.',
    gate=gate, required_fixed_gain='fallen recovery>=5, or entry followup>=9, or fallen followup>=3, in addition to every unchanged nonregression gate.',
    decision_scope='Native function diagnostic only; no saved-mean qualification, model adoption or PPO extension.',
    new_PPO=0, actual_experiment_consumed=2638, repair_optimizer_updates=258,
    qualification=None, hardware_started=False, created_unix=time.time())
protocol_path = R / 'fallen_wheel_scope_protocol001.json'
assert not protocol_path.exists()
write_json(protocol_path, protocol)
queue = R / 'diagnosis_queue041.py'
assert not queue.exists() and not (R / 'diagnosis_plan041.json').exists()
queue.write_text((R / 'diagnosis_queue040.py').read_text().replace('plan040', 'plan041').replace('queue040', 'queue041'))
output = str(B / 'diagnosis041_entry_protected_wheels')
assert not Path(output).exists()
job = dict(name='entry_protected_fallen_wheels', checkpoint=cp, output=output,
    log=str(R / 'diagnosis041_entry_protected_wheels.log'),
    command=['/home/noir/miniconda3/envs/isaacsim6/bin/python', str(wrapper), '--checkpoint', cp,
        '--output', output, '--bank', str(R / 'pose_bank_rmcs003.json'),
        '--actuator-timing', 'native', '--seconds', '8', '--seed', '241201'])
frozen = dict(prior['source_sha256'])
extra = [wrapper, probe, R / 'preflight_fallen_wheel_scope001.py', R / 'fallen_wheel_scope_preflight001.json',
    R / 'fallen_wheel_entry_scope001.json', R / 'audit_fallen_wheel_entry_scope001.py',
    R / 'fallen_partition_observed_decision001.json', R / 'review_fallen_partition001.py',
    R / 'pair_fallen_partition_diagnostic001.py', R / 'fallen_partition_standard_guard_rejection001.json',
    R / 'diagnosis040_closed_source_seal001.json', protocol_path, queue, Path(__file__)]
frozen.update({str(p): digest(p) for p in extra})
plan = dict(schema='serial-native-diagnosis-v2', jobs=[job], source_sha256=frozen,
    protocol=str(protocol_path), new_PPO=0, actual_experiment_consumed=2638, repair_optimizer_updates=258,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'diagnosis_plan041.json', plan)
write_json(R / 'fallen_wheel_scope_admission001.json', dict(plan=str(R / 'diagnosis_plan041.json'),
    plan_sha256=digest(R / 'diagnosis_plan041.json'), frozen_inputs=len(frozen), new_PPO=0,
    no_automatic_PPO_or_adoption=True, qualification=None, hardware_started=False))
requirements = json.loads((R / 'completion_requirements001.json').read_text())
requirements['latest_research_status'].update(repair_total_PPO=258, experiment_consumed=2638,
    latest_branch='014 completed8 updates1501to1509; fixed fallen followup2to0 and speed24.11665to24.48754, rejected.',
    current_diagnosis='041 zero-PPO entry-protected wheel update;040 wheel-only5/6 recovery but entryfollow6/12, leg-only5/6 but nativepeak24.1486>24.11665; neither adopted.')
requirements['updated_unix'] = time.time()
write_json(R / 'completion_requirements001.json', requirements)
print(json.dumps(dict(plan=str(R / 'diagnosis_plan041.json'), frozen_inputs=len(frozen), new_PPO=0), indent=2))

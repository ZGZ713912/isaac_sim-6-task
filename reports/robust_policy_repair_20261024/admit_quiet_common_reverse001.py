"""Test only common response after the native yaw mechanism passes."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
prior = json.loads((R / 'diagnosis_plan051.json').read_text())
assert json.loads((R / 'diagnosis_queue051.json').read_text())['status'] == 'completed'
assert json.loads((R / 'diagnosis051_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert all(digest(p) == s for p, s in prior['source_sha256'].items())
assert json.loads((R / 'quiet_yaw_restore_development_decision001.json').read_text())['mechanism_gate_pass']
preflight = json.loads((R / 'quiet_wheel_common_reverse_preflight001.json').read_text())
assert all(preflight['checks'].values())
wrapper, probe = R / 'evaluate_quiet_wheel_common_reverse001.py', R / 'quiet_wheel_common_reverse_probe001.py'
assert digest(wrapper) == preflight['wrapper_sha256'] and digest(probe) == preflight['probe_sha256']
assert digest(preflight['checkpoint']) == preflight['checkpoint_sha256']
template = json.loads((R / 'diagnosis_plan050.json').read_text())['jobs'][0]
job = dict(template)
output = B / 'diagnosis052_quiet_common_reverse_development'
assert not output.exists()
job.update(name='quiet_common_reverse_development', output=str(output),
    log=str(R / 'diagnosis052_quiet_common_reverse_development.log'))
job['command'] = [str(wrapper) if x == str(R / 'evaluate_quiet_wheel_yaw_restore001.py')
    else str(output) if x == template['output'] else x for x in template['command']]
protocol = dict(schema='quiet-common-sign-after-yaw-protocol-v1',
    control='runs/robust_policy_repair_20261024/diagnosis038_quiet_stance050_development',
    differential_control=template['output'], checkpoint=preflight['checkpoint'],
    checkpoint_sha256=preflight['checkpoint_sha256'], full_population=51, seed=241204, seconds=8,
    hypothesis='050 halves low/mid unintended yaw but leaves4 position failures. Retain its exact differential response and reverse only common translation response to test first-second overshoot/drift. No inference of closed-loop stability from actuator Jacobians.',
    intervention='Wheel submatrix is negative diag(original row sums), hence the050 differential response is unchanged and its common response is reversed. Every leg/other weight,.5 authority,observer and physical guard unchanged.',
    gates=dict(standing_survived=21, standing_pose_position_at_least=19,
        lost_previous_development_passes_at_most=0, max_gap_m=.003, passive_motor_peak_nm=0,
        yaw_cases=['stand_low__roll_plus2', 'stand__roll_plus2'],
        yaw_tail_metric='mean(abs(native_body_omega_z)) over6to8seconds,rad/s',
        yaw_at_most_fraction_of_original=.5),
    scope='Previously used development cases, not a fresh holdout. Saved mean modified only in memory; explicitly excluded from checkpoint qualification.',
    decision_scope='Pass permits separate zero-PPO confirmation only. Fail rejects this intervention. No PPO/adoption/export.',
    new_PPO=0, actual_experiment_consumed=2654, repair_optimizer_updates=274,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'quiet_common_reverse_protocol001.json', protocol)
queue = R / 'diagnosis_queue052.py'
assert not queue.exists() and not (R / 'diagnosis_plan052.json').exists()
queue.write_text((R / 'diagnosis_queue051.py').read_text().replace('plan051', 'plan052').replace('queue051', 'queue052'))
frozen = dict(prior['source_sha256'])
extra = [wrapper, probe, R / 'preflight_quiet_wheel_common_reverse001.py',
    R / 'quiet_wheel_common_reverse_preflight001.json', R / 'quiet_common_reverse_protocol001.json',
    R / 'review_fallen_legs_original_target001.py', R / 'fallen_legs_target_diagnostic_result001.json',
    R / 'diagnosis051_closed_source_seal001.json', queue, Path(__file__)]
frozen.update({str(p): digest(p) for p in extra})
write_json(R / 'diagnosis_plan052.json', dict(schema='serial-native-diagnosis-v2', jobs=[job],
    source_sha256=frozen, protocol=str(R / 'quiet_common_reverse_protocol001.json'), new_PPO=0,
    actual_experiment_consumed=2654, repair_optimizer_updates=274, qualification=None,
    hardware_started=False, created_unix=time.time()))
p = json.loads((R / 'progress.json').read_text())
p.update(status='quiet_common_response_zero_PPO_diagnosis', active_diagnosis_queue=str(R / 'diagnosis_queue052.json'),
    active_evaluation=str(output), new_PPO_admitted=0, planned_PPO='No PPO admitted; common-response single-variable diagnosis.',
    updated_unix=time.time())
write_json(R / 'progress.json', p)
print(json.dumps(dict(plan=str(R / 'diagnosis_plan052.json'), frozen_inputs=len(frozen), new_PPO=0)))

"""A native test of differential sign, with unchanged common wheel response."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
prior = json.loads((R / 'diagnosis_plan049.json').read_text())
assert json.loads((R / 'diagnosis_queue049.json').read_text())['status'] == 'completed'
assert json.loads((R / 'diagnosis049_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert all(digest(p) == s for p, s in prior['source_sha256'].items())
preflight = json.loads((R / 'quiet_wheel_yaw_restore_preflight001.json').read_text())
assert all(preflight['checks'].values())
wrapper = R / 'evaluate_quiet_wheel_yaw_restore001.py'
probe = R / 'quiet_wheel_yaw_restore_probe001.py'
assert digest(wrapper) == preflight['wrapper_sha256'] and digest(probe) == preflight['probe_sha256']
assert digest(preflight['checkpoint']) == preflight['checkpoint_sha256']
template = prior['jobs'][0]
job = dict(template)
output = B / 'diagnosis050_quiet_yaw_restore_development'
assert not output.exists()
job.update(name='quiet_yaw_restore_development', output=str(output),
    log=str(R / 'diagnosis050_quiet_yaw_restore_development.log'))
job['command'] = [str(wrapper) if x == str(R / 'evaluate_quiet_wheel_modal001.py')
    else str(output) if x == template['output'] else x for x in template['command']]
protocol = dict(schema='quiet-wheel-opposite-differential-protocol-v1',
    control='runs/robust_policy_repair_20261024/diagnosis038_quiet_stance050_development',
    same_common_control=template['output'], checkpoint=preflight['checkpoint'],
    checkpoint_sha256=preflight['checkpoint_sha256'], full_population=51, seed=241204, seconds=8,
    hypothesis='Common translation and differential yaw have different balance dynamics. Relative to049 diagonal allocation, reverse only the differential response while exactly retaining both common row sums. Test native yaw reduction rather than inferring stability from the immediate actuator Jacobian.',
    intervention='For every existing height expert set wheel submatrix [[0,S_left],[S_right,0]], where S is the original wheel row sum. All leg and other weights,.5 quiet authority,35D sensors,observer,physical guards unchanged.',
    mechanism_gate=dict(cases=['stand_low__roll_plus2', 'stand__roll_plus2'],
        metric='mean(abs(native_body_omega_z)) over6to8seconds,rad/s', at_most_fraction_of_original=.5,
        standing_survived=21, lost_previous_passes_at_most=0),
    standing_gate=dict(pose_and_position_at_least=19, max_gap_m=.003, passive_motor_peak_nm=0),
    scope='Previously used development cases; not a fresh holdout. Explicit diagnostic mean override excludes saved-checkpoint qualification. A yaw mechanism pass is not a standing qualification or PPO admission.',
    decisions='Fail yaw reduction rejects the proposed mechanism. Passing yaw with failed standing allows only a separately justified zero-PPO common-response investigation; never PPO/adoption/export.',
    new_PPO=0, actual_experiment_consumed=2654, repair_optimizer_updates=274,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'quiet_yaw_restore_protocol001.json', protocol)
queue = R / 'diagnosis_queue050.py'
assert not queue.exists() and not (R / 'diagnosis_plan050.json').exists()
queue.write_text((R / 'diagnosis_queue049.py').read_text().replace('plan049', 'plan050').replace('queue049', 'queue050'))
frozen = dict(prior['source_sha256'])
extra = [wrapper, probe, R / 'preflight_quiet_wheel_yaw_restore001.py',
    R / 'quiet_wheel_yaw_restore_preflight001.json', R / 'quiet_yaw_restore_protocol001.json',
    R / 'quiet_wheel_modal_development_decision001.json', R / 'diagnosis049_closed_source_seal001.json', queue, Path(__file__)]
frozen.update({str(p): digest(p) for p in extra})
write_json(R / 'diagnosis_plan050.json', dict(schema='serial-native-diagnosis-v2', jobs=[job],
    source_sha256=frozen, protocol=str(R / 'quiet_yaw_restore_protocol001.json'), new_PPO=0,
    actual_experiment_consumed=2654, repair_optimizer_updates=274, qualification=None,
    hardware_started=False, created_unix=time.time()))
p = json.loads((R / 'progress.json').read_text())
p.update(status='quiet_yaw_sign_zero_PPO_diagnosis', active_diagnosis_queue=str(R / 'diagnosis_queue050.json'),
    active_evaluation=str(output), new_PPO_admitted=0, planned_PPO='No PPO admitted; differential-sign mechanism diagnosis only.',
    updated_unix=time.time())
write_json(R / 'progress.json', p)
print(json.dumps(dict(plan=str(R / 'diagnosis_plan050.json'), frozen_inputs=len(frozen), new_PPO=0)))

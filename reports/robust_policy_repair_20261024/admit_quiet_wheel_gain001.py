"""One-variable quiet wheel feedback diagnosis, with the full paired population."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
previous = json.loads((R / 'diagnosis_plan047.json').read_text())
assert json.loads((R / 'diagnosis_queue047.json').read_text())['status'] == 'completed'
assert json.loads((R / 'diagnosis047_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert all(digest(p) == sha for p, sha in previous['source_sha256'].items())
preflight = json.loads((R / 'quiet_wheel_gain_preflight001.json').read_text())
assert all(preflight['checks'].values())
wrapper = R / 'evaluate_quiet_wheel_gain001.py'
probe = R / 'quiet_wheel_gain_probe001.py'
assert digest(wrapper) == preflight['wrapper_sha256']
assert digest(probe) == preflight['probe_sha256']
assert digest(preflight['checkpoint']) == preflight['checkpoint_sha256']
template = json.loads((R / 'diagnosis_plan038.json').read_text())['jobs'][0]
job = dict(template)
output = B / 'diagnosis048_quiet_wheel_gain3_development'
assert not output.exists()
job.update(name='quiet_wheel_gain3_development', output=str(output),
    log=str(R / 'diagnosis048_quiet_wheel_gain3_development.log'))
job['command'] = [str(wrapper) if x == 'scripts/evaluate_gap_robustness.py'
    else str(output) if x == template['output'] else x for x in template['command']]
protocol = dict(schema='quiet-wheel-gain-single-variable-protocol-v1',
    checkpoint=preflight['checkpoint'], checkpoint_sha256=preflight['checkpoint_sha256'],
    control=template['output'], full_population=51, seed=241204, seconds=8,
    scope='Previously used development cases, not a fresh holdout;0PPO. Explicit in-memory saved-mean diagnostic, not an authenticated checkpoint or candidate adoption.',
    hypothesis='Multiplying only the existing quiet wheel displacement feedback by3 may reduce persistent drift under the same sampled sensor/actuator biases. No claim of unique drift cause.',
    intervention='12 existing weight entries in3 existing height experts, wheel rows only. Existing.5 quiet authority, leg outputs, outside-quiet outputs and original checkpoint bytes unchanged.',
    gates=dict(standing_survived=21, standing_pose_position_at_least=19,
        lost_previous_development_passes_at_most=0, max_gap_m=.003, passive_motor_peak_nm=0),
    physical_pairing='Require same full51 population, precontrol native states and sampled material/rigid/sensor/actuator parameters. Do not use a21-case subset that would change RNG draws.',
    decision_scope='Pass permits further zero-PPO assessment only; fail rejects this intervention. No PPO/adoption/export/qualification.',
    new_PPO=0, actual_experiment_consumed=2646, repair_optimizer_updates=266,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'quiet_wheel_gain_protocol001.json', protocol)
queue = R / 'diagnosis_queue048.py'
assert not queue.exists() and not (R / 'diagnosis_plan048.json').exists()
queue.write_text((R / 'diagnosis_queue047.py').read_text().replace('plan047', 'plan048').replace('queue047', 'queue048'))
frozen = dict(previous['source_sha256'])
extras = [wrapper, probe, R / 'preflight_quiet_wheel_gain001.py', R / 'quiet_wheel_gain_preflight001.json',
    R / 'quiet_wheel_gain_protocol001.json', R / 'diagnosis047_closed_source_seal001.json',
    R / 'motion015_target_diagnostic_result001.json', R / 'review_motion015_original_target001.py',
    R / 'pair_declared_mean_probe001.py', queue, Path(__file__)]
frozen.update({str(p): digest(p) for p in extras})
write_json(R / 'diagnosis_plan048.json', dict(schema='serial-native-diagnosis-v2', jobs=[job],
    source_sha256=frozen, protocol=str(R / 'quiet_wheel_gain_protocol001.json'), new_PPO=0,
    actual_experiment_consumed=2646, repair_optimizer_updates=266,
    qualification=None, hardware_started=False, created_unix=time.time()))
p = json.loads((R / 'progress.json').read_text())
p.update(status='quiet_wheel_gain_zero_PPO_diagnosis', active_diagnosis_queue=str(R / 'diagnosis_queue048.json'),
    active_evaluation=str(output), active_training=None, new_PPO_admitted=0,
    planned_PPO='No new PPO admitted; gain3 development diagnosis only.', updated_unix=time.time())
write_json(R / 'progress.json', p)
print(json.dumps(dict(plan=str(R / 'diagnosis_plan048.json'), frozen_inputs=len(frozen), new_PPO=0)))

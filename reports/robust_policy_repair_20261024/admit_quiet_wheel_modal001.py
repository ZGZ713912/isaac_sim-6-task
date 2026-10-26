"""Independent zero-PPO wheel-mode diagnosis after the reward trial is sealed."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
previous = json.loads((R / 'plan016_followup_reward001.json').read_text())
assert json.loads((Path(previous['supervisor_output']) / 'progress.json').read_text())['status'] == 'review_ready'
assert json.loads((R / 'round016_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert all(digest(p) == sha for p, sha in previous['source_sha256'].items())
assert json.loads((R / 'quiet_wheel_gain_development_decision001.json').read_text())['decision'] == 'reject_intervention'
preflight = json.loads((R / 'quiet_wheel_modal_preflight001.json').read_text())
assert all(preflight['checks'].values())
wrapper, probe = R / 'evaluate_quiet_wheel_modal001.py', R / 'quiet_wheel_modal_probe001.py'
assert digest(wrapper) == preflight['wrapper_sha256'] and digest(probe) == preflight['probe_sha256']
assert digest(preflight['checkpoint']) == preflight['checkpoint_sha256']
template = json.loads((R / 'diagnosis_plan038.json').read_text())['jobs'][0]
job = dict(template)
output = B / 'diagnosis049_quiet_wheel_modal_development'
assert not output.exists()
job.update(name='quiet_wheel_modal_development', output=str(output),
    log=str(R / 'diagnosis049_quiet_wheel_modal_development.log'))
job['command'] = [str(wrapper) if x == 'scripts/evaluate_gap_robustness.py'
    else str(output) if x == template['output'] else x for x in template['command']]
protocol = dict(schema='quiet-wheel-common-preserved-diagonal-protocol-v1',
    checkpoint=preflight['checkpoint'], checkpoint_sha256=preflight['checkpoint_sha256'],
    control=template['output'], full_population=51, seed=241204, seconds=8,
    scope='Independent standing diagnosis, not an extension/adoption of016. Previously used development cases, not a fresh holdout;0PPO. Explicit in-memory mean intervention cannot qualify the saved checkpoint.',
    hypothesis='Recorded unwanted yaw accompanies differential wheel coordinate accumulation; right wheel differential feedback nearly cancels. Test own-wheel diagonal displacement feedback while preserving each wheel row sum, to distinguish differential response from the rejected common gain increase.',
    intervention='Replace each2x2 wheel submatrix by diag(its two row sums). All three existing height experts; all leg/other weights,.5 quiet authority,35D observations and physical guards retained. No observer modification.',
    evidence=str(R / 'quiet_yaw_modes_audit001.json'),
    gates=dict(standing_survived=21, standing_pose_position_at_least=19,
        lost_previous_development_passes_at_most=0, max_gap_m=.003, passive_motor_peak_nm=0),
    physical_pairing='Same full51 population, initial native states and rigid/material/sensor/actuator random draws.',
    decision_scope='Pass permits further zero-PPO assessment only. Fail rejects this intervention; no PPO/adoption/export/qualification.',
    new_PPO=0, actual_experiment_consumed=2654, repair_optimizer_updates=274,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'quiet_wheel_modal_protocol001.json', protocol)
queue = R / 'diagnosis_queue049.py'
assert not queue.exists() and not (R / 'diagnosis_plan049.json').exists()
queue.write_text((R / 'diagnosis_queue048.py').read_text().replace('plan048', 'plan049').replace('queue048', 'queue049'))
frozen = dict(previous['source_sha256'])
extras = [wrapper, probe, R / 'preflight_quiet_wheel_modal001.py', R / 'quiet_wheel_modal_preflight001.json',
    R / 'quiet_wheel_modal_protocol001.json', R / 'quiet_yaw_modes_audit001.json',
    R / 'audit_quiet_yaw_modes001.py', R / 'round016_closed_source_seal001.json',
    R / 'followup_round016_endpoint001.json', R / 'followup_round016_fixed_decision001.json',
    R / 'review_quiet_wheel_modal001.py', queue, Path(__file__)]
frozen.update({str(p): digest(p) for p in extras})
write_json(R / 'diagnosis_plan049.json', dict(schema='serial-native-diagnosis-v2', jobs=[job],
    source_sha256=frozen, protocol=str(R / 'quiet_wheel_modal_protocol001.json'), new_PPO=0,
    actual_experiment_consumed=2654, repair_optimizer_updates=274,
    qualification=None, hardware_started=False, created_unix=time.time()))
p = json.loads((R / 'progress.json').read_text())
assert p['actual_experiment_consumed'] == 2654 and p['repair_optimizer_updates'] == 274
p.update(status='quiet_wheel_modal_zero_PPO_diagnosis', active_diagnosis_queue=str(R / 'diagnosis_queue049.json'),
    active_evaluation=str(output), active_training=None, active_supervisor=None,
    new_PPO_admitted=0, planned_PPO='No new PPO admitted; common-preserved wheel mode diagnosis only.', updated_unix=time.time())
write_json(R / 'progress.json', p)
print(json.dumps(dict(plan=str(R / 'diagnosis_plan049.json'), frozen_inputs=len(frozen), new_PPO=0)))

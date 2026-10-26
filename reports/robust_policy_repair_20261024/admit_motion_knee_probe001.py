"""Admit one fixed zero-PPO test of knee-only damping from the unchanged best."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
prior = json.loads((R/'diagnosis_plan041.json').read_text())
assert json.loads((R/'diagnosis041_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert all(digest(Path(n))==h for n,h in prior['source_sha256'].items())
assert json.loads((R/'fallen_wheel_scope_decision001.json').read_text())['decision']=='reject_adoption_and_PPO_extension'
test = json.loads((R/'motion_knee_preflight001.json').read_text())
assert all(test['checks'].values())
wrapper, probe = R/'evaluate_motion_knee_probe001.py', R/'motion_knee_damping_probe001.py'
assert digest(wrapper)==test['wrapper_sha256'] and digest(probe)==test['probe_sha256']
cp = B/'motion_damping080002/model_probe.pt'
assert digest(cp)==test['source_sha256']
gate = json.loads((R/'plan014_fallen_output001.json').read_text())['review_gate']
protocol = dict(schema='motion-knee-only-native-probe-protocol-v1', checkpoint=str(cp),
    checkpoint_sha256=digest(cp), knee_increment_alpha=.6,
    hypothesis='Prior all-leg alpha.6 lost fallen followup. Separating knee opening from common rotation may reduce support disturbances without adding balance delay. Recorded spectra show both modes; neither is established as the cause.',
    intervention='Additional scalar antisymmetric knee target correction only on existing sensor motion gate. Common rotation/wheels/inactive inputs exact; correction constrained by original .12rad error, action bound and paired knee geometry.',
    gate=gate, required_gain='fallen recovery>=5 OR entry followup>=9 OR fallen followup>=3, together with every nonregression gate.',
    unchanged='Original51 cases,241201,8s, all physical/DR/observer/contact/duration/torque/impact/closure predicates; checkpoint/source neural weights and std.',
    research_only=True, eligible_for_checkpoint_mean_qualification=False,
    new_PPO=0, repair_optimizer_updates=258, actual_experiment_consumed=2638,
    qualification=None, hardware_started=False, created_unix=time.time())
protocol_path = R/'motion_knee_protocol001.json'
assert not protocol_path.exists()
write_json(protocol_path, protocol)
queue = R/'diagnosis_queue042.py'
assert not queue.exists() and not (R/'diagnosis_plan042.json').exists()
queue.write_text((R/'diagnosis_queue041.py').read_text().replace('plan041','plan042').replace('queue041','queue042'))
output = B/'diagnosis042_motion_knee060'
assert not output.exists()
job = dict(name='motion_knee_only060', checkpoint=str(cp), output=str(output),
    log=str(R/'diagnosis042_motion_knee060.log'),
    command=['/home/noir/miniconda3/envs/isaacsim6/bin/python', str(wrapper),
        '--checkpoint', str(cp), '--output', str(output), '--bank', str(R/'pose_bank_rmcs003.json'),
        '--actuator-timing', 'native', '--seconds', '8', '--seed', '241201'])
frozen = dict(prior['source_sha256'])
extras = [wrapper, probe, R/'preflight_motion_knee_probe001.py', R/'motion_knee_preflight001.json',
    R/'motion_knee_preflight_negative001.json', R/'preflight_motion_knee_probe_negative001.py',
    R/'post_recovery_modes001.json', R/'audit_post_recovery_modes001.py',
    R/'diagnosis041_closed_source_seal001.json', R/'fallen_wheel_scope_decision001.json',
    R/'pair_declared_mean_probe001.py', R/'review_fallen_wheel_scope001.py', protocol_path, queue, Path(__file__)]
frozen.update({str(p):digest(p) for p in extras})
plan = dict(schema='serial-native-diagnosis-v2', jobs=[job], source_sha256=frozen,
    protocol=str(protocol_path), new_PPO=0, actual_experiment_consumed=2638, repair_optimizer_updates=258,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R/'diagnosis_plan042.json', plan)
write_json(R/'motion_knee_admission001.json', dict(plan=str(R/'diagnosis_plan042.json'),
    plan_sha256=digest(R/'diagnosis_plan042.json'), frozen_inputs=len(frozen),
    new_PPO=0, no_automatic_PPO_or_adoption=True, qualification=None))
p = json.loads((R/'progress.json').read_text())
p.update(status='motion_knee_zero_PPO_probe_admitted', active_diagnosis_queue=str(R/'diagnosis_queue042.json'),
    active_evaluation=str(output), new_PPO_admitted=0, updated_unix=time.time())
write_json(R/'progress.json', p)
print(json.dumps(dict(plan=str(R/'diagnosis_plan042.json'), frozen_inputs=len(frozen), new_PPO=0)))

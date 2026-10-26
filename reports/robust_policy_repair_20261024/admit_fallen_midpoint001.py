"""Evaluate the already consumed fourth update; never spend or refund PPO."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
import torch
from train_robust_getup import digest, write_json
from export_robust_getup import load_mean

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
prior = json.loads((R/'diagnosis_plan042.json').read_text())
assert json.loads((R/'diagnosis042_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert all(digest(Path(n))==h for n,h in prior['source_sha256'].items())
assert json.loads((R/'motion_knee_decision001.json').read_text())['decision']=='reject_adoption_and_PPO_extension'
cp = B/'pilot014_fallen_output/model_1504.pt'
assert digest(cp)=='aebf1b888147cccad06cccc1b730b00beac486a2eceebf3a9c0c473a6ed2c8f0'
mean, info = load_mean(cp)
learning = info['policy_function_learning']
assert info['completed_updates']==1505 and learning['new_optimizer_updates']==4 and learning['source_learning_clock']==1501
assert learning['native_Adam_steps_per_update']==12
state = torch.load(cp, map_location='cpu', weights_only=False)
source = torch.load(learning['source_checkpoint'], map_location='cpu', weights_only=False)
names = info['policy_function_root']['optimizer_actor_parameter_names']
ids = state['optimizer_state_dict']['param_groups'][0]['params']
steps = [float(state['optimizer_state_dict']['state'][ids[names.index(k)]]['step']-source['optimizer_state_dict']['state'][ids[names.index(k)]]['step'])
    for k in ('mlp.mean.recovery.4.weight', 'mlp.mean.recovery.4.bias')]
assert steps==[48.,48.]
gate = json.loads((R/'plan014_fallen_output001.json').read_text())['review_gate']
protocol = dict(schema='fallen014-native-midpoint-assessment-v1', checkpoint=str(cp), checkpoint_sha256=digest(cp),
    source_clock=1501, midpoint_clock=1505, midpoint_updates_already_counted=4, native_Adam_step_increment=steps,
    hypothesis='The failed8-update endpoint may have passed through a better intermediate mean. Evaluate the sole earlier sealed4-update checkpoint before excluding that learning interval. This fixed development selection is not heldout qualification.',
    gate=gate, required_gain='fallen recovery>=5 OR entry followup>=9 OR fallen followup>=3, plus every unchanged nonregression gate.',
    native_mean_unmodified=True, new_PPO=0, repair_optimizer_updates=258, actual_experiment_consumed=2638,
    qualification=None, hardware_started=False, created_unix=time.time())
protocol_path = R/'fallen_midpoint_protocol001.json'
assert not protocol_path.exists()
write_json(protocol_path, protocol)
queue = R/'diagnosis_queue043.py'
assert not queue.exists() and not (R/'diagnosis_plan043.json').exists()
queue.write_text((R/'diagnosis_queue042.py').read_text().replace('plan042','plan043').replace('queue042','queue043'))
output = B/'diagnosis043_fallen014_midpoint4'
assert not output.exists()
job = dict(name='fallen014_midpoint4', checkpoint=str(cp), output=str(output),
    log=str(R/'diagnosis043_fallen014_midpoint4.log'),
    command=['/home/noir/miniconda3/envs/isaacsim6/bin/python', 'scripts/evaluate_gap_robustness.py',
        '--checkpoint', str(cp), '--output', str(output), '--bank', str(R/'pose_bank_rmcs003.json'),
        '--actuator-timing', 'native', '--seconds', '8', '--seed', '241201'])
frozen = dict(prior['source_sha256'])
extras = [cp, R/'motion_knee_decision001.json', R/'diagnosis042_closed_source_seal001.json',
    R/'review_motion_knee_probe001.py', R/'audit_motion_output_scope001.py', R/'motion_output_scope001.json',
    protocol_path, queue, Path(__file__)]
frozen.update({str(p):digest(p) for p in extras})
plan = dict(schema='serial-native-diagnosis-v2', jobs=[job], source_sha256=frozen,
    protocol=str(protocol_path), new_PPO=0, actual_experiment_consumed=2638, repair_optimizer_updates=258,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R/'diagnosis_plan043.json', plan)
p = json.loads((R/'progress.json').read_text())
p.update(status='fallen014_midpoint_assessment', active_diagnosis_queue=str(R/'diagnosis_queue043.json'),
    active_evaluation=str(output), new_PPO_admitted=0, updated_unix=time.time())
write_json(R/'progress.json', p)
print(json.dumps(dict(plan=str(R/'diagnosis_plan043.json'), frozen_inputs=len(frozen), new_PPO=0, clock=1505)))

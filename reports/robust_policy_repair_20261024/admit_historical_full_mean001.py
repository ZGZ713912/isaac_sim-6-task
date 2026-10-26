"""One zero-PPO historical-function diagnosis after017 is fully closed."""
from pathlib import Path
import json
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
prior = json.loads((R / 'plan017_quiet_output001.json').read_text())
assert json.loads((R / 'round017_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert json.loads((Path(prior['supervisor_output']) / 'progress.json').read_text())['status'] == 'review_ready'
assert all(digest(p) == h for p, h in prior['source_sha256'].items())
for file in ('historical_mean_compatibility_preflight001.json', 'historical_full_wrapper_preflight001.json'):
    assert all(json.loads((R / file).read_text())['checks'].values()), file
preflight = json.loads((R / 'historical_full_wrapper_preflight001.json').read_text())
wrapper = R / 'evaluate_historical_full_mean001.py'
assert digest(wrapper) == preflight['wrapper_sha256']
assert digest('scripts/evaluate_robust_getup_batch.py') == preflight['batch_sha256']
source = Path('runs/unified_robust_20261022/offline_recovery096/model_migrated.pt')
proof = json.loads((R / 'historical_mean_compatibility_preflight001.json').read_text())
assert digest(source) == proof['checkpoint_sha256']
old_process = Path('reports/unified_robust_20261021/early_roll_push107_process.json')
command = list(json.loads(old_process.read_text())['command'])
command[1] = str(wrapper)
checkpoint = B / 'motion_damping080002/model_probe.pt'
output = B / 'diagnosis055_historical_full_mean_moving'
assert not output.exists()
for key, value in (('--checkpoint', str(checkpoint)), ('--output', str(output))):
    command[command.index(key) + 1] = value
if '--checkpoint-task-profile' not in command: command.append('--checkpoint-task-profile')
assert command[command.index('--replicas') + 1] == '8'
assert command[command.index('--seed') + 1] == '211322'
protocol = dict(schema='historical-full-mean-native-protocol-v1',
    hypothesis='Six historical recovery tensors in the current decoder did not reproduce moving recovery. Test the whole archived096 forward function to distinguish learned tensor transfer from broader runtime-function changes. CPU verification executes archived forward and coordinate code and confirms32768 recorded inputs exactly.',
    context='Current9a environment/35D observer, original native physical/impact/closure guards. Historical48population shape, original moving-left/right cases,8replicas,seed211322,12s and the original20to40/50 diagnostic gates. This is not full historical episode reproduction because current observer/environment source is retained.',
    intervention='Whole historical096 mean via the existing explicit archived-source migration interface. The current extra actor target limiter is absent from this historical diagnostic and that difference is disclosed. No learned weight or checkpoint file is changed.',
    target='Actual moving fall, recovery and resumed command under the frozen12s practical predicate. Report original strict latch, all impacts and leg speed; compare common pre-push physical traces to the recorded107 window without assuming equality.',
    limits='No current saved-mean qualification, candidate adoption, PPO or export. Positive function evidence would require a paired current-function control and explicit final-action-space investigation. No guard is relaxed.',
    new_PPO=0, actual_experiment_consumed=2662, repair_optimizer_updates=282,
    qualification=None, hardware_started=False, created_unix=time.time())
protocol_path = R / 'historical_full_mean_native_protocol001.json'
assert not protocol_path.exists()
write_json(protocol_path, protocol)
queue = R / 'diagnosis_queue055.py'
assert not queue.exists() and not (R / 'diagnosis_plan055.json').exists()
queue.write_text((R / 'diagnosis_queue054.py').read_text().replace('plan054', 'plan055').replace('queue054', 'queue055'))
job = dict(name='historical_full_mean_moving', checkpoint=str(checkpoint), output=str(output),
    historical_checkpoint=str(source), log=str(R / 'diagnosis055_historical_full_mean_moving.log'), command=command)
frozen = dict(prior['source_sha256'])
extra = [wrapper, R / 'preflight_historical_mean_compatibility001.py', R / 'historical_mean_compatibility_preflight001.json',
    R / 'preflight_historical_full_wrapper001.py', R / 'historical_full_wrapper_preflight001.json',
    R / 'audit_quiet_round017_endpoint001.py', R / 'quiet_round017_endpoint001.json',
    R / 'review_quiet_round017_fixed001.py', R / 'quiet_round017_fixed_decision001.json',
    R / 'round017_closed_source_seal001.json', old_process,
    source.parent / 'source/src/wheeled_algo/robust_getup.py',
    source.parent / 'source/src/wheeled_algo/recovery_coordinates.py',
    Path('runs/unified_robust_20261022/early_roll_push107/robust/report.json'),
    Path('runs/unified_robust_20261022/early_roll_push107/robust/trajectory.npz'),
    protocol_path, queue, Path(__file__)]
frozen.update({str(p): digest(p) for p in extra})
write_json(R / 'diagnosis_plan055.json', dict(schema='serial-native-diagnosis-v2', jobs=[job],
    source_sha256=frozen, protocol=str(protocol_path), new_PPO=0,
    actual_experiment_consumed=2662, repair_optimizer_updates=282, qualification=None,
    hardware_started=False, created_unix=time.time()))
p = json.loads((R / 'progress.json').read_text())
p.update(status='historical_full_function_zero_PPO_diagnosis', active_training=None, active_supervisor=None,
    active_evaluation=str(output), active_diagnosis_queue=str(R / 'diagnosis_queue055.json'),
    new_PPO_admitted=0, planned_PPO='No PPO admitted; archived full mean/current observer diagnosis only.',
    updated_unix=time.time())
write_json(R / 'progress.json', p)
print(json.dumps(dict(plan=str(R / 'diagnosis_plan055.json'), frozen_inputs=len(frozen), command=command)))

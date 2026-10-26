"""Admit one native GAE collection, with every optimizer update forbidden."""
from pathlib import Path
import ast
import json
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
prior = json.loads((R / 'diagnosis_plan053.json').read_text())
assert json.loads((R / 'diagnosis_queue053.json').read_text())['status'] == 'completed'
assert json.loads((R / 'diagnosis053_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
changes = {p: dict(before=h, after=digest(p)) for p, h in prior['source_sha256'].items() if digest(p) != h}
assert set(changes) == {'scripts/train_robust_getup.py', 'scripts/export_robust_getup.py'}, changes
pre = json.loads((R / 'quiet_output_preflight001.json').read_text())
assert all(pre['checks'].values())
for key, path in dict(training_sha256='scripts/train_robust_getup.py',
    export_sha256='scripts/export_robust_getup.py', contract_sha256='src/wheeled_algo/quiet_output_checkpoint.py',
    checkpoint_sha256=pre['checkpoint'], source_sha256=pre['source']).items():
    assert digest(path) == pre[key]
collector = R / 'collect_quiet_output_rollout001.py'
tree = ast.parse(collector.read_text())
forbid = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'forbid_update')
ns = {}; exec(compile(ast.Module(body=[forbid], type_ignores=[]), 'forbidden-native-update', 'exec'), ns)
try: ns['forbid_update'](None)
except RuntimeError: pass
else: raise AssertionError('Zero collector permits an update')
protocol = dict(schema='quiet-output-native-zero-protocol-v1',
    hypothesis='Three existing quiet neural output layers can use gyro, gravity, leg pose and velocities to coordinate state-dependent corrections. Prior010/011 tuned only24 displacement-feedback parameters; repeated wheel-matrix diagnostics did not fix the17/21 development standing failures. This changes training scope only, not the initial actor or runtime action space.',
    source=pre['checkpoint'], source_sha256=pre['checkpoint_sha256'],
    source_function=pre['source'], source_function_sha256=pre['source_sha256'], source_clock=1501,
    trainable=pre['trainable'], actor_parameters=1170,
    retained='All existing hidden layers,24feedback parameters,parent,motion,fallen,entry,std,Adam namespace,observations,gains,gates,torque/impulse/closure protections. Initial saved actor/critic/Adam/RNG are exact. Only declared training exposure changes; reward constants are inherited.',
    required='Actual131072 native sampled transitions and native GAE; strict original mean/conditioned-std equality on every frame; exact full-state restoration; no attempted optimizer update.',
    boundaries='No PPO admission, adoption, export or qualification from this collection. A separately reviewed bounded pilot and complete original qualification remain required.',
    reviewed_source_changes=changes, new_PPO=0, actual_experiment_consumed=2654,
    repair_optimizer_updates=274, qualification=None, hardware_started=False, created_unix=time.time())
protocol_path = R / 'quiet_output_native_zero_protocol001.json'
assert not protocol_path.exists()
write_json(protocol_path, protocol)
queue = R / 'diagnosis_queue054.py'
assert not queue.exists() and not (R / 'diagnosis_plan054.json').exists()
queue.write_text((R / 'diagnosis_queue053.py').read_text().replace('plan053', 'plan054').replace('queue053', 'queue054'))
output = B / 'diagnosis054_quiet_output_native_zero'
assert not output.exists()
args = list(pre['command']); args[args.index('--output') + 1] = str(output)
job = dict(name='quiet_output_native_zero', checkpoint=pre['checkpoint'], output=str(output),
    log=str(R / 'diagnosis054_quiet_output_native_zero.log'), completion_status='stopped',
    command=['/home/noir/miniconda3/envs/isaacsim6/bin/python', str(collector), '--diagnostic-rollout-only', *args])
frozen = {p: digest(p) for p in prior['source_sha256']}
extra = [collector, Path('src/wheeled_algo/quiet_output_checkpoint.py'),
    Path('scripts/migrate_quiet_output_function_root.py'), Path(pre['checkpoint']), Path(pre['checkpoint']).with_suffix('.json'),
    R / 'preflight_quiet_output001.py', R / 'quiet_output_preflight001.json',
    R / 'review_historical_recovery_donor001.py', R / 'historical_recovery_donor_result001.json',
    R / 'diagnosis053_closed_source_seal001.json', protocol_path, queue, Path(__file__)]
frozen.update({str(p): digest(p) for p in extra})
write_json(R / 'diagnosis_plan054.json', dict(schema='serial-native-diagnosis-v2', jobs=[job],
    source_sha256=frozen, reviewed_source_changes=changes, protocol=str(protocol_path), new_PPO=0,
    actual_experiment_consumed=2654, repair_optimizer_updates=274, qualification=None,
    hardware_started=False, created_unix=time.time()))
progress = json.loads((R / 'progress.json').read_text())
progress.update(status='quiet_output_native_zero_diagnosis', active_evaluation=str(output),
    active_diagnosis_queue=str(R / 'diagnosis_queue054.json'),
    latest_completed_evaluation=str(B / 'diagnosis053_historical_recovery_donor'),
    latest_research_decision='053 donor rejected:moving0,side recovery reduced; no unique forgetting cause established.',
    new_PPO_admitted=0, planned_PPO='No PPO admitted; native quiet-output scope/function preflight.', updated_unix=time.time())
write_json(R / 'progress.json', progress)
print(json.dumps(dict(plan=str(R / 'diagnosis_plan054.json'), frozen_inputs=len(frozen), new_PPO=0)))

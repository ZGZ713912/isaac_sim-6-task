"""One serial zero-optimizer native collection before any motion-output PPO."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
prior = json.loads((R/'diagnosis_plan043.json').read_text())
assert json.loads((R/'diagnosis043_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
changed = [n for n,h in prior['source_sha256'].items() if digest(Path(n))!=h]
assert set(changed)=={'scripts/train_robust_getup.py', 'scripts/export_robust_getup.py'}, changed
for name in ('motion_output_contract_CPU_preflight002.json', 'motion_decoder_preflight001.json',
             'motion_prepare_actual_preflight001.json', 'motion_collector_preflight001.json',
             'motion_pilot_wrapper_preflight002.json'):
    assert all(json.loads((R/name).read_text())['checks'].values()), name
test = json.loads((R/'motion_output_regression001.json').read_text())
assert test['passed']==101 and test['exit_code']==0
assert all(digest(Path(n))==h for n,h in test['files_sha256'].items())
prepare = json.loads((R/'motion_prepare_actual_preflight001.json').read_text())
assert digest(Path('scripts/train_robust_getup.py'))==prepare['draft_sha256']
collector = R/'collect_motion_output_rollout001.py'
assert digest(collector)==json.loads((R/'motion_collector_preflight001.json').read_text())['collector_sha256']
wrapper = R/'run_motion_output_pilot001.py'
assert digest(wrapper)==json.loads((R/'motion_pilot_wrapper_preflight002.json').read_text())['wrapper_sha256']
cp = B/'motion_output_root001/model_initial.pt'
assert digest(cp)==prepare['source_sha256']
protocol = dict(schema='motion-output-native-zero-preflight-v1', checkpoint=str(cp), checkpoint_sha256=digest(cp),
    source_clock=1501, intended_trainable_actor_tensors=['mlp.mean.arc.4.weight', 'mlp.mean.arc.4.bias'],
    intended_trainable_actor_parameters=390,
    hypothesis='The existing motion output layer directly controls recorded post-recovery failures while the fallen output layer has zero sensitivity there. Change only the native learning scope; retain runtime function and source state.',
    required='Original mean/conditioned std bit-exact on all131072 native rollout frames; actor/critic/Adam/RNG restored; zero optimizer attempts. Compare every native sampled rollout/GAE tensor to the preceding039 reference before admitting new PPO.',
    callback_order='Native compute_returns before the outer attempted-PPO counter; update is forbidden in this collector.',
    historical_reference=str(B/'diagnosis039_fallen_output_native_zero/rollout_diagnosis.json'),
    new_PPO=0, actual_experiment_consumed=2638, repair_optimizer_updates=258,
    qualification=None, hardware_started=False, created_unix=time.time())
protocol_path = R/'motion_native_zero_protocol001.json'
assert not protocol_path.exists()
write_json(protocol_path, protocol)
queue = R/'diagnosis_queue044.py'
assert not queue.exists() and not (R/'diagnosis_plan044.json').exists()
queue.write_text((R/'diagnosis_queue043.py').read_text().replace('plan043','plan044').replace('queue043','queue044'))
output = B/'diagnosis044_motion_output_native_zero'
assert not output.exists()
args = list(prepare['command'])
args[args.index('--output')+1] = str(output)
job = dict(name='motion_output_native_zero', checkpoint=str(cp), output=str(output),
    log=str(R/'diagnosis044_motion_output_native_zero.log'), completion_status='stopped',
    command=['/home/noir/miniconda3/envs/isaacsim6/bin/python', str(collector), '--diagnostic-rollout-only']+args)
frozen = {n:digest(Path(n)) for n in prior['source_sha256']}
extras = [cp, cp.with_suffix('.json'), Path('src/wheeled_algo/motion_output_checkpoint.py'),
    Path('scripts/migrate_motion_output_function_root.py'), wrapper, collector,
    R/'motion_output_contract_draft001.py', R/'motion_output_contract_CPU_preflight002.json',
    R/'preflight_motion_output_contract002.py', R/'motion_output_regression001.json',
    R/'preflight_motion_decoder001.py', R/'motion_decoder_preflight001.json',
    R/'preflight_motion_prepare_actual001.py', R/'motion_prepare_actual_preflight001.json',
    R/'preflight_motion_collector001.py', R/'motion_collector_preflight001.json',
    R/'preflight_motion_pilot_wrapper002.py', R/'motion_pilot_wrapper_preflight002.json',
    R/'fallen_midpoint_decision001.json', R/'review_fallen_midpoint001.py',
    R/'diagnosis043_closed_source_seal001.json', protocol_path, queue, Path(__file__)]
frozen.update({str(p):digest(p) for p in extras})
plan = dict(schema='serial-native-diagnosis-v2', jobs=[job], source_sha256=frozen,
    protocol=str(protocol_path), new_PPO=0, actual_experiment_consumed=2638, repair_optimizer_updates=258,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R/'diagnosis_plan044.json', plan)
p = json.loads((R/'progress.json').read_text())
p.update(status='motion_output_native_zero_admitted', active_diagnosis_queue=str(R/'diagnosis_queue044.json'),
    active_evaluation=str(output), active_training=None, new_PPO_admitted=0, updated_unix=time.time())
write_json(R/'progress.json', p)
print(json.dumps(dict(plan=str(R/'diagnosis_plan044.json'), frozen_inputs=len(frozen), new_PPO=0)))

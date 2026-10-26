"""Admit native exposure/credit diagnosis only, with no PPO authorization."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
prior = json.loads((R / 'plan015_motion_output001.json').read_text())
assert json.loads((R / 'round015_closed_source_seal001.json').read_text())['all_inputs_exact']
assert json.loads((R / 'motion_round015_fixed_decision001.json').read_text())['decision'] == 'reject_extension_and_adoption'
changed = [p for p, sha in prior['source_sha256'].items() if digest(p) != sha]
assert set(changed) == {'scripts/train_robust_getup.py', 'scripts/export_robust_getup.py'}, changed
for name in ('followup_reward_preflight001.json', 'followup_collector_preflight001.json', 'followup_reward_replay001.json'):
    assert all(json.loads((R / name).read_text())['checks'].values()), name
cpu = json.loads((R / 'followup_reward_preflight001.json').read_text())
assert all(digest(p) == sha for p, sha in cpu['files_sha256'].items())
prepared = json.loads((R / 'followup_collector_preflight001.json').read_text())
collector = R / 'collect_followup_reward_credit001.py'
assert digest(collector) == prepared['collector_sha256']
assert digest('scripts/train_robust_getup.py') == prepared['trainer_sha256']
queue = R / 'diagnosis_queue045.py'
assert not queue.exists() and not (R / 'diagnosis_plan045.json').exists()
queue.write_text((R / 'diagnosis_queue044.py').read_text().replace('plan044', 'plan045').replace('queue044', 'queue045'))
output = B / 'diagnosis045_followup_credit'
assert not output.exists()
args = list(prepared['command'])
args[args.index('--output') + 1] = str(output)
cp = args[args.index('--resume') + 1]
job = dict(name='native_followup_credit_zero', checkpoint=cp, output=str(output),
    log=str(R / 'diagnosis045_followup_credit.log'), completion_status='stopped',
    command=['/home/noir/miniconda3/envs/isaacsim6/bin/python', str(collector), '--diagnostic-rollout-only'] + args)
protocol = dict(schema='native-followup-reward-credit-diagnosis-v1',
    requirement='Four512-step native fixed-policy windows,256envs, original push80N/32Nm/300ms at5s. First-window original-reward native sampling/GAE must match044 all16 tensors; actor/critic/Adam must remain exact. Require actual moving-fall history and nonzero reward on native continuous-followup timer before any new bounded PPO.',
    single_variable='Only training reward; original runtime function, native success/contact/timer gates, task exposure, forces, DR, masks, likelihood, GAE, optimizer namespace and std retained.',
    no_automatic_PPO=True, qualification=None, new_PPO=0,
    actual_experiment_consumed=2646, repair_optimizer_updates=266, created_unix=time.time())
write_json(R / 'followup_credit_protocol001.json', protocol)
frozen = {p: digest(p) for p in prior['source_sha256']}
extras = [Path('src/wheeled_tasks/chassis/followup_continuity_reward.py'),
    R / 'preflight_followup_reward001.py', R / 'followup_reward_preflight001.json',
    R / 'preflight_followup_collector001.py', R / 'followup_collector_preflight001.json',
    R / 'audit_followup_reward_replay001.py', R / 'followup_reward_replay001.json',
    R / 'audit_motion_followup001.py', R / 'motion_round015_followup_audit001.json',
    R / 'round015_closed_source_seal001.json', R / 'motion_round015_endpoint001.json',
    R / 'motion_round015_fixed_decision001.json', R / 'followup_credit_protocol001.json',
    B / 'diagnosis044_motion_output_native_zero/rollout_diagnosis.json', collector, queue, Path(__file__)]
frozen.update({str(p): digest(p) for p in extras})
plan = dict(schema='serial-native-diagnosis-v2', jobs=[job], source_sha256=frozen,
    protocol=str(R / 'followup_credit_protocol001.json'), new_PPO=0,
    actual_experiment_consumed=2646, repair_optimizer_updates=266,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'diagnosis_plan045.json', plan)
p = json.loads((R / 'progress.json').read_text())
p.update(status='followup_reward_credit_native_diagnosis_admitted',
    active_diagnosis_queue=str(R / 'diagnosis_queue045.json'), active_evaluation=str(output),
    active_training=None, new_PPO_admitted=0, planned_PPO='No PPO admitted; require native target exposure first.', updated_unix=time.time())
write_json(R / 'progress.json', p)
print(json.dumps(dict(plan=str(R / 'diagnosis_plan045.json'), frozen_inputs=len(frozen), new_PPO=0)))

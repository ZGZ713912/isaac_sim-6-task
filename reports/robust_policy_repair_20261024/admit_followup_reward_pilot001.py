"""An independent eight-update reward-only experiment after native credit proof."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
previous = json.loads((R / 'diagnosis_plan048.json').read_text())
assert json.loads((R / 'diagnosis_queue048.json').read_text())['status'] == 'completed'
assert json.loads((R / 'diagnosis048_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert all(digest(p) == sha for p, sha in previous['source_sha256'].items())
checks = ('followup_reward_preflight001.json', 'followup_collector_preflight002.json',
          'followup_pilot_wrapper_preflight001.json', 'followup_credit_native_audit001.json')
for name in checks:
    assert all(json.loads((R / name).read_text())['checks'].values()), name
preflight = json.loads((R / checks[0]).read_text())
assert all(digest(p) == sha for p, sha in preflight['files_sha256'].items())
wrapper = R / 'run_followup_reward_pilot001.py'
assert digest(wrapper) == json.loads((R / checks[2]).read_text())['wrapper_sha256']
assert digest('scripts/train_robust_getup.py') == json.loads((R / checks[1]).read_text())['trainer_sha256']
reference = R / 'followup_native_first_rollout_reference001.json'
ref = json.loads(reference.read_text())
assert digest(ref['native_audit']) == ref['native_audit_sha256']
assert ref['new_PPO'] == ref['optimizer_attempts'] == 0
p = json.loads((R / 'progress.json').read_text())
assert p['actual_experiment_consumed'] == 2646 and p['repair_optimizer_updates'] == 266
template = json.loads((R / 'plan015_motion_output001.json').read_text())
source = Path(template['source_checkpoint'])
assert digest(source) == template['source_checkpoint_sha256']
plan = dict(template)
training = B / 'pilot016_followup_reward'
command = list(template['training_command'])
command[1] = str(wrapper)
command[command.index('--zero-rollout-reference') + 1] = str(reference)
command[command.index('--output') + 1] = str(training)
command += ['--followup-continuity-weight', '5']
plan.update(schema='bounded-native-followup-reward-PPO-v1', training_command=command,
    supervisor_output=str(B / 'round016'), training_output=str(training),
    evaluation_output=str(B / 'pilot016_followup_fixed'),
    prior_experiment_consumed=2646, prior_repair_consumed=266,
    hypothesis='The native continuous two-second timer is repeatedly broken after recovery. A bounded quadratic density from this unchanged timer may improve uninterrupted command followup; zero-PPO046 confirms real moving-fall exposure and native GAE credit, whose normalized sign is not uniformly positive.',
    algorithm=template['algorithm'], frozen_state=template['frozen_state'],
    changes='Independent source1501 branch with exactly the same8 native updates, seed,390 trainable parameters,80pct moving exposure, original optimizer state/LRs and physical task as015. Only training reward is changed.',
    reward='Native reward plus5*0.005*clamp(native_followup_hold/2,0,1)^2*policy_learning_mask; at most.025 per step. Actual pre-autoreset timer, no success/physics/observation modification.',
    reward_contract=preflight['contract'],
    first_rollout_gate='All16 actual native rollout/GAE tensors must match046 before any optimizer attempt. Mismatch seals unchanged initial state with0attempts.',
    zero_PPO_evidence=str(R / 'followup_credit_native_audit001.json'),
    comparison='015 is the completed original-reward control. Both are independent1501to1509 branches, not sequential lineage.',
    no_automatic_extension=True, admission_receipt=str(R / 'followup_reward_pilot_admission001.json'),
    created_unix=time.time())
frozen = dict(previous['source_sha256'])
extras = [wrapper, reference, R / 'diagnosis048_closed_source_seal001.json',
    R / 'quiet_wheel_gain_development_decision001.json', Path(__file__)]
extras += [R / n for n in checks]
extras += [R / n for n in ('preflight_followup_pilot_wrapper001.py', 'preflight_followup_collector002.py')]
extras += [R / n for n in ('audit_followup_round016_endpoint001.py', 'review_followup_round016_fixed001.py',
                           'review_quiet_wheel_gain001.py', 'quiet_wheel_gain_development_pairing001.json')]
frozen.update({str(f): digest(f) for f in extras})
plan['source_sha256'] = frozen
target = R / 'plan016_followup_reward001.json'
assert not target.exists() and all(not Path(plan[k]).exists() for k in ('training_output', 'supervisor_output', 'evaluation_output'))
write_json(target, plan)
write_json(R / 'followup_reward_pilot_admission001.json', dict(schema='bounded-followup-reward-admission-v1',
    plan=str(target), plan_sha256=digest(target), source_sha256=digest(source),
    admitted_updates=8, source_clock=1501, maximum_branch_clock=1509,
    prior_experiment_consumed=2646, maximum_experiment_consumption=2654,
    maximum_repair_consumption=274, frozen_inputs=len(frozen),
    no_automatic_extension=True, qualification=None, hardware_started=False, created_unix=time.time()))
p.update(status='followup_reward_admitted_bounded8', active_diagnosis_queue=None,
    active_evaluation=None, active_training=None, active_supervisor=None,
    new_PPO_admitted=8, training_admission=str(target),
    planned_PPO='Independent8-update continuity-reward branch1501to1509, then fixed51 and unchanged strict nonregression gates. No automatic extension.',
    updated_unix=time.time())
write_json(R / 'progress.json', p)
print(json.dumps(dict(plan=str(target), frozen_inputs=len(frozen), admitted_updates=8,
    already_consumed=2646, maximum_consumption=2654)))

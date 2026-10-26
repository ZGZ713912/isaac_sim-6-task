"""One new bounded projected-Adam algorithm test from the best function root."""
import json
from pathlib import Path
import sys
import time
sys.path[:0]=['src','scripts']
from train_robust_getup import digest,write_json

R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
files=[R/'projected_entry_preflight003.json',R/'projected_entry_preflight004.json',R/'projected_entry_negative_stop001.json']
pre=[json.loads(p.read_text()) for p in files]
assert all(all(d['checks'].values()) for d in pre)
wrapper=R/'run_projected_entry_pilot001.py';projector=R/'common_phase_projector001.py'
assert all(d['wrapper_sha256']==digest(wrapper) for d in pre)
assert all(d['projector_sha256']==digest(projector) for d in pre[:2])
known=json.loads((R/'native_common_projected_fixed_decision001.json').read_text())
assert all(v for k,v in known['checks'].items() if k!='measured_improvement')
assert known['checks']['measured_improvement'] is False
assert json.loads((R/'credit_rollout_supervisor001.json').read_text())['status']=='completed_diagnosis'
assert all(json.loads((R/'credit_rollout_audit001.json').read_text())['checks'].values())
base=json.loads((R/'plan012_common_entry002.json').read_text())
progress=json.loads((R/'progress.json').read_text());assert progress['actual_experiment_consumed']==2622
frozen=json.loads((R/'credit_rollout_plan001.json').read_text())['source_sha256']
assert all(digest(Path(n))==h for n,h in frozen.items())
extra=files+[wrapper,projector,R/'preflight_projected_entry003.py',R/'preflight_projected_entry004.py',
    R/'test_projected_entry_stop001.py',R/'common_entry_endpoint001.json',R/'native_common_projected_fixed_decision001.json',
    R/'credit_rollout_audit001.json',R/'analyze_credit_rollout001.py',Path(__file__)]
frozen.update({str(p):digest(p) for p in extra})
train=str(B/'pilot013_projected_entry');command=base['training_command'].copy();command[1]=str(wrapper)
command[command.index('--output')+1]=train
plan=dict(base,schema='bounded-projected-common-PPO-v1',status='admitted_once_bounded8',
    supervisor_output=str(B/'round013'),training_output=train,training_command=command,
    evaluation_output=str(B/'pilot013_projected_fixed'),source_sha256=frozen,
    prior_experiment_consumed=2622,new_updates=8,source_clock=1501,
    admission_receipt=str(R/'projected_entry_pilot_admission001.json'),
    hypothesis='After each original-reward common-only PPO update, explicit projection preserves full-path direction/stop constraints while the remaining feasible update may improve physical recovery.',
    role='new-projected-Adam-algorithm-test',
    algorithm='Unchanged native samples/logprob/GAE/PPO losses/Adam with the same common-only gradient mask as012, followed by one bounded coefficient projection. Explicit projected Adam, not unmodified native PPO.',
    changed_training_dose='Same3e-4 LR,256x512,seed241509,DR.35/severity.8,side exposure,original reward and conditional noise as012. New intervention is the post-update phase projection and its stop checks.',
    phase_guard=dict(dataset=str(R/'entry_phase_path_dataset001.npz'),far_min_rad=.014,
        stopped_peak_max_rad=.006,knee_logit_roundoff_max=2e-5,recorded_step_target_peak_max_rad=.005,
        direction_common_logit_min=.25,stopped_common_logit_max=.06,
        max_projection_cut_rounds=8,solver_seconds_per_round=45.,
        feature_cpu_threads=1,native_thread_count_restored=True,
        failure='Restore prior feasible entry weights if solver fails; count completed native update and seal. No next rollout.',
        development_only_not_native_qualification=True),
    rationale='The one projected012 step passed every fixed physical/nonregression gate but had no measured improvement; its adoption/extension remains rejected. This new8-update algorithm experiment starts independently from the best1501 initialization, not from that rejected checkpoint.',
    predecessor_unused_budget='012 remaining7 and009 remaining6 stay closed; no refunds or transfers.',
    qualification=None,hardware_started=False,created_unix=time.time(),admitted_unix=time.time())
target=R/'plan013_projected_entry001.json';assert not target.exists() and not Path(plan['supervisor_output']).exists()
write_json(target,plan)
write_json(R/'projected_entry_pilot_admission001.json',dict(schema='bounded-projected-entry-admission-v1',
    plan=str(target),plan_sha256=digest(target),source_checkpoint=plan['source_checkpoint'],
    source_sha256=plan['source_checkpoint_sha256'],new_PPO_admitted=8,source_clock=1501,
    maximum_branch_clock=1509,prior_experiment_consumed=2622,maximum_experiment_consumption=2630,
    frozen_inputs=len(frozen),no_automatic_extension=True,qualification=None,hardware_started=False,created_unix=time.time()))
progress.update(status='projected_common_entry_admitted_bounded8',new_PPO_admitted=8,
    active_job=None,active_evaluation=None,training_admission=str(target),
    planned_PPO='One new8-update projected-Adam branch from1501, then fixed51. No automatic extension or adoption.',updated_unix=time.time())
write_json(R/'progress.json',progress)
print(json.dumps(dict(plan=str(target),frozen_inputs=len(frozen),new_PPO_admitted=8,prior_consumed=2622),indent=2))

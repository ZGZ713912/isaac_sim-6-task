"""Admit one finite native common-only entry trial after positive/negative QA."""
import json
from pathlib import Path
import sys
import time

sys.path[:0]=['src','scripts']
from train_robust_getup import digest,write_json

R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
positive=json.loads((R/'entry_common_guard_preflight001.json').read_text())
negative=json.loads((R/'entry_common_guard_negative_stop001.json').read_text())
assert all(positive['checks'].values()) and all(negative['checks'].values())
assert negative['wrapper_sha256']==digest(R/'run_entry_common_guard001.py')
assert positive['mask_source_sha256']==digest(R/'entry_common_guard001.py')
assert negative['native_outer_sha256']==digest(Path('scripts/train_robust_getup.py'))
cp=Path(positive['checkpoint'])
assert digest(cp)==positive['checkpoint_sha256']==negative['source_checkpoint_sha256']
progress=json.loads((R/'progress.json').read_text())
assert progress['actual_experiment_consumed']==2621 and progress['new_PPO_completed']==241
assert progress['active_job'] is None and progress['active_training'] is None
base=json.loads((R/'plan009_function_root_entry001.json').read_text())
frozen=json.loads((R/'plan011_quiet_tight001.json').read_text())['source_sha256']
assert all(digest(Path(n))==h for n,h in frozen.items())
files=[cp,R/'entry_phase_path_dataset001.npz',Path(__file__),
       R/'entry_common_guard001.py',R/'run_entry_common_guard001.py',
       R/'preflight_entry_common_guard001.py',R/'entry_common_guard_preflight001.json',
       R/'test_entry_common_guard_stop001.py',R/'entry_common_guard_negative_stop001.json',
       R/'run_common_guard_round001.py']
frozen.update({str(p):digest(p) for p in files})
train=str(B/'pilot012_common_entry')
command=base['training_command'].copy()
command[1]=str(R/'run_entry_common_guard001.py')
for flag,value in [('--output',train),('--resume',str(cp)),('--actor-lr','.0003')]:
    command[command.index(flag)+1]=value
plan=dict(base,schema='bounded-common-entry-round-v1',status='admitted_once_bounded8',
    supervisor_output=str(B/'round012'),training_output=train,source_checkpoint=str(cp),
    source_checkpoint_sha256=digest(cp),training_command=command,
    evaluation_output=str(B/'pilot012_common_fixed'),prior_experiment_consumed=2621,
    source_clock=1501,new_updates=8,effective_trainable_entry_parameters=258,
    source_sha256=frozen,admission_receipt=str(R/'entry_common_pilot_admission001.json'),
    hypothesis='Frozen entry feature basis and paired common-row updates may preserve the full-path direction/stop constraints while improving closed-loop recovery.',
    frozen_actor='Feature basis gradients zero; two final common rows tied per leg; knee/wheel gradients zero. Existing native Adam/likelihood/GAE unchanged. Fresh six entry slots explicitly inherited from authenticated initialization.',
    role='common-only original-reward guarded entry fine-tune',
    phase_guard=dict(dataset=str(R/'entry_phase_path_dataset001.npz'),
        far_min_rad=.014,stopped_peak_max_rad=.006,knee_logit_roundoff_max=2e-5,
        after_every_completed_update=True,violation='Count and seal completed update; stop before collecting another rollout',
        development_only_not_native_behavior=True),
    changed_training_dose='LR3e-4 for258 effective coefficients, versus1e-5 full21894 in009. This trial is not a single-variable causal comparison against009.',
    created_unix=time.time(),admitted_unix=time.time(),qualification=None,hardware_started=False)
target=R/'plan012_common_entry001.json'
assert not target.exists() and not Path(plan['supervisor_output']).exists()
write_json(target,plan)
write_json(R/'entry_common_pilot_admission001.json',dict(schema='common-only-entry-admission-v1',
    plan=str(target),plan_sha256=digest(target),source_checkpoint_sha256=digest(cp),
    positive_preflight_sha256=digest(R/'entry_common_guard_preflight001.json'),
    negative_stop_fixture_sha256=digest(R/'entry_common_guard_negative_stop001.json'),
    new_PPO_admitted=8,source_clock=1501,maximum_terminal_clock=1509,
    prior_experiment_consumed=2621,maximum_experiment_consumption=2629,
    old009_unused_six='closed, not transferred or refunded',frozen_inputs=len(frozen),
    no_automatic_extension=True,qualification=None,hardware_started=False,created_unix=time.time()))
progress.update(status='common_only_entry_admitted_bounded8',new_PPO_admitted=8,
    training_admission=str(target),planned_PPO='One new common-only branch, maximum8; per-update phase guard and fixed native review; no automatic extension.',
    lineage='Best current branch1501. Repair241 includes independent quiet branches8+8 after prior225; total2621 before012. Branch clocks and cumulative consumption remain separate.',
    updated_unix=time.time())
write_json(R/'progress.json',progress)
print(json.dumps(dict(plan=str(target),frozen_inputs=len(frozen),new_PPO_admitted=8,prior_consumed=2621),indent=2))

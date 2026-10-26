"""Admit one eight-update native fallen-output experiment after its real zero-PPO audit."""
import json
import sys
import time
from pathlib import Path

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
audit = json.loads((R / 'fallen_native_zero_audit001.json').read_text())
wrapper_test = json.loads((R / 'fallen_pilot_wrapper_preflight001.json').read_text())
prepare = json.loads((R / 'fallen_prepare_actual_preflight001.json').read_text())
assert all(audit['checks'].values()) and all(wrapper_test['checks'].values()) and all(prepare['checks'].values())
wrapper = R / 'run_fallen_output_pilot001.py'
assert digest(wrapper) == wrapper_test['wrapper_sha256']
assert digest(Path('scripts/train_robust_getup.py')) == prepare['draft_sha256']
progress = json.loads((R / 'progress.json').read_text())
assert progress['actual_experiment_consumed'] == 2630 and progress['repair_optimizer_updates'] == 250
source = Path(prepare['source_checkpoint'])
assert digest(source) == prepare['source_sha256'] == audit['source_sha256']
old = json.loads((R / 'diagnosis_plan039.json').read_text())
frozen = dict(old['source_sha256'])
assert all(digest(Path(n)) == h for n, h in frozen.items())
zero = B / 'diagnosis039_fallen_output_native_zero'
extra = [wrapper, R / 'fallen_pilot_wrapper_preflight001.json', R / 'fallen_prepare_actual_preflight001.json',
    R / 'audit_fallen_native_zero001.py', R / 'fallen_native_zero_audit001.json',
    R / 'fallen_native_zero_restore001.json', R / 'diagnosis039_closed_source_seal001.json',
    zero / 'rollout_diagnosis.json', zero / 'native_function_readback.json',
    zero / 'native_fallen_rollout_and_GAE.npz', zero / 'model_initial.pt',
    Path(__file__)]
frozen.update({str(p): digest(p) for p in extra})
train = B / 'pilot014_fallen_output'
reference = zero / 'rollout_diagnosis.json'
args = list(prepare['command'])
args[args.index('--output') + 1] = str(train)
command = ['/home/noir/miniconda3/envs/isaacsim6/bin/python', str(wrapper),
           '--zero-rollout-reference', str(reference), *args]
plan = dict(schema='bounded-native-fallen-output-PPO-v1', status='admitted_once_bounded8',
    supervisor_output=str(B / 'round014'), training_output=str(train), training_command=command,
    evaluation_output=str(B / 'pilot014_fallen_fixed'),
    baseline_evaluation=str(B / 'diagnosis025_motion_damping080_fixed'),
    bank=str(R / 'pose_bank_rmcs003.json'), source_checkpoint=str(source), source_checkpoint_sha256=digest(source),
    source_sha256=frozen, prior_experiment_consumed=2630, prior_repair_consumed=250,
    source_clock=1501, maximum_branch_clock=1509, new_updates=8, trainable_actor_parameters=774,
    hypothesis='At observed moving falls the entry head has zero output sensitivity and the fallen head has nonzero sensitivity. Train the existing fallen output layer with the existing80pct moving-push exposure.',
    algorithm='Native sampling, conditioned std, log likelihood, GAE, PPO and Adam with exactly two existing fallen output tensors trainable; no output projection or new network.',
    frozen_state='All other actor tensors, feature basis, entry, quiet feedback, std and26 frozen actor Adam slots stay exact. The eight critic slots learn normally.',
    changes='Native V5 load/save authentication and trainability scope; existing motion-recovery-focus changes only motion fraction.5to.8 and push fraction.25to1 plus version metadata.',
    reward='Unchanged source reward and density/potential; support_weight0, original noise, DR.35/severity.8/native timing and all guards.',
    first_rollout_gate='All recorded native rollout/GAE tensors must exactly match039 before optimizer attempt accounting. Mismatch seals the exact initial state with zero attempts.',
    source_clock_not_physical_continuation=True, rollout_samples=8*256*512, seed=241510,
    no_automatic_extension=True, qualification=None, hardware_started=False,
    review_gate=dict(standing_at_least=19, motion_MAE_at_most=.135,
        entry_recovery_at_least=8, entry_followup_at_least=8,
        fallen_recovery_at_least=4, fallen_followup_at_least=2,
        entry_peak_at_most_rad_s=8.911178255081177, fallen_peak_at_most_rad_s=24.11665153503418,
        impacts_at_most=2, max_gap_m=.003, passive_motor_peak_nm=0,
        expanded_RMCS_recovery_at_least=57, expanded_RMCS_followup_at_least=36,
        expanded_RMCS_tail90_at_least=56, expanded_RMCS_leg_peak_at_most_rad_s=18.907798767089844,
        expanded_RMCS_survive_at_least=70, expanded_RMCS_impacts_at_most=2,
        nominal_families_followup_at_least=dict(compound=8,inverted=10,right=4,left=2),
        motion_survival_at_least=42, actual_moving_fall_recovery_at_least=1,
        decision_order='Fixed51 physical and behavior nonregression permits zero-PPO expanded tests only. A measured gain on the actual moving-fall target and expanded nonregression are required before adoption or further PPO.',
        qualification='Original complete acceptance_protocol004, long standing and fresh holdouts still required.'),
    admission_receipt=str(R / 'fallen_output_pilot_admission001.json'), created_unix=time.time())
target = R / 'plan014_fallen_output001.json'
assert not target.exists() and not train.exists() and not Path(plan['supervisor_output']).exists()
write_json(target, plan)
write_json(R / 'fallen_output_pilot_admission001.json', dict(schema='bounded-fallen-output-admission-v1',
    plan=str(target), plan_sha256=digest(target), source_sha256=digest(source),
    new_PPO_admitted=8, source_clock=1501, maximum_branch_clock=1509,
    prior_experiment_consumed=2630, maximum_experiment_consumption=2638,
    maximum_repair_consumption=258, frozen_inputs=len(frozen), no_automatic_extension=True,
    qualification=None, hardware_started=False, created_unix=time.time()))
progress.update(status='fallen_output_admitted_bounded8', new_PPO_admitted=8,
    active_job=None, active_evaluation=None, active_supervisor=None, active_native_queue=None,
    active_diagnosis_queue=None, training_admission=str(target),
    source_promotion=None,
    source_promotion_suspended='c025 is a research component: paired development17/21 for both c025 and current9a; c025 formal30s gates fail. Best9a remains unqualified.',
    planned_PPO='One independent8-update native fallen-output branch from1501, then fixed51; nonregression permits expanded zero-PPO evaluation only. No automatic extension.',
    updated_unix=time.time())
write_json(R / 'progress.json', progress)
requirements = json.loads((R / 'completion_requirements001.json').read_text())
requirements['interim_pilot_gate'] = plan['review_gate']['decision_order']
requirements['requirements'][2]['status'] = 'Current best9a has not passed formal30s. c025 completed96 formal episodes but fails position/recentering and physical3..95Hz gates at both seeds.'
requirements['requirements'][3]['status'] = 'Current9a fixed19/21 and paired development17/21; c025 fixed20/21 and same development17/21. Both remain unqualified; no fresh holdout.'
requirements['latest_research_status'].update(latest_branch='014 admitted8 updates1501to1509 on existing774-parameter fallen output; not completed.',
    current_diagnosis='039 real zero-PPO131072-frame mean/std bit-exact and14 restore/source/counter checks pass; first training rollout must reproduce it.',
    development_standing_current9a=17, development_standing_c025=17,
    c025_formal30s='reject_robust_qualification', selected_deployment_model=None)
requirements['updated_unix'] = time.time()
write_json(R / 'completion_requirements001.json', requirements)
print(json.dumps(dict(plan=str(target), frozen_inputs=len(frozen), new_PPO_admitted=8,
    prior_consumed=2630, maximum_consumption=2638), indent=2))

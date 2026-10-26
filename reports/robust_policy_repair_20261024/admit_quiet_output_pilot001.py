"""Admit one eight-update native quiet-output pilot after zero-PPO readback."""
from pathlib import Path
import json
import re
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
previous = json.loads((R / 'diagnosis_plan054.json').read_text())
assert json.loads((R / 'diagnosis_queue054.json').read_text())['status'] == 'completed'
assert json.loads((R / 'diagnosis054_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
assert all(digest(p) == h for p, h in previous['source_sha256'].items())
for name in ('quiet_output_preflight001.json', 'quiet_output_native_zero_audit001.json', 'quiet_pilot_wrapper_preflight001.json'):
    assert all(json.loads((R / name).read_text())['checks'].values()), name
pre = json.loads((R / 'quiet_output_preflight001.json').read_text())
wrapper = R / 'run_quiet_output_pilot001.py'
assert digest(wrapper) == json.loads((R / 'quiet_pilot_wrapper_preflight001.json').read_text())['wrapper_sha256']
log = R / 'quiet_output_regression001.log'
assert re.search(r'47 passed in', log.read_text())
write_json(R / 'quiet_output_regression001.json', dict(schema='quiet-output-targeted-regression-v1',
    passed=47, exit_code=0, log=str(log), log_sha256=digest(log),
    tests=['tests/test_quiet_stance_authority.py', 'tests/test_function_root_checkpoint.py', 'tests/test_gap_tuning_precedence.py'],
    files_sha256={str(p): digest(p) for p in (Path('scripts/train_robust_getup.py'), Path('scripts/export_robust_getup.py'),
        Path('src/wheeled_algo/quiet_output_checkpoint.py'))}, new_PPO=0))
output = B / 'pilot017_quiet_output'
supervisor = B / 'round017'
evaluation = B / 'pilot017_quiet_output_fixed'
assert not any(p.exists() for p in (output, supervisor, evaluation))
args = list(pre['command']); args[args.index('--output') + 1] = str(output)
reference = B / 'diagnosis054_quiet_output_native_zero/rollout_diagnosis.json'
command = ['/home/noir/miniconda3/envs/isaacsim6/bin/python', str(wrapper), '--zero-rollout-reference', str(reference), *args]
gate = dict(standing_at_least=20, motion_MAE_at_most=.135,
    entry_recovery_at_least=9, entry_followup_at_least=9,
    fallen_recovery_at_least=4, fallen_followup_at_least=2,
    entry_peak_at_most_rad_s=8.911178255081177, fallen_peak_at_most_rad_s=24.11665153503418,
    impacts_at_most=1, max_gap_m=.003, passive_motor_peak_nm=0,
    development_standing_strictly_greater_than=17,
    decision_order='Fixed51 nonregression permits only zero-PPO development/long-standing reviews. Actual standing improvement, original moving target, RMCS nonregression and full unchanged acceptance remain required before adoption or more PPO.',
    qualification='Original acceptance_protocol004 including long30s standing and fresh holdouts.')
extra = [wrapper, R / 'preflight_quiet_pilot_wrapper001.py', R / 'quiet_pilot_wrapper_preflight001.json',
    R / 'audit_quiet_output_native_zero001.py', R / 'quiet_output_native_zero_audit001.json',
    R / 'quiet_output_native_zero_restore001.json', R / 'diagnosis054_closed_source_seal001.json',
    R / 'quiet_output_regression001.json', log, reference, Path(__file__),
    Path('scripts/run_bounded_gap_round.py')]
frozen = dict(previous['source_sha256'])
frozen.update({str(p): digest(p) for p in extra})
plan = dict(schema='bounded-native-quiet-output-PPO-v1', status='admitted_once_bounded8',
    supervisor_output=str(supervisor), training_output=str(output), evaluation_output=str(evaluation),
    training_command=command, baseline_evaluation=str(B / 'diagnosis036_quiet_stance050_implementation'),
    bank=str(R / 'pose_bank_rmcs003.json'), source_checkpoint=pre['checkpoint'],
    source_checkpoint_sha256=pre['checkpoint_sha256'], source_sha256=frozen,
    prior_experiment_consumed=2654, prior_repair_consumed=274, source_clock=1501,
    maximum_branch_clock=1509, new_updates=8, trainable_actor_parameters=1170,
    hypothesis='Learn state-dependent six-motor standing correction through the three existing quiet output layers, instead of the24 displacement-only parameters used in010/011. Current function begins exactly at c025; observer, runtime forward, std and original reward are retained.',
    algorithm='Native RSL sampling/log-probability/GAE/PPO/Adam. Exactly six existing output tensors; all22 other actor Adam slots and all other actor tensors remain exact. Eight critic tensors learn normally.',
    first_rollout_gate='All actual native sampled rollout/GAE tensor hashes must equal054 before the first optimizer attempt. Mismatch seals0attempts.',
    zero_PPO_evidence=str(R / 'quiet_output_native_zero_audit001.json'),
    source_clock_not_physical_continuation=True, rollout_samples=1048576, seed=241510,
    no_automatic_extension=True, review_gate=gate, qualification=None, hardware_started=False,
    created_unix=time.time())
plan_path = R / 'plan017_quiet_output001.json'
assert not plan_path.exists()
write_json(plan_path, plan)
p = json.loads((R / 'progress.json').read_text())
p.update(status='quiet_output_bounded8_admitted', active_training=str(output), active_supervisor=str(supervisor),
    active_evaluation=None, active_diagnosis_queue=None, active_learning_clock=1501,
    new_PPO_admitted=8, training_admission=str(plan_path),
    planned_PPO='Independent017 source1501 to at most1509,8updates; no extension.', updated_unix=time.time())
write_json(R / 'progress.json', p)
print(json.dumps(dict(plan=str(plan_path), frozen_inputs=len(frozen), new_updates=8)))

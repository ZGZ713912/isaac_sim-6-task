"""Report current-decoder donor behavior and retain its counterfactual limits."""
from pathlib import Path
import json
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
from analyze_robust_getup import analyze
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
plan = json.loads((R / 'diagnosis_plan053.json').read_text())
assert json.loads((R / 'diagnosis_queue053.json').read_text())['status'] == 'completed'
assert all(digest(p) == sha for p, sha in plan['source_sha256'].items())
new = B / 'diagnosis053_historical_recovery_donor'
old = B / 'diagnosis026_motion_damping080_nominal_pair'
results = {}
for variant in ('robust', 'reference'):
    a = analyze(new / variant)
    write_json(R / ('historical_donor_' + variant + '_analysis001.json'), a)
    report = json.loads((new / variant / 'report.json').read_text())
    results[variant] = {k: a[k] for k in ('actual_settled_attitudes', 'motion_episodes',
        'motion_survived', 'practical_moving_fall_recoveries', 'max_gap_m', 'passive_motor_peak_nm')}
    results[variant].update(strict_followup=sum(x['followup_success'] and x['survived'] for x in report['lanes']),
        impacts=sum(any('impact' in c for c in x['termination_causes']) for x in report['lanes']))
with np.load(old / 'all_precontrol_state.npz') as a, np.load(new / 'all_precontrol_state.npz') as b:
    initial = {k: bool(np.array_equal(a[k], b[k], equal_nan=True)) for k in a.files}
with np.load(old / 'robust/trajectory.npz') as a, np.load(new / 'reference/trajectory.npz') as b:
    same = np.ones(70, dtype=bool)
    for k in a.files:
        same &= np.all((a[k] == b[k]) | (np.isnan(a[k]) & np.isnan(b[k])),
                       axis=tuple(i for i in range(a[k].ndim) if i != 1))
with np.load(new / 'robust/trajectory.npz') as a, np.load(new / 'reference/trajectory.npz') as b:
    obs = bool(np.array_equal(a['observations'][0], b['observations'][0]))
runtime = json.loads((new / 'runtime_function_readback.json').read_text())
reports = {v: json.loads((new / v / 'report.json').read_text()) for v in ('robust', 'reference')}
command = plan['jobs'][0]['command']
actual_reference = command[command.index('--reference-checkpoint') + 1]
checks = dict(all_initial_native_states_exact=all(initial.values()),
    within_batch_initial_observations_exact=obs,
    exact_same_SHA_control=digest(actual_reference) == runtime['checkpoint_sha256'] == runtime['reference_sha256'],
    only_six_existing_NN_tensors=runtime['runtime_donor']['transplanted_tensor_count'] == 6,
    current_decoder_retained=runtime['runtime_donor']['current_decoder_retained'],
    historical_function_not_claimed=runtime['historical_policy_function_reproduced'] is False,
    explicit_unqualified_labels=all(
        r['evaluation_overrides']['historical_recovery_donor_probe']['eligible_for_checkpoint_mean_qualification'] is False
        for r in reports.values()))
assert all(checks.values()), checks
target = results['robust']['practical_moving_fall_recoveries'] >= 1
out = dict(schema='historical-recovery-neural-donor-result-v1', results=results,
    checks=checks, actual_moving_target_pass=target,
    decision='target_gain_requires_separate_nonregression_confirmation' if target else 'reject_historical_weight_donor',
    reference_whole_lanes_exact_to026=int(same.sum()),
    reference_nonidentical_lanes=np.flatnonzero(~same).tolist(),
    reference_job_metadata_limit=dict(
        stale_template_reference_field=plan['jobs'][0].get('reference_checkpoint'),
        actual_executed_reference=actual_reference,
        evidence='Frozen command, wrapper hash checks and native runtime/report identify the executed same-SHA9a control; stale job display field did not control native behavior.'),
    limits='Counterfactual current-decoder mean, not historical096 reproduction or a saved-checkpoint qualification. Across-run replay differences are disclosed, never treated as complete determinism.',
    new_PPO=0, actual_experiment_consumed=2654, repair_optimizer_updates=274,
    qualification=None, hardware_ready=False, created_unix=time.time())
write_json(R / 'historical_recovery_donor_result001.json', out)
write_json(R / 'diagnosis053_closed_source_seal001.json', dict(schema='native-diagnosis-source-closure-v1',
    queue=str(R / 'diagnosis_queue053.json'), status='completed', all_frozen_sources_verified=True,
    frozen_inputs=len(plan['source_sha256']), source_map_sha256=digest(R / 'diagnosis_plan053.json'),
    review_sha256=digest(__file__), new_PPO=0, actual_experiment_consumed=2654,
    qualification=None, created_unix=time.time()))
print(json.dumps(out, indent=2))

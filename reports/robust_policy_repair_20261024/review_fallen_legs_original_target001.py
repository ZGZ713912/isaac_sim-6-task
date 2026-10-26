"""Retain the original target result and explicit across-run replay limits."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
from analyze_robust_getup import analyze
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
plan = json.loads((R / 'diagnosis_plan051.json').read_text())
assert json.loads((R / 'diagnosis_queue051.json').read_text())['status'] == 'completed'
assert all(digest(p) == sha for p, sha in plan['source_sha256'].items())
old = B / 'diagnosis026_motion_damping080_nominal_pair'
new = B / 'diagnosis051_fallen_legs_original_target'
results = {}
for variant in ('robust', 'reference'):
    analysis = analyze(new / variant)
    write_json(R / ('fallen_legs_target_' + variant + '_analysis001.json'), analysis)
    report = json.loads((new / variant / 'report.json').read_text())
    results[variant] = {k: analysis[k] for k in ('actual_settled_attitudes', 'motion_episodes',
        'motion_survived', 'practical_moving_fall_recoveries', 'max_gap_m', 'passive_motor_peak_nm')}
    results[variant].update(strict_followup=sum(x['followup_success'] and x['survived'] for x in report['lanes']),
        impacts=sum(any('impact' in c for c in x['termination_causes']) for x in report['lanes']))
with np.load(old / 'all_precontrol_state.npz') as a, np.load(new / 'all_precontrol_state.npz') as b:
    assert a.files == b.files
    initial = {k: bool(np.array_equal(a[k], b[k], equal_nan=True)) for k in a.files}
with np.load(old / 'reference/trajectory.npz') as a, np.load(new / 'reference/trajectory.npz') as b:
    assert a.files == b.files
    same = np.ones(70, dtype=bool)
    arrays = {}
    for k in a.files:
        same &= np.all((a[k] == b[k]) | (np.isnan(a[k]) & np.isnan(b[k])), axis=tuple(i for i in range(a[k].ndim) if i != 1))
        arrays[k] = bool(np.array_equal(a[k], b[k], equal_nan=True))
with np.load(old / 'robust/trajectory.npz') as a, np.load(new / 'robust/trajectory.npz') as b:
    observations_exact = bool(np.array_equal(a['observations'][0], b['observations'][0]))
x, y = [json.loads((p / 'robust/report.json').read_text()) for p in (old, new)]
fields = ('name', 'command', 'sequence', 'recovery', 'push', 'push_sign', 'pose_index',
          'world_origin_m', 'first_reset_domain', 'material_coefficients')
cases = all(all(a.get(k) == b.get(k) for k in fields) for a, b in zip(x['lanes'], y['lanes'], strict=True))
checks = dict(all_precontrol_states_exact=all(initial.values()),
    robust_initial_observations_exact=observations_exact, cases_and_domains_exact=cases)
assert all(checks.values()), checks
replay = dict(reference_arrays_exact=arrays, reference_whole_lanes_exact=int(same.sum()),
    reference_nonidentical_lanes=np.flatnonzero(~same).tolist(),
    limit='Reference array/lane equality is reported without a determinism assumption. Initial native states and domains match; any reference divergence limits across-run causal attribution. Explicit counterfactual mean cannot qualify a saved checkpoint.')
runtime = json.loads((new / 'runtime_function_readback.json').read_text())
assert runtime['runtime_partition']['final_layer_runtime_sha256'] == 'fada0ed894d2d818d4eb83a0b5ccb8e0c57f53a7050198e4dc946bad945a4ecd'
assert runtime['eligible_for_checkpoint_mean_qualification'] is False
for variant in ('robust', 'reference'):
    report = json.loads((new / variant / 'report.json').read_text())
    override = report['evaluation_overrides']['fallen_legs_original_target_probe']
    assert override['schema'] == 'fallen-legs-only-original-target-diagnostic-v1'
    assert override['eligible_for_checkpoint_mean_qualification'] is False
    assert override['applied_variants'] == ['robust'] and override['reference_variant_unchanged']
checks['runtime_layer_and_explicit_unqualified_labels'] = True
out = dict(schema='fallen-legs-original-target-diagnostic-result-v1', results=results,
    pairing_checks=checks, replay=replay, original_fixed_rejection_preserved=True,
    original_moving_fall_target_pass=results['robust']['practical_moving_fall_recoveries'] >= 1,
    decision='target_gain_requires_separate_confirmation' if results['robust']['practical_moving_fall_recoveries'] else 'reject_original_target', new_PPO=0,
    actual_experiment_consumed=2654, repair_optimizer_updates=274,
    qualification=None, hardware_ready=False, created_unix=time.time())
write_json(R / 'fallen_legs_target_diagnostic_result001.json', out)
write_json(R / 'diagnosis051_closed_source_seal001.json', dict(schema='native-diagnosis-source-closure-v1',
    queue=str(R / 'diagnosis_queue051.json'), status='completed', all_frozen_sources_verified=True,
    frozen_inputs=len(plan['source_sha256']), source_map_sha256=digest(R / 'diagnosis_plan051.json'),
    review_sha256=digest(__file__), new_PPO=0, actual_experiment_consumed=2654,
    qualification=None, created_unix=time.time()))
print(json.dumps({k: v for k, v in out.items() if k != 'replay'}, indent=2))

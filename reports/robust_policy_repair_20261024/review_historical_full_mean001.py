"""Preserve the positive historical-function result and exact replay limits."""
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
plan = json.loads((R / 'diagnosis_plan055.json').read_text())
assert json.loads((R / 'diagnosis_queue055.json').read_text())['status'] == 'completed'
assert all(digest(p) == h for p, h in plan['source_sha256'].items())
new = B / 'diagnosis055_historical_full_mean_moving'
old = Path('runs/unified_robust_20261022/early_roll_push107')
results = {}
for variant in ('robust', 'early_roll20_to_40', 'early_roll20_to_50'):
    a = analyze(new / variant)
    write_json(R / ('historical_full_' + variant + '_analysis001.json'), a)
    report = json.loads((new / variant / 'report.json').read_text())
    sides = {}
    for side in ('left', 'right'):
        rows = [x for x in report['lanes'] if x['name'] == 'moving_push_' + side]
        sides[side] = dict(episodes=len(rows), survived=sum(x['survived'] for x in rows),
            recovered=sum(x['recovered'] for x in rows), strict_followup=sum(x['followup_success'] for x in rows),
            impacts=sum(any('impact' in c for c in x['termination_causes']) for x in rows),
            max_contact_impulse_ns=max(x['max_contact_impulse_ns'] for x in rows))
    results[variant] = dict(sides=sides, actual_moving_fall_followup=a['practical_moving_fall_recoveries'],
        motion_survived=a['motion_survived'], motion_episodes=a['motion_episodes'],
        max_gap_m=a['max_gap_m'], passive_motor_peak_nm=a['passive_motor_peak_nm'])
with np.load(old / 'robust/trajectory.npz') as a, np.load(new / 'robust/trajectory.npz') as b:
    common = sorted(set(a.files) & set(b.files))
    first = {k: bool(np.array_equal(a[k][0], b[k][0], equal_nan=True)) for k in common}
    before_push = {k: bool(np.array_equal(a[k][:1000], b[k][:1000], equal_nan=True)) for k in common}
    entire = {k: bool(np.array_equal(a[k], b[k], equal_nan=True)) for k in common}
runtime = json.loads((new / 'runtime_function_readback.json').read_text())
checks = dict(explicit_historical_mean=runtime['historical_mean_function_used'],
    current_observer_disclosed=runtime['current_observer_retained'],
    target_limiter_absence_disclosed=runtime['current_actor_target_limiter_retained'] is False,
    strict_followup_left8=results['robust']['sides']['left']['strict_followup'] == 8,
    actual_moving_fall_target_positive=results['robust']['actual_moving_fall_followup'] >= 1,
    physical_closure_all=all(x['max_gap_m'] < .003 and x['passive_motor_peak_nm'] == 0 for x in results.values()),
    explicit_not_qualified=runtime['eligible_for_checkpoint_mean_qualification'] is False)
assert all(checks.values()), checks
out = dict(schema='historical-full-mean-native-result-v1', results=results, checks=checks,
    decision='admit_matched_current_function_zero_PPO_control_only',
    historical_replay=dict(common_fields=common, first_frame_exact=first,
        prepush_first1000_frames_exact=before_push, entire_window_exact=entire,
        limitation='Field equality is reported explicitly. Current observer/source and actor-function version differ from current9a; no unique limiter or gate cause is inferred without matched controls.'),
    new_PPO=0, actual_experiment_consumed=2662, repair_optimizer_updates=282,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'historical_full_mean_result001.json', out)
write_json(R / 'diagnosis055_closed_source_seal001.json', dict(schema='native-diagnosis-source-closure-v1',
    queue=str(R / 'diagnosis_queue055.json'), status='completed', all_frozen_sources_verified=True,
    frozen_inputs=len(plan['source_sha256']), source_map_sha256=digest(R / 'diagnosis_plan055.json'),
    review_sha256=digest(__file__), new_PPO=0, actual_experiment_consumed=2662,
    qualification=None, created_unix=time.time()))
print(json.dumps({k: v for k, v in out.items() if k != 'historical_replay'}, indent=2))

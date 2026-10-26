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
plan = json.loads((R / 'diagnosis_plan047.json').read_text())
assert json.loads((R / 'diagnosis_queue047.json').read_text())['status'] == 'completed'
assert all(digest(p) == sha for p, sha in plan['source_sha256'].items())
old = B / 'diagnosis026_motion_damping080_nominal_pair'
new = B / 'diagnosis047_motion015_nominal_target_diagnosis'
results = {}
for variant in ('robust', 'reference'):
    analysis = analyze(new / variant)
    write_json(R / ('motion015_target_' + variant + '_analysis001.json'), analysis)
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
    limit='Reference lane39 first differs at physical contact at1.20s;69/70 reference lanes replay exactly. Initial native states and domains match, but whole-batch trajectory determinism is not established. No unique causal attribution or qualification.')
out = dict(schema='motion015-original-target-diagnostic-result-v1', results=results,
    pairing_checks=checks, replay=replay, original_fixed_rejection_preserved=True,
    original_moving_fall_target_pass=results['robust']['practical_moving_fall_recoveries'] >= 1,
    decision='reject_extension_and_adoption', new_PPO=0,
    actual_experiment_consumed=2646, repair_optimizer_updates=266,
    qualification=None, hardware_ready=False, created_unix=time.time())
write_json(R / 'motion015_target_diagnostic_result001.json', out)
write_json(R / 'diagnosis047_closed_source_seal001.json', dict(schema='native-diagnosis-source-closure-v1',
    queue=str(R / 'diagnosis_queue047.json'), status='completed', all_frozen_sources_verified=True,
    frozen_inputs=len(plan['source_sha256']), source_map_sha256=digest(R / 'diagnosis_plan047.json'),
    review_sha256=digest(__file__), new_PPO=0, actual_experiment_consumed=2646,
    qualification=None, created_unix=time.time()))
print(json.dumps({k: v for k, v in out.items() if k != 'replay'}, indent=2))

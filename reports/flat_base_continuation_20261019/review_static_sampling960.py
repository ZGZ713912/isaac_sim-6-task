"""Verify the completed paired native diagnostics without fitting any policy."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/flat_base_20261019'
HERE = Path(__file__).resolve().parent
output = HERE / 'static_sampling960_native_review001.json'
if output.exists():
    raise SystemExit('Preserve the completed diagnostic review')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
reports, arrays, checks, summaries = {}, {}, {}, {}
for hz, name in ((200, 'static_mean_sampling960_native002'),
                 (1000, 'static_mean_sampling960_1k_native001')):
    folder = RUN / name
    path = folder / 'audit.json'
    report = json.loads(path.read_text())
    if report['status'] != 'audited' or Path('/proc/' + str(report['pid'])).exists():
        raise SystemExit(f'Wait for the completed, exited native diagnostic: {name}')
    trace = np.load(folder / 'traces.npz')
    reports[hz], arrays[hz] = report, trace
    prefix = str(hz) + '_'
    checks[prefix + '500_native_steps_zero_optimization'] = (
        report['native_environment_steps'] == 500 and
        report['optimizer_updates_performed'] == report['checkpoint_files_written'] == 0)
    checks[prefix + '288_lanes_all500_valid_frames'] = (
        trace['active'].shape == (500, 288) and trace['active'].all() and
        all(v['active_frames_per_lane'] == [500] * 4 for v in report['results'].values()))
    checks[prefix + '50hz_policy_correct_physics_clock'] = report['policy_hz'] == 50 and report['physics_hz'] == hz
    checks[prefix + 'unchanged_frozen_sources'] = all(sha(ROOT / n) == v for n,v in report['source_sha256'].items())
    checks[prefix + 'trace_and_events_hashes'] = (sha(folder / 'traces.npz') == report['traces_sha256'] and
        bool(report['tensorboard_events']) and all(sha(folder / n) == v for n,v in report['tensorboard_events'].items()))
    checks[prefix + 'paired_initial_observations_and_reset_samples'] = all(
        row['jitter_samples_bit_exact'] and row['initial_obs_max_abs_difference'] == 0.
        for row in report['reset_pairing'].values())
    checks[prefix + 'nominal_training_pose_read_back'] = max(report['nominal_birth_readback'].values()) <= 1e-6
    checks[prefix + 'effective_std_equal_both_checkpoints'] = report['effective_std_policy_order']['896'] == report['effective_std_policy_order']['960']
    event = EventAccumulator(str(folder), size_guidance={'scalars': 0}).Reload()
    tags = event.Tags()['scalars']
    checks[prefix + '144_tensorboard_tags_two_correct_steps'] = len(tags) == 144 and all(
        [p.step for p in event.Scalars(tag)] == [896, 960] for tag in tags)
    groups = trace['groups']
    metrics_match, noise_pairing = [], []
    for name, row in report['results'].items():
        ids = np.flatnonzero(groups == name)
        position = trace['position'][:, ids, :2]
        drift = np.linalg.norm(position - position[:1], axis=-1).max()
        velocity = np.abs(trace['velocity'][50:, ids, 0]).mean()
        metrics_match.append(abs(float(drift) - row['drift_max_m']) < 1e-7 and
            abs(float(velocity) - row['post1s_vx_mae_m_s']) < 1e-7)
        if row['condition'] == 'mean':
            noise_pairing.append(not trace['action_noise'][:, ids].any())
        else:
            matches = [n for n,r in report['results'].items() if r['source_case'] == row['source_case'] and
                r['condition'] == row['condition']]
            noise_pairing.extend(np.array_equal(trace['action_noise'][:, ids],
                trace['action_noise'][:, np.flatnonzero(groups == other)]) for other in matches)
    checks[prefix + 'all_metrics_recomputed_from_native_trace'] = all(metrics_match)
    checks[prefix + 'common_random_numbers_within_all_factors'] = all(noise_pairing)
    by_height = defaultdict(list)
    for row in report['results'].values():
        by_height[row['source_case']].append(row)
    summaries[str(hz)] = {height: {str(step): {noise: dict(
        drift_range_m=[min(r['drift_max_m'] for r in rows if r['checkpoint'] == step and r['condition'] == noise),
            max(r['drift_max_m'] for r in rows if r['checkpoint'] == step and r['condition'] == noise)],
        passed_3cm_conditions=sum(r['drift_max_m'] <= .03 for r in rows if r['checkpoint'] == step and r['condition'] == noise),
        requested_conditions=4) for noise in ('mean', 'half_sigma', 'training_sigma')}
        for step in (896, 960)} for height,rows in by_height.items()}
checks['same_condition_design_across_clocks'] = reports[200]['conditions'] == reports[1000]['conditions']
checks['same_checkpoints_across_clocks'] = reports[200]['checkpoint_sha256'] == reports[1000]['checkpoint_sha256']
checks['identical_noise_bytes_across_clocks'] = np.array_equal(arrays[200]['action_noise'], arrays[1000]['action_noise'])
checks['identical_initial_observations_across_clocks'] = np.array_equal(
    arrays[200]['initial_observations'], arrays[1000]['initial_observations'])
review = dict(schema='flat-static-960-factorial-native-review-v1', checks=checks, passed=all(checks.values()),
    audit_sha256={str(hz): sha(RUN / name / 'audit.json') for hz,name in
        ((200,'static_mean_sampling960_native002'),(1000,'static_mean_sampling960_1k_native001'))},
    checkpoints=reports[200]['checkpoint_sha256'], summaries=summaries,
    native_policy_steps=1000, native_physics_steps=12000, optimizer_updates_performed=0,
    behavior_qualified=False, limitations='Same seeded replicas and common action noise; diagnostic conditions are not release evidence. '
        'Factor effects do not identify a unique physical or learning cause. No evaluation trace was used to fit weights.')
if not review['passed']:
    print(json.dumps(dict(checks=checks), indent=2))
    raise SystemExit('An independent diagnostic check failed; preserve native outputs')
output.write_text(json.dumps(review, indent=2, allow_nan=False) + '\n')
print(json.dumps(review, indent=2))

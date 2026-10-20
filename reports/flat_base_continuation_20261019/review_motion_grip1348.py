"""Seal the actual motion-only pilot and compare every basic behavior with1338."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_flat_base import compare
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_tasks.chassis.flat_base_continuation import digest, encode

torch.set_num_threads(1)
read = lambda path: json.loads(path.read_text())
candidate = HERE / 'motion_grip_candidate001'
target = candidate / 'completion001.json'
if target.exists():
    raise SystemExit('Preserve the completed motion grip review')
queue_path = RUN / 'motion_grip1348_queue001.json'
queue = read(queue_path)
if queue['status'] != 'complete':
    raise SystemExit('Wait for the complete native pilot and fixed replay')
for key in ('pid', 'worker_pid'):
    path = Path('/proc') / str(queue[key])
    if path.exists() and path.joinpath('stat').read_text().split(') ', 1)[1].split()[0] != 'Z':
        raise SystemExit('Wait for both bounded queue owners to exit')
prep = read(candidate / 'preparation001.json')
trial = RUN / 'pilot030_motion_grip1348'
checkpoint_path = trial / 'model_001348.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
info = checkpoint['infos']
baseline_path = ROOT / prep['baseline_file']
evaluation_path = RUN / 'ground025_motion1348_basic001/evaluation.json'
baseline, evaluation = read(baseline_path), read(evaluation_path)
checks = dict(three_children_exit0=len(queue['history']) == 3 and all(row['exit_code'] == 0 for row in queue['history']),
    exact_baseline=digest(baseline_path) == prep['baseline_sha256'],
    all_frozen_execution_sources_exact=all(digest(ROOT / name) == value for name, value in queue['source_sha256'].items()),
    exact_checkpoint=digest(checkpoint_path) == queue['checkpoint_sha256'] == evaluation['checkpoint_sha256'],
    final_counters=info['successful_updates_total'] == 1348 and info['actor_updates_total'] == 1164
        and info['motion_grip_actor_updates'] == 8 and info['static_grip_actor_updates'] == 8 and info['critic_warmup_remaining'] == 0,
    no_hardware_or_subsequent_training=not queue['hardware_motion_started'] and not queue['subsequent_training_queued'])
for name in ('native_entry001', 'native_1339', 'native_1341', 'native_1348'):
    row = read(candidate / (name + '.json'))
    checks[name] = row['passed'] and all(row['checks'].values())
pair = compare(baseline, evaluation)
maximum_path = RUN / 'ground025_motion1348_maximum_envelope001/evaluation.json'
maximum_baseline_path = RUN / 'ground025_static1338_maximum_envelope001/evaluation.json'
maximum, maximum_baseline = read(maximum_path), read(maximum_baseline_path)
maximum_pair = compare(maximum_baseline, maximum)
checks['exact_maximum_checkpoint_and_replay'] = (maximum['status'] == 'evaluated'
    and maximum['checkpoint_sha256'] == digest(checkpoint_path)
    and digest(maximum_path) == queue['maximum_evaluation_sha256']
    and len(maximum['result']['cases']) == 22
    and all(row['episodes'] == row['requested_episodes'] == 4 for row in maximum['result']['cases'].values()))
checks['maximum_identical_material_samples'] = (maximum['startup']['contact_material']['ground_static_friction_per_environment']
    == maximum_baseline['startup']['contact_material']['ground_static_friction_per_environment'])
events = EventAccumulator(str(trial), size_guidance={'scalars': 0})
events.Reload()
steps = list(range(1339, 1349))
motion_steps = events.Scalars('FlatTransfer/motion_grip_actor_updates')
checks['ten_completed_steps_in_RSL_event'] = [row.step for row in motion_steps] == steps
checks['two_critic_then_eight_actor_in_RSL_event'] = [row.value for row in motion_steps] == [0., 0.] + list(range(1, 9))
expected = {evaluation['tensorboard_namespace'] + '/passed_cases':sum(row['passed'] for row in evaluation['result']['cases'].values())}
for name, row in evaluation['result']['cases'].items():
    for metric in ('survival_rate', 'height_mae_m', 'vx_mae_m_s', 'yaw_mae_rad_s',
                   'stand_drift_max_m', 'planar_speed_mean_m_s', 'closure_gap_max_m'):
        expected[evaluation['tensorboard_namespace'] + '/' + name + '/' + metric] = row[metric]
expected[maximum['tensorboard_namespace'] + '/passed_cases'] = sum(row['passed'] for row in maximum['result']['cases'].values())
for name, row in maximum['result']['cases'].items():
    for metric in ('survival_rate', 'height_mae_m', 'vx_mae_m_s', 'yaw_mae_rad_s',
                   'stand_drift_max_m', 'planar_speed_mean_m_s', 'closure_gap_max_m'):
        expected[maximum['tensorboard_namespace'] + '/' + name + '/' + metric] = row[metric]
tags = set(events.Tags()['scalars'])
checks['all_basic_metrics_in_RSL_event'] = all(tag in tags and len(events.Scalars(tag)) == 1
    and events.Scalars(tag)[0].step == 1348
    and math.isclose(events.Scalars(tag)[0].value, value, rel_tol=2e-7, abs_tol=2e-8)
    for tag, value in expected.items())
source = ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx'
parent = mean_from_checkpoint(torch.load(ROOT / prep['parent_checkpoint'], map_location='cpu', weights_only=True), source)
child = mean_from_checkpoint(checkpoint, source)
trace_path = baseline_path.with_name('traces.npz')
with np.load(trace_path) as trace:
    obs = torch.from_numpy(trace['observations'][trace['active'].astype(bool)].copy())
static = (child.moving_mix(obs).squeeze(-1) == 0.) & (child.authority_gate(obs).squeeze(-1) == 0.)
assert int(static.sum()) > 1000
with torch.inference_mode():
    checks['same_input_stable_static_function_exact'] = torch.equal(parent(obs[static]), child(obs[static]))
losses = {}
for name in ('value', 'approx_policy_kl', 'policy_clip_fraction', 'actor_gradient_norm', 'critic_gradient_norm'):
    points = events.Scalars('Loss/' + name)
    checks['finite_native_' + name] = len(points) == 10 and [row.step for row in points] == list(range(1338, 1348))
    checks['finite_native_' + name] &= all(math.isfinite(row.value) for row in points)
    losses[name] = dict(critic_only=[row.value for row in points[:2]], motion_actor=[row.value for row in points[2:]])
termination = {}
plan = read(ROOT / info['plan_file'])
for flag in info['termination_metrics_contract']['flags']:
    counts = {step: 0 for step in steps}
    for group in plan['train_groups']:
        for band in ('height_low', 'height_nominal', 'height_high'):
            tag = f"FlatTermination/whole/{group['name']}/{band}/{flag}_events"
            points = events.Scalars(tag)
            checks['termination_points_' + flag] = checks.get('termination_points_' + flag, True) and (
                [row.step for row in points] == steps and all(row.value >= 0 and int(row.value) == row.value for row in points))
            for row in points:
                counts[row.step] += int(row.value)
    termination[flag] = dict(flagged_policy_frames=sum(counts.values()), by_completed_update=counts)
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
target.write_bytes(encode(dict(schema='flat-motion-grip-completion-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), execution_verified=True, checks=checks,
    queue_sha256=digest(queue_path), checkpoint_sha256=digest(checkpoint_path),
    baseline_sha256=digest(baseline_path), evaluation_sha256=digest(evaluation_path),
    basic_behavior_passed=evaluation['result']['passed'], comparison=pair,
    passed_cases=sum(row['passed'] for row in evaluation['result']['cases'].values()),
    requested_cases=len(evaluation['result']['cases']),
    cases=evaluation['result']['cases'], maximum_comparison=maximum_pair,
    maximum_baseline_sha256=digest(maximum_baseline_path), maximum_evaluation_sha256=digest(maximum_path),
    maximum_cases=maximum['result']['cases'],
    maximum_passed_cases=sum(row['passed'] for row in maximum['result']['cases'].values()),
    PPO_diagnostics=losses, native_termination_counts=termination,
    native_count_scope='Whole-window group/height strata only; command subviews excluded. Flags may overlap and are not inferred unique episodes.',
    same_input_static_probe_frames=int(static.sum()),
    event_file_sha256={path.name:digest(path) for path in trial.glob('events.out.tfevents.*')},
    new_optimizer_updates_by_reader=0, full_training_admitted=False, qualified=False,
    scope='Completed pilot, all basic cases and full maximum-height replay; full default-height range, independent seed, self-righting and deployment remain unqualified')))
print(json.dumps(dict(execution_verified=True, basic_passed=evaluation['result']['passed'],
    passed_cases=sum(row['passed'] for row in evaluation['result']['cases'].values()),
    lost_checks=pair['lost_baseline_checks'], maximum_passed_cases=sum(row['passed'] for row in maximum['result']['cases'].values()),
    maximum_lost_checks=maximum_pair['lost_baseline_checks'])))

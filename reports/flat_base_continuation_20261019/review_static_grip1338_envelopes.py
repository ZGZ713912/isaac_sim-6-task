"""Audit all four current-material RMCS replay cohorts without changing judgments."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_flat_base import compare
from wheeled_tasks.chassis.flat_base_continuation import digest, encode

read = lambda path: json.loads(path.read_text())
candidate = HERE / 'static_grip_candidate002'
target = candidate / 'envelopes_completion001.json'
if target.exists():
    raise SystemExit('Preserve the completed envelope audit')
queue_path = RUN / 'static_grip1338_envelopes_queue001.json'
queue = read(queue_path)
if queue['status'] != 'complete':
    raise SystemExit('Wait for all four full native replay cohorts')
if any(Path('/proc/' + str(queue[key])).exists() for key in ('pid', 'worker_pid')):
    raise SystemExit('Wait for both serial replay owners to exit')
checks = dict(four_children_exit0=len(queue['history']) == 4 and all(row['exit_code'] == 0 for row in queue['history']),
    exact_execution_contract=digest(candidate / 'envelopes_execution001.json') == queue['contract_sha256'],
    frozen_sources_exact=all(digest(ROOT / name) == value for name, value in queue['source_sha256'].items()),
    no_additional_learning_or_hardware=queue['new_optimizer_updates'] == 0
        and not queue['subsequent_training_queued'] and not queue['hardware_motion_started'])
rows, comparisons, relative = {}, {}, {}
for height in ('default', 'maximum'):
    cohort = {}
    for kind, step, trial in (('parent1328', 1328, 'pilot028_wheel_rate1328'), ('static1338', 1338, 'pilot029_static_grip1338')):
        name = f'ground025_{kind}_{height}_envelope001'
        path = RUN / name / 'evaluation.json'
        report = read(path)
        history = next(row for row in queue['history'] if row['phase'] == name)
        material = report['startup']['contact_material']
        cases = report['result']['cases']
        checkpoint_path = RUN / trial / f'model_{step:06d}.pt'
        checks[name + '_identity'] = (digest(path) == history['evaluation_sha256']
            and report['checkpoint_sha256'] == digest(checkpoint_path)
            and report['status'] == 'evaluated' and report['seed'] == 191029
            and (report['policy_hz'], report['physics_hz']) == (50, 1000))
        checks[name + '_material'] = (material['ground_static_range'] == material['ground_dynamic_range'] == [.2, .5]
            and material['wheel_static_friction'] == material['wheel_dynamic_friction'] == .1
            and material['native_readback']['usd_collision_bindings_verified']
            and material['native_readback']['physx_robot_shape_coefficients_verified'])
        checks[name + '_coverage'] = (len(cases) == 22 and all(row['requested_episodes'] == 4 and row['episodes'] == 4
            for row in cases.values()))
        event = EventAccumulator(str(RUN / trial), size_guidance={'scalars': 0})
        event.Reload()
        tags = set(event.Tags()['scalars'])
        expected = {report['tensorboard_namespace'] + '/passed_cases': sum(row['passed'] for row in cases.values())}
        for case_name, row in cases.items():
            for metric in ('survival_rate', 'height_mae_m', 'vx_mae_m_s', 'yaw_mae_rad_s',
                           'stand_drift_max_m', 'planar_speed_mean_m_s', 'closure_gap_max_m'):
                expected[report['tensorboard_namespace'] + '/' + case_name + '/' + metric] = row[metric]
        checks[name + '_RSL_events'] = all(tag in tags and len(event.Scalars(tag)) == 1
            and event.Scalars(tag)[0].step == step
            and math.isclose(event.Scalars(tag)[0].value, value, rel_tol=2e-7, abs_tol=2e-8)
            for tag, value in expected.items())
        rows[name] = dict(evaluation_sha256=digest(path), traces_sha256=digest(path.with_name('traces.npz')),
            checkpoint_sha256=report['checkpoint_sha256'],
            passed_cases=sum(row['passed'] for row in cases.values()), requested_cases=22,
            full_horizon_episodes=sum(row['full_horizon_episodes'] for row in cases.values()), requested_episodes=88,
            failed_checks={case_name:[key for key, value in row['checks'].items() if not value]
                for case_name, row in cases.items() if not row['passed']}, cases=cases,
            ground_samples=material['ground_static_friction_per_environment'])
        cohort[kind] = report
    comparison = compare(cohort['parent1328'], cohort['static1338'])
    checks[height + '_saved_comparison_exact'] = comparison == queue['comparisons'][height]
    checks[height + '_identical_ground_samples'] = (cohort['parent1328']['startup']['contact_material']['ground_static_friction_per_environment']
        == cohort['static1338']['startup']['contact_material']['ground_static_friction_per_environment'])
    comparisons[height] = comparison
    relative[height] = []
    for name, row in cohort['static1338']['result']['cases'].items():
        old = cohort['parent1328']['result']['cases'][name]
        for metric, tolerance in (('vx_mae_m_s', .02), ('yaw_mae_rad_s', .05)):
            if row[metric] > old[metric] + tolerance:
                relative[height].append(dict(case=name, metric=metric, before=old[metric], after=row[metric], tolerance=tolerance))
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
target.write_bytes(encode(dict(schema='flat-static-grip-paired-envelopes-completion-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), execution_verified=True, checks=checks,
    queue_sha256=digest(queue_path), cohorts=rows, comparisons=comparisons, relative_regressions=relative,
    no_lost_parent_checks=all(row['no_lost_baseline_checks'] for row in comparisons.values()),
    all_complete_behavior_targets_passed=all(row['all_behavior_targets_passed'] for row in comparisons.values()),
    new_optimizer_updates=0, full_training_admitted=False, qualified=False,
    scope='Complete paired development replays at both packaged heights; no independent-seed, recovery or hardware qualification inferred')))
print(json.dumps(dict(execution_verified=True,
    cohorts={name:[row['passed_cases'],row['full_horizon_episodes']] for name,row in rows.items()},
    lost_checks={name:row['lost_baseline_checks'] for name,row in comparisons.items()}, relative_regressions=relative)))

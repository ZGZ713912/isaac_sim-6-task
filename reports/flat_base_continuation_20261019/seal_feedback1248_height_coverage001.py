"""Read and bind completed height coverage without granting release status."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'runs/flat_base_20261019'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
queuepath = RUN/'feedback1248_height_coverage_queue001.json'
output = HERE/'feedback1248_height_coverage_completion001.json'
if output.exists():
    raise SystemExit('Preserve completed coverage receipt')
queue = read(queuepath)
if queue['status'] != 'complete' or any(Path('/proc/'+str(queue[k])).exists() for k in ('pid','worker_pid')):
    raise SystemExit('Wait for the development queue and all its native owners to exit')
checks = dict(all_four_native_children_exit0=len(queue['history']) == 4
        and all(r['exit_code'] == 0 for r in queue['history']),
    all128_frozen_sources_unchanged=len(queue['source_sha256']) == 128
        and all(sha(ROOT/n) == v for n, v in queue['source_sha256'].items()),
    original_pilot_seal_unchanged=sha(HERE/'feedback1248_queue_completion001.json') == queue['previous_completion_sha256'],
    checkpoint_unchanged=sha(RUN/'pilot023/model_001248.pt') == queue['checkpoint_sha256'],
    no_new_training_or_heldout=queue['new_optimizer_updates'] == 0 and not queue['heldout_seed_used']
        and not queue['subsequent_training_queued'] and not queue['hardware_motion_started'])
pilot = EventAccumulator(str(RUN/'pilot023'), size_guidance={'scalars':0}); pilot.Reload()
cohorts = {}
metric_names = ('survival_rate','height_mae_m','vx_mae_m_s','yaw_mae_rad_s',
    'stand_drift_max_m','planar_speed_mean_m_s','closure_gap_max_m')
for row in queue['history']:
    name = row['phase']
    path = RUN/name/'evaluation.json'
    data = read(path)
    original = name == 'rmcs_maximum_source_envelope001'
    cases = data['result']['cases']
    checks[name+'_identity'] = (data['status'] == 'evaluated' and sha(path) == row['evaluation_sha256']
        and data['seed'] == 191029 and data['physics_hz'] == 1000 and data['policy_hz'] == 50
        and data['new_optimizer_updates'] == 0 and len(cases) == 22
        and data.get('checkpoint_sha256') == (None if original else queue['checkpoint_sha256']))
    accumulator = pilot
    if original:
        accumulator = EventAccumulator(str(RUN/name), size_guidance={'scalars':0}); accumulator.Reload()
    expected = {data['tensorboard_namespace']+'/passed_cases':sum(c['passed'] for c in cases.values())}
    for case_name, case in cases.items():
        for metric in metric_names:
            expected[data['tensorboard_namespace']+'/'+case_name+'/'+metric] = case[metric]
    event_matches = []
    for tag, value in expected.items():
        points = accumulator.Scalars(tag)
        event_matches.append(len(points) == 1 and points[0].step == (0 if original else 1248)
            and math.isclose(points[0].value, value, rel_tol=2e-7, abs_tol=2e-8))
    checks[name+'_all155_metrics_in_existing_TensorBoard'] = len(event_matches) == 155 and all(event_matches)
    if row['comparison']:
        comparison = row['comparison']
        checks[name+'_comparison_bound'] = sha(ROOT/comparison['path']) == comparison['sha256']
    cohorts[name] = dict(evaluation_sha256=sha(path), passed_cases=sum(c['passed'] for c in cases.values()),
        requested_cases=len(cases), full_horizon_episodes=sum(c['full_horizon_episodes'] for c in cases.values()),
        requested_episodes=sum(c['requested_episodes'] for c in cases.values()),
        accounted_episodes=sum(c['episodes'] for c in cases.values()),
        termination_totals={k:sum(c['termination_reasons'][k] for c in cases.values())
            for k in next(iter(cases.values()))['termination_reasons']},
        comparison=row['comparison'],
        tensorboard_namespace=data['tensorboard_namespace'], tensorboard_metrics_checked=len(event_matches),
        metrics={n:{**{k:c[k] for k in metric_names},'checks':c['checks'],
            'full_horizon_episodes':c['full_horizon_episodes'], 'termination_reasons':c['termination_reasons']}
            for n,c in cases.items()},
        traces_sha256=sha(RUN/name/'traces.npz'))
report = dict(schema='flat-feedback1248-height-coverage-completion-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=all(checks.values()), checks=checks,
    queue_sha256=sha(queuepath), checkpoint_sha256=queue['checkpoint_sha256'], cohorts=cohorts,
    event_file_sha256={str(p.relative_to(ROOT)):sha(p) for folder in (RUN/'pilot023',RUN/'rmcs_maximum_source_envelope001')
        for p in sorted(folder.glob('events.out.tfevents.*'))},
    new_optimizer_updates=0, heldout_seed_used=False, behavior_qualified=False,
    scope='Development coverage receipt only; reports every failed target and lost source check.',
    all_release_cohorts_complete=False, hardware_motion_started=False)
if not report['passed']:
    print(json.dumps(checks, indent=2)); raise SystemExit('Height coverage artifact or TensorBoard checks failed')
output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps({n:{k:c[k] for k in ('passed_cases','requested_cases','full_horizon_episodes',
    'requested_episodes','termination_totals','comparison')} for n,c in cohorts.items()}, indent=2))

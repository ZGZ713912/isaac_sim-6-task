"""Seal the sixteen-update v6 wheel-head trial from composed1280 and all three original development replays."""
from datetime import datetime, timezone
import hashlib
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
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_flat_base import compare
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
out = HERE / 'wheel_from1280_completion001.json'
if out.exists():
    raise SystemExit('Preserve the completed trial seal')
queue_path = RUN / 'wheel_from1280_queue001.json'
queue = read(queue_path)
if queue['status'] != 'complete' or any(Path('/proc/' + str(queue[k])).exists() for k in ('pid', 'worker_pid')):
    raise SystemExit('Wait for all bounded trial owners to exit')
contract_path = HERE / 'wheel_from1280_contract001.json'
contract = read(contract_path)
checkpoint = RUN / 'pilot027_from1280/model_001296.pt'
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
info = state['infos']
native_path = HERE / 'wheel_from1280_native_final001.json'
native = read(native_path)
checks = dict(all_five_children_exit0=len(queue['history']) == 5 and all(r['exit_code'] == 0 for r in queue['history']),
    first_native_passed=queue['first_native_verification_exit_code'] == 0,
    full_native_state_learning_proof=native['passed'] and all(native['checks'].values()),
    exact_contract=sha(contract_path) == queue['contract_sha256'],
    all_frozen_sources_exact=all(sha(ROOT / name) == value for name, value in queue['source_sha256'].items()),
    exact_final_checkpoint=sha(checkpoint) == queue['checkpoint_sha256'],
    exact_lifetime_counts=(info['successful_updates_total'], info['actor_updates_total'],
        info['active_branch_actor_updates'], info['feedback_branch_actor_updates'], info['post_split_actor_updates'],
        info['inherited_shared_feedback_updates'], info['consumed_optimizer_updates_total'],
        info['critic_warmup_remaining'], info['training_transitions_total']) == (1296, 1116, 44, 738, 434, 304, 1297, 0, 80216064),
    no_long_training_heldout_or_hardware=not queue['subsequent_training_queued']
        and not queue['heldout_seed_used'] and not queue['hardware_motion_started'])
events = EventAccumulator(str(RUN / 'pilot027_from1280'), size_guidance={'scalars': 0})
events.Reload()
tags = set(events.Tags()['scalars'])
cohorts = {}
specs = (
    ('wheel_from1280_basic001', 'deployment1280_basic001', 'static_recall1280_basic001',
     'deployment1248_basic001', 'deployment_source001'),
    ('wheel_from1280_default001', 'rmcs_default1280_envelope001', 'static_recall1280_default_envelope001',
     'rmcs_default1248_envelope001', 'rmcs_default_source_envelope001'),
    ('wheel_from1280_maximum001', 'rmcs_maximum1280_envelope001', 'static_recall1280_maximum_envelope001',
     'rmcs_maximum1248_envelope001', 'rmcs_maximum_source_envelope001'))
for name, parent_name, initial_name, old_name, source_name in specs:
    path = RUN / name / 'evaluation.json'
    data = read(path)
    row = next(r for r in queue['history'] if r['phase'] == name)
    checks[name + '_identity'] = (data['status'] == 'evaluated' and data['checkpoint_sha256'] == sha(checkpoint)
        and sha(path) == row['evaluation_sha256'])
    cases = data['result']['cases']
    expected = {data['tensorboard_namespace'] + '/passed_cases':sum(c['passed'] for c in cases.values())}
    for case_name, case in cases.items():
        for metric in ('survival_rate', 'height_mae_m', 'vx_mae_m_s', 'yaw_mae_rad_s',
                       'stand_drift_max_m', 'planar_speed_mean_m_s', 'closure_gap_max_m'):
            expected[data['tensorboard_namespace'] + '/' + case_name + '/' + metric] = case[metric]
    points_ok = []
    for tag, value in expected.items():
        points = events.Scalars(tag) if tag in tags else []
        points_ok.append(len(points) == 1 and points[0].step == 1312
            and math.isclose(points[0].value, value, rel_tol=2e-7, abs_tol=2e-8))
    checks[name + '_RSL_events'] = all(points_ok)
    comparisons = {}
    for label, reference in (('parent1280', parent_name), ('initial_static_composition', initial_name),
                             ('parent1248', old_name), ('source', source_name)):
        old_path = RUN / reference / 'evaluation.json'
        old = read(old_path)
        pair = compare(old, data)
        relative = []
        for case_name, case in cases.items():
            vx, yaw, _ = case['command']
            if abs(vx) >= .75 or abs(yaw) >= 1.5:
                for metric, tolerance in (('vx_mae_m_s', .02), ('yaw_mae_rad_s', .05)):
                    before = old['result']['cases'][case_name][metric]
                    if case[metric] > before + tolerance:
                        relative.append(dict(case=case_name, metric=metric, before=before,
                                             after=case[metric], tolerance=tolerance))
        comparisons[label] = dict(reference_evaluation_sha256=sha(old_path),
            lost_checks=pair['lost_baseline_checks'], no_lost_checks=pair['no_lost_baseline_checks'],
            relative_regressions=relative, changes=pair['changes'])
    trace_path = RUN / name / 'traces.npz'
    trace = np.load(trace_path)
    reference_trace = np.load(RUN / old_name / 'traces.npz')
    equality = {}
    for case_name in ('stand_230mm', 'stand_305mm', 'stand_430mm'):
        lanes = trace['groups'] == case_name
        old_lanes = reference_trace['groups'] == case_name
        active = trace['active'][:, lanes].astype(bool)
        equality[case_name] = {key:bool(np.array_equal(trace[key][:, lanes][active],
            reference_trace[key][:, old_lanes][active]))
            for key in ('observations', 'actions', 'velocity', 'position', 'height', 'gravity')}
    trace.close()
    reference_trace.close()
    cohorts[name] = dict(evaluation_sha256=sha(path), traces_sha256=sha(trace_path),
        passed_cases=sum(c['passed'] for c in cases.values()), requested_cases=len(cases),
        full_horizon_episodes=sum(c['full_horizon_episodes'] for c in cases.values()),
        requested_episodes=sum(c['requested_episodes'] for c in cases.values()),
        comparisons=comparisons, static1248_trace_bitwise=equality, cases=cases)
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
out.write_text(json.dumps(dict(schema='flat-wheel-from1280-completion-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
    queue_sha256=sha(queue_path), contract_sha256=sha(contract_path), checkpoint_sha256=sha(checkpoint),
    native_verification_sha256=sha(native_path), cohorts=cohorts,
    event_file_sha256={p.name:sha(p) for p in (RUN / 'pilot027_from1280').glob('events.out.tfevents.*')},
    behavior_qualified=False, long_training_eligible=False, all_release_cohorts_complete=False,
    heldout_seed_used=False, hardware_motion_started=False,
    scope='Completed state and development replay sealing. Individual failures remain failures; no long training or deployment qualification is inferred.'), indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=True, checks=len(checks), cohorts={n:{k:v for k,v in d.items()
    if k in ('passed_cases', 'requested_cases', 'full_horizon_episodes', 'requested_episodes')}
    for n,d in cohorts.items()}), indent=2))

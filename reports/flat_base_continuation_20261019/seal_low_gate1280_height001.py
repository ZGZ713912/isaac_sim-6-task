"""Seal the two1280 height development cohorts after both owners exit."""
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
output = HERE/'low_gate1280_height_completion001.json'
if output.exists():
    raise SystemExit('Preserve the completed height receipt')
queue_path = RUN/'low_gate1280_height_queue001.json'
queue = read(queue_path)
if queue['status'] != 'complete' or any(Path('/proc/'+str(queue[k])).exists() for k in ('pid', 'worker_pid')):
    raise SystemExit('Wait for both replays and all their owners to exit')
contract_path = HERE/'low_gate1280_height_contract001.json'
contract = read(contract_path)
basic_path = HERE/'low_gate1280_completion001.json'
basic = read(basic_path)
checkpoint = RUN/'pilot025/model_001280.pt'
checks = dict(two_serial_children_exit0=len(queue['history']) == 2
    and all(row['exit_code'] == 0 for row in queue['history']),
    same_complete1280_checkpoint=sha(checkpoint) == queue['checkpoint_sha256'] == basic['checkpoint_sha256'],
    bounded_trial_seal_bound=basic['passed'] and sha(basic_path) == contract['bounded_completion_sha256'],
    exact_height_contract=sha(contract_path) == queue['contract_sha256'],
    all_frozen_execution_sources_exact=all(sha(ROOT/name) == value for name,value in queue['source_sha256'].items()),
    no_optimizer_heldout_or_hardware=queue['optimizer_updates'] == 0 and not queue['subsequent_training_queued']
        and not queue['heldout_seed_used'] and not queue['hardware_motion_started'])
events = EventAccumulator(str(checkpoint.parent), size_guidance={'scalars': 0})
events.Reload()
tags = set(events.Tags()['scalars'])
cohorts = {}
for row in queue['history']:
    name = row['phase']
    path = RUN/name/'evaluation.json'
    data = read(path)
    checks[name+'/identity'] = (data['status'] == 'evaluated' and sha(path) == row['evaluation_sha256']
        and data['checkpoint_sha256'] == sha(checkpoint) and data['seed'] == 191029
        and data['policy_hz'] == 50 and data['physics_hz'] == 1000)
    checks[name+'/comparisons'] = all(sha(ROOT/value['path']) == value['sha256']
        for value in row['comparisons'].values())
    cases = data['result']['cases']
    expected = {data['tensorboard_namespace']+'/passed_cases': sum(c['passed'] for c in cases.values())}
    for case_name, case in cases.items():
        for metric in ('survival_rate', 'height_mae_m', 'vx_mae_m_s', 'yaw_mae_rad_s',
                       'stand_drift_max_m', 'planar_speed_mean_m_s', 'closure_gap_max_m'):
            expected[data['tensorboard_namespace']+'/'+case_name+'/'+metric] = case[metric]
    event_checks = []
    for tag, value in expected.items():
        points = events.Scalars(tag) if tag in tags else []
        event_checks.append(len(points) == 1 and points[0].step == 1280
            and math.isclose(points[0].value, value, rel_tol=2e-7, abs_tol=2e-8))
    checks[name+'/existing_RSL_events'] = all(event_checks)
    cohorts[name] = dict(evaluation_sha256=sha(path), traces_sha256=sha(RUN/name/'traces.npz'),
        passed_cases=sum(c['passed'] for c in cases.values()), requested_cases=len(cases),
        full_horizon_episodes=sum(c['full_horizon_episodes'] for c in cases.values()),
        requested_episodes=sum(c['requested_episodes'] for c in cases.values()),
        termination_reasons={reason: sum(c['termination_reasons'].get(reason, 0) for c in cases.values())
            for reason in sorted({r for c in cases.values() for r in c['termination_reasons']})},
        comparisons=row['comparisons'])
if not all(checks.values()):
    print(json.dumps(checks, indent=2))
    raise SystemExit('Height identity or metrics sealing failed; preserve all outputs')
result = dict(schema='flat-low-gate1280-height-development-completion-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
    queue_sha256=sha(queue_path), contract_sha256=sha(contract_path), bounded_completion_sha256=sha(basic_path),
    checkpoint_sha256=sha(checkpoint), cohorts=cohorts,
    event_file_sha256={p.name: sha(p) for p in checkpoint.parent.glob('events.out.tfevents.*')},
    all_release_cohorts_complete=False, behavior_qualified=False, optimizer_updates=0,
    heldout_seed_used=False, hardware_motion_started=False)
output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(passed=True, checks=len(checks), cohorts=cohorts), indent=2))

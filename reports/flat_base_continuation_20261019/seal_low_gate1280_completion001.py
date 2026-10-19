"""Seal the gate-only bounded trial and its first fixed development screen."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
queue_path = RUN / 'low_gate_1280_queue001.json'
queue = read(queue_path)
OUT = HERE / 'low_gate1280_completion001.json'
if OUT.exists():
    raise SystemExit('Preserve the completed trial seal')
if queue['status'] != 'complete' or any(Path('/proc/' + str(queue[k])).exists() for k in ('pid','worker_pid')):
    raise SystemExit('Wait for every bounded trial owner to exit')
contract_path = HERE / 'low_gate_1280_contract001.json'
contract = read(contract_path)
folder = RUN / 'pilot025'
checkpoint = folder / 'model_001280.pt'
native_path = HERE / 'low_gate_native1280.json'
native = read(native_path)
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
info = state['infos']
evaluation_path = RUN / 'deployment1280_basic001/evaluation.json'
evaluation = read(evaluation_path)
cases = evaluation['result']['cases']
checks = dict(all_three_children_exit0=len(queue['history']) == 3
    and all(r['exit_code'] == 0 for r in queue['history']),
    first_native_passed=queue['first_native_verification_exit_code'] == 0,
    full_native_restore_learning_and_cause_proof=native['passed'] and all(native['checks'].values()),
    exact_contract=sha(contract_path) == queue['contract_sha256'],
    all_frozen_sources_exact=all(sha(ROOT/n) == value for n,value in queue['source_sha256'].items()),
    complete1280_checkpoint=sha(checkpoint) == queue['checkpoint_sha256'] == evaluation['checkpoint_sha256'],
    all_counts_and_history_preserved=(info['successful_updates_total'],info['actor_updates_total'],
        info['active_branch_actor_updates'],info['feedback_branch_actor_updates'],info['post_split_actor_updates'],
        info['inherited_shared_feedback_updates'],info['consumed_optimizer_updates_total'],
        info['critic_warmup_remaining'],info['training_transitions_total']) == (1280,1100,44,722,418,304,1281,0,73924608),
    same_parent_and_diagnostic_provenance=sha(HERE/'feedback1264_population_completion001.json') == contract['parent_completion_sha256']
        and sha(HERE/'low_gate1264_diagnostic_review001.json') == contract['gate_diagnostic_review_sha256']
        and sha(HERE/'full_gate1264_basic_review001.json') == contract['full_gate_review_sha256'],
    basic_fixed_identity=evaluation['status'] == 'evaluated'
        and sha(evaluation_path) == queue['history'][-1]['evaluation_sha256'],
    comparisons_bound=all(sha(ROOT/row['path']) == row['sha256']
        for row in queue['history'][-1]['comparisons'].values()),
    no_long_training_heldout_or_hardware=not queue['subsequent_training_queued']
        and not queue['heldout_seed_used'] and not queue['hardware_motion_started'])
events = EventAccumulator(str(folder),size_guidance={'scalars':0})
events.Reload()
tags = set(events.Tags()['scalars'])
prefix = evaluation['tensorboard_namespace']
expected = {prefix+'/passed_cases':sum(c['passed'] for c in cases.values())}
for name,case in cases.items():
    for metric in ('survival_rate','height_mae_m','vx_mae_m_s','yaw_mae_rad_s',
                   'stand_drift_max_m','planar_speed_mean_m_s','closure_gap_max_m'):
        expected[prefix+'/'+name+'/'+metric] = case[metric]
event_checks = []
for tag,value in expected.items():
    points = events.Scalars(tag) if tag in tags else []
    event_checks.append(len(points) == 1 and points[0].step == 1280
        and math.isclose(points[0].value,value,rel_tol=2e-7,abs_tol=2e-8))
checks['basic_metrics_written_to_existing_RSL_events'] = all(event_checks)
if not all(checks.values()):
    print(json.dumps(checks,indent=2))
    raise SystemExit('Trial identity or event sealing failed; preserve all outputs')
result = dict(schema='flat-low-demand-gate1280-completion-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),passed=True,checks=checks,
    queue_sha256=sha(queue_path),contract_sha256=sha(contract_path),
    checkpoint_sha256=sha(checkpoint),native_verification_sha256=sha(native_path),
    evaluation_sha256=sha(evaluation_path),traces_sha256=sha(RUN/'deployment1280_basic001/traces.npz'),
    basic_cases_passed=sum(c['passed'] for c in cases.values()),requested_cases=len(cases),
    full_horizon_episodes=sum(c['full_horizon_episodes'] for c in cases.values()),
    requested_episodes=sum(c['requested_episodes'] for c in cases.values()),
    comparisons=queue['history'][-1]['comparisons'],
    event_file_sha256={p.name:sha(p) for p in folder.glob('events.out.tfevents.*')},
    all_release_cohorts_complete=False,behavior_qualified=False,heldout_seed_used=False,hardware_motion_started=False)
OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),basic_cases_passed=result['basic_cases_passed']),indent=2))

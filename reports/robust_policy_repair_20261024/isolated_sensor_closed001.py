"""Close the actual sensor-isolation pilot without promoting its failed policy."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT = BASE / 'reports/robust_policy_repair_20261024'
RUN = BASE / 'runs/scratch_standing_20261026/pilot012_isolated_sensor_requested001'
OUT = REPORT / 'isolated_sensor_closed001.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


assert not OUT.exists()
seal_path = REPORT / 'scratch_pilot012_closed_seal001.json'
seal = read(seal_path)
assert seal['all_checks_pass'] and seal['completed_updates'] == 128 and len(seal['checks']) == 20
assert not Path(f"/proc/{seal['pid']}").exists()
progress = read(RUN / 'progress.json')
review_path = REPORT / 'isolated_sensor_terminal_review001.json'
review = read(review_path)
middle, = [group for group in review['groups'] if group['height_m'] == .305]
assert middle['standing_pass_v2_count'] == 0
terminal = [row for row in progress['evaluation']['cases'] if row['height_m'] == .305]
assert len(terminal) == 7 and all(row['full_horizon'] for row in terminal)
queue_path = REPORT / 'requested_v2_evaluation_queue004'
queue = read(queue_path / 'receipt.json')
assert all(sha(ROOT / name) == value for name, value in queue['source_sha256'].items())
stamp = datetime.now(timezone.utc).isoformat()
result = dict(schema='isolated-sensor-pilot-closed-v1', observed_utc=stamp, run=str(RUN),
    seal=str(seal_path), seal_sha256=sha(seal_path),
    completed_fresh_updates=128, prior_closed_fresh_updates=3585, total_closed_fresh_updates=3713,
    final_checkpoint_sha256=seal['final_checkpoint_sha256'], source_count_at_seal=265,
    terminal_review=str(review_path), terminal_review_sha256=sha(review_path),
    terminal_middle_group=middle,
    terminal_tilt_peak_deg_range=[min(row['tilt_peak_deg'] for row in terminal), max(row['tilt_peak_deg'] for row in terminal)],
    common100_comparison=str(REPORT / 'isolated_sensor_first100_003.json'),
    final128_event_comparison=str(REPORT / 'isolated_sensor_final128_004.json'),
    nominal_learning_timing=str(REPORT / 'nominal_warmup_window_evidence002.json'),
    conclusion='The actual sensor-cache bug is fixed, but this128-update policy has not restored balance. No claim that longer cache-fix-only training cannot learn; old nominal009 improves mainly at129-160.',
    next_action='Finish cold serial queue004. Optional full-width sample-admission curriculum is prepared for a separate fresh512-update foundation experiment.',
    queue=dict(path=str(queue_path), pid=queue['pid'], start_ticks=queue['start_ticks'], status=queue['status']),
    long_training_started=False, ONNX_exported=False, qualified=False, hardware_ready=False,
    producer_sha256=sha(__file__))
write(OUT, result)
(REPORT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
ledger_path = REPORT / 'scratch_campaign001.json'
raw = ledger_path.read_bytes()
ledger = json.loads(raw)
assert ledger['active_pid'] == seal['pid'] and ledger['prior_closed_fresh_updates'] == 3585
ledger.update(status='readonly_evaluation_unqualified', active_pid=None, active_start_ticks=None,
    completed_updates_observed=128, active_completed_updates_observed=128,
    total_from_zero_completed=3713, total_from_zero_completed_observed=3713,
    prior_closed_fresh_updates=3713, prior_completed_fresh_updates=3713, prior_from_zero_completed=3713,
    last_completed_run=str(RUN), last_run_seal=str(seal_path), latest_live_read=str(OUT),
    latest_event_comparison=result['final128_event_comparison'], latest_user_acceptance_review=str(review_path),
    active_evaluation=str(queue_path), active_evaluation_pid=queue['pid'], active_evaluation_start_ticks=queue['start_ticks'],
    do_not_modify_live_src=False, do_not_modify_training_entrypoint=False,
    do_not_modify_frozen_evaluator_sources=True,
    prepared_next_fresh_experiment=dict(updates=512, warmup_updates=192, ramp_updates=128,
        change='PPO/normalizer sample admission only; same full physical requested widths',
        from_zero=True, checkpoint_or_teacher_reads_forbidden=True, approved_for_launch=False,
        prerequisite='Queue004 actually exits and fixed results are reviewed; actual integrated preflight and committed sources'),
    latest_observation=result['conclusion'], next_action=result['next_action'],
    observed_utc=stamp, last_observed_utc=stamp)
assert ledger_path.read_bytes() == raw
temporary = ledger_path.with_suffix('.isolatedclosed001.tmp')
write(temporary, ledger)
temporary.replace(ledger_path)
print(json.dumps(dict(closed_fresh_updates=3713, terminal_middle_v2_pass=0,
    tilt_peak_deg_range=result['terminal_tilt_peak_deg_range'], queue_status=queue['status'])))

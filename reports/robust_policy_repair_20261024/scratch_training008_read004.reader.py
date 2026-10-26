"""Authenticate the original live008 handle and record completed event windows."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUN = ROOT / 'runs/scratch_standing_20261026/pilot008_leg_target_rate020'
OUTPUT = REPORTS / 'scratch_training008_read004.json'
assert not OUTPUT.exists()
identity_path = REPORTS / 'scratch_pilot008_launch_identity001.json'
identity = json.loads(identity_path.read_text())
pid = identity['pid']
stat = Path(f'/proc/{pid}/stat').read_text()
fields = stat[stat.rfind(')') + 2:].split()
assert int(fields[19]) == identity['start_ticks'] and fields[0] != 'Z'
actual_command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
assert [x.decode() for x in actual_command if x] == identity['command']
gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                     capture_output=True, text=True, check=True)
assert str(pid) in gpu.stdout.splitlines()
progress = json.loads((RUN / 'progress.json').read_text())
assert progress['status'] == 'training' and progress['checkpoint_reads'] == []
assert progress['requested_updates'] == 512 and progress['completed_updates'] >= 192
sources = json.loads((RUN / 'source_hashes.json').read_text())
assert len(sources) == 263
for name, expected in sources.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256((RUN / 'source_snapshot' / name).read_bytes()).hexdigest() == expected, name
comparison_path = REPORTS / 'scratch_training007_008_window032_read006.json'
comparison = json.loads(comparison_path.read_text())
payment_path = REPORTS / 'scratch_leg_target_payment008_read001.json'
payment = json.loads(payment_path.read_text())
for source in comparison['sources'].values():
    with Path(source['path']).open('rb') as stream:
        raw = stream.read(source['observed_prefix_bytes'])
    assert hashlib.sha256(raw).hexdigest() == source['observed_prefix_sha256']
prefix = payment['event_prefix']
with Path(prefix['path']).open('rb') as stream:
    raw = stream.read(prefix['bytes'])
assert hashlib.sha256(raw).hexdigest() == prefix['sha256']
assert payment['all_payment_checks_pass'] and payment['all_live_and_frozen_sources_match']
now = datetime.now(timezone.utc).isoformat()
report = dict(
    schema='scratch-standing-live-read-v1', observed_utc=now, run=str(RUN), pid=pid,
    start_ticks=identity['start_ticks'], original_process=True, original_PID_in_GPU_compute=True,
    status=progress['status'], completed_updates=progress['completed_updates'], requested_updates=512,
    source_count=len(sources), all_live_and_frozen_sources_match=True, training_checkpoint_reads=0,
    prior_completed_fresh_updates=1921, total_from_zero_completed_observed=1921 + progress['completed_updates'],
    event_comparison=str(comparison_path),
    event_comparison_sha256=hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
    last_complete_window=comparison['windows'][-1], payment_observation=str(payment_path),
    payment_observation_sha256=hashlib.sha256(payment_path.read_bytes()).hexdigest(),
    all71_logged_payment_checks_pass=payment['complete_updates'] == 71 and payment['all_payment_checks_pass'],
    reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    new_PPO_by_reader=0, own_native_evaluation_finished=False, behavior_qualification=False, hardware_ready=False,
)
feedback_path = REPORTS / 'scratch_trajectory_feedback007_read001.json'
feedback = json.loads(feedback_path.read_text())
torque_path = REPORTS / 'scratch_feedback_torque007_validation001.json'
torque = json.loads(torque_path.read_text())
assert len(feedback['rows']) == 68 and torque['all_actual_torque_derivative_checks_pass']
assert torque['diagnostic_sha256'] == hashlib.sha256(feedback_path.read_bytes()).hexdigest()
report.update(feedback_response=str(feedback_path), actual_V5_derivative_verification=str(torque_path),
    feedback_response_sha256=hashlib.sha256(feedback_path.read_bytes()).hexdigest(),
    actual_V5_derivative_verification_sha256=hashlib.sha256(torque_path.read_bytes()).hexdigest(),
    positive_leg_local_gain_states=sum(max(row['leg_command_gain_symmetric_eigenvalues']) > 0. for row in feedback['rows']),
    checked_feedback_states=68)
probe_path = REPORTS / 'scratch_leg_encoder_preflight001.json'
probe = json.loads(probe_path.read_text())
assert probe['all_checks_pass'] and not probe['native_validation'] and len(probe['checks']) == 14
for name, expected in probe['source_sha256'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
report.update(prepared_leg_encoder_probe=str(probe_path),
    prepared_leg_encoder_probe_sha256=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
    leg_encoder_probe_native_run_started=False)
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
campaign_path = REPORTS / 'scratch_campaign001.json'
campaign = json.loads(campaign_path.read_text())
assert campaign['active_pid'] == pid and campaign['active_start_ticks'] == identity['start_ticks']
campaign.update(
    completed_updates_observed=progress['completed_updates'],
    total_from_zero_completed=1921 + progress['completed_updates'],
    total_from_zero_completed_observed=1921 + progress['completed_updates'],
    latest_live_read=str(OUTPUT), last_observed_utc=now, latest_event_comparison=str(comparison_path),
    latest_leg_target_payment=str(payment_path), latest_native_launch=str(identity_path),
    next_action='Continue original fresh008 PID709399/start31429889 and263 frozen sources to512 plus its own '
                'strict21-case8s evaluation. Early32-update windows are mixed and do not establish standing improvement. '
                'Actual target-rate cost payment is verified, not behavior. After the original process and GPU exit, seal '
                'and inspect every1ms support sample. No resume, hold adoption, hardware qualification or new native job.',
)
campaign.update(latest_feedback_response=str(feedback_path),
    latest_actual_V5_feedback_verification=str(torque_path),
    latest_feedback_timeline=str(REPORTS / 'scratch_feedback_gain007_timeline001.json'),
    latest_feedback_plot=str(REPORTS / 'scratch_feedback_gain007_plot001.png'),
    next_action='Continue original fresh008 PID709399/start31429889 and263 sources unchanged to512 plus own strict21-case8s evaluation. '
                'Through192 updates the stochastic windows are mixed. Completed007 has positive local leg command-velocity gain '
                'at65of68 checked native states, independently verified through V5; this is a next-hypothesis diagnostic, not '
                'the proven contact-loss cause. Inspect whether008 improves both strict support and feedback before selecting '
                'any further fresh-only training change. No restart, resume, parallel native job or hardware qualification.')
campaign.update(prepared_leg_encoder_probe=str(probe_path),
    prepared_leg_encoder_native_validation=False,
    next_action='Continue original fresh008 PID709399/start31429889 and263 sources unchanged to512 plus own21-case8s evaluation. '
                'Window161-192 has better stochastic support and leg RMS than007 but not100% and is not a fixed-case pass. '
                'After terminal process/GPU exit seal first. If strict support still fails and direct leg-velocity feedback '
                'needs separation, run only one declared leg-speed response0 diagnostic after validating its native baseline '
                'and initial/untouched states. If support passes, prioritize prepared30s/seed evaluation instead. '
                'All subsequent training is fresh-only; no checkpoint continuation, hold adoption or hardware qualification.')
temporary = campaign_path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(campaign, indent=2, allow_nan=False) + '\n')
temporary.replace(campaign_path)
print(json.dumps({k: v for k, v in report.items() if k != 'last_complete_window'}))

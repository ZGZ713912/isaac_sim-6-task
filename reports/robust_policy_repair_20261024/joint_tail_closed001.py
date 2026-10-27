"""Record the completed fresh128 pilot and its unsuccessful reward-only result."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT=BASE/'reports/robust_policy_repair_20261024'
RUN=BASE/'runs/scratch_standing_20261026/pilot011_joint_tail_requested001'
OUT=REPORT/'joint_tail_closed001.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


assert not OUT.exists()
seal_path=REPORT/'scratch_pilot011_closed_seal001.json'
seal=read(seal_path)
assert seal['all_checks_pass'] and seal['completed_updates']==128 and len(seal['checks'])==20
assert not Path(f"/proc/{seal['pid']}").exists()
progress=read(RUN/'progress.json')
mid=[row for row in progress['evaluation']['cases'] if row['height_m']==.305]
assert len(mid)==7 and all(row['full_horizon'] for row in mid)
comparison=REPORT/'joint_tail_first100_003.json'
events=read(comparison)
queue_path=REPORT/'requested_v2_evaluation_queue003'
queue=read(queue_path/'receipt.json')
assert all(sha(ROOT/name)==value for name,value in queue['source_sha256'].items())
stamp=datetime.now(timezone.utc).isoformat()
result=dict(schema='joint-tail-reward-only-closed-v1',observed_utc=stamp,run=str(RUN),
    final_checkpoint_sha256=seal['final_checkpoint_sha256'],completed_fresh_updates=128,
    prior_closed_fresh_updates=3457,total_closed_fresh_updates=3585,
    seal=str(seal_path),seal_sha256=sha(seal_path),
    first100_comparison=str(comparison),first100_comparison_sha256=sha(comparison),
    first100_windows=events['windows'],
    middle_height_terminal_cases=mid,
    terminal_standing_pass_count=sum(row['standing_pass'] for row in mid),
    terminal_tilt_peak_deg_range=[min(row['tilt_peak_deg'] for row in mid),max(row['tilt_peak_deg'] for row in mid)],
    contact_label='Terminal trainer stores historical raw zero-loss flags; use queue003 v2 review for retained losses over2ms.',
    queue=dict(path=str(queue_path),pid=queue['pid'],start_ticks=queue['start_ticks'],status=queue['status']),
    conclusion='Reward-only short pilot has not restored balance; first100 metrics and cold terminal standing remain negative. This is not a long-training candidate.',
    next_action='Complete frozen queue003; fix reproduced partial-reset sensor-clock leak, validate native behavior, and start a separate fresh controlled run. Requested final widths remain unchanged.',
    long_training_started=False,ONNX_exported=False,qualified=False,hardware_ready=False,
    producer_sha256=sha(__file__))
write(OUT,result)
(REPORT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
ledger_path=REPORT/'scratch_campaign001.json';raw=ledger_path.read_bytes();ledger=json.loads(raw)
assert ledger['active_pid']==seal['pid'] and ledger['prior_closed_fresh_updates']==3457
ledger.update(status='readonly_evaluation_unqualified',active_pid=None,active_start_ticks=None,
    active_completed_updates_observed=128,completed_updates_observed=128,
    total_from_zero_completed=3585,total_from_zero_completed_observed=3585,
    prior_closed_fresh_updates=3585,prior_completed_fresh_updates=3585,prior_from_zero_completed=3585,
    last_completed_run=str(RUN),last_run_seal=str(seal_path),latest_live_read=str(OUT),
    latest_event_comparison=str(comparison),active_evaluation=str(queue_path),
    active_evaluation_pid=queue['pid'],active_evaluation_start_ticks=queue['start_ticks'],
    latest_observation=result['conclusion'],next_action=result['next_action'],
    observed_utc=stamp,last_observed_utc=stamp)
assert ledger_path.read_bytes()==raw
temporary=ledger_path.with_suffix('.jointclosed001.tmp');write(temporary,ledger);temporary.replace(ledger_path)
print(json.dumps(dict(closed_fresh_updates=3585,terminal_standing_pass_count=result['terminal_standing_pass_count'],
    terminal_tilt_peak_deg_range=result['terminal_tilt_peak_deg_range'],queue_status=queue['status'])))

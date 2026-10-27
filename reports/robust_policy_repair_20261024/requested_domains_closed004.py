"""Record the actual completed fresh learner separately from its native queue."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

WORK=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT=BASE/'reports/robust_policy_repair_20261024'
RUN=BASE/'runs/scratch_standing_20261026/pilot010_requested_domains_retry001'
QUEUE=REPORT/'requested_v2_evaluation_queue002'
OUT=REPORT/'requested_domains_closed004.json'
assert not OUT.exists()

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

seal_file=REPORT/'scratch_pilot010_closed_seal001.json'
seal=read(seal_file)
assert seal['all_checks_pass'] and all(seal['checks'].values()) and seal['completed_updates']==512
assert not Path('/proc/280546').exists()
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert '280546' not in gpu.splitlines()
progress=read(RUN/'progress.json')
assert progress['status']=='completed_unqualified' and progress['checkpoint_reads']==[]
assert sha(RUN/'model_final.pt')==seal['final_checkpoint_sha256']==progress['last_checkpoint']['sha256']
manifest=read(RUN/'source_hashes.json')
assert len(manifest)==265 and all(sha(WORK/name)==value for name,value in manifest.items())
queue_raw=(QUEUE/'receipt.json').read_bytes()
queue=json.loads(queue_raw)
assert queue['status']=='serial_native_evaluation'
assert int(Path(f"/proc/{queue['pid']}/stat").read_text().rsplit(')',1)[1].split()[19])==queue['start_ticks']
assert all(sha(WORK/name)==value for name,value in queue['source_sha256'].items())
terminal=read(REPORT/'requested_010_terminal_review001.json')
nominal=read(QUEUE/'nominal.review.json')
mid=lambda value:next(row for row in value['groups'] if row['height_m']==.305)
assert mid(terminal)['standing_pass_v2_count']==mid(nominal)['standing_pass_v2_count']==0
stamp=datetime.now(timezone.utc).isoformat()
snapshot=REPORT/'requested_domains_closed004.queue.json'
snapshot.write_bytes(queue_raw)
result=dict(schema='requested-domain-fresh-pilot-completed-native-comparison-in-progress-v1',
    observed_utc=stamp,run=str(RUN),completed_fresh_updates=512,
    total_closed_fresh_updates_before_this_run=2945,total_closed_fresh_updates=3457,
    actual_original_process_and_GPU_exited=True,checkpoint=str(RUN/'model_final.pt'),
    checkpoint_sha256=seal['final_checkpoint_sha256'],integrity_seal=str(seal_file),integrity_seal_sha256=sha(seal_file),
    all265_original_sources_unchanged_at_close=True,
    terminal_randomized_middle_height=mid(terminal),heldout_nominal_middle_height=mid(nominal),
    terminal_review_sha256=sha(REPORT/'requested_010_terminal_review001.json'),
    heldout_nominal_review_sha256=sha(QUEUE/'nominal.review.json'),
    terminal_all_heights=terminal['groups'],nominal_all_heights=nominal['groups'],
    serial_queue=dict(path=str(QUEUE),pid=queue['pid'],start_ticks=queue['start_ticks'],
        active_child=queue['active_child'],receipt_snapshot_sha256=sha(snapshot),
        all12_frozen_evaluator_sources_unchanged=True),
    conclusion='This trained actor also fails with randomization removed, unlike009 nominal. The regression includes nominal closed-loop balance; native ablations still determine immediate group effects, not a unique training cause.',
    next_action='Complete same-final-checkpoint requested/paired ablations, then run a new fresh experiment targeting the demonstrated feedback/reward or sensor-correlation problem. Preserve all permanent capability gates.',
    hardware_ready=False,behavior_qualified=False,long_training_started=False,ONNX_exported=False,RMCS_deployed=False,
    producer_sha256=sha(Path(__file__)))
write(OUT,result)
(REPORT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
ledger_file=REPORT/'scratch_campaign001.json'
raw=ledger_file.read_bytes()
ledger=json.loads(raw)
assert ledger['active_pid']==280546 and ledger['active_start_ticks']==1391023
assert ledger['prior_closed_fresh_updates']==2945
ledger.update(status='evaluating_unqualified',active_run=None,active_pid=None,active_start_ticks=None,
    completed_updates_observed=512,active_completed_updates_observed=0,
    prior_closed_fresh_updates=3457,prior_from_zero_completed=3457,prior_completed_fresh_updates=3457,
    total_from_zero_completed=3457,total_from_zero_completed_observed=3457,
    last_completed_run=str(RUN),last_run_seal=str(seal_file),latest_live_read=str(OUT),
    latest_user_acceptance_review=str(REPORT/'requested_010_terminal_review001.json'),
    latest_event_comparison=str(REPORT/'requested_domains_first400003.json'),
    serial_evaluation_queue=str(QUEUE),serial_evaluation_queue_pid=queue['pid'],serial_evaluation_queue_start_ticks=queue['start_ticks'],
    do_not_modify_live_src=False,do_not_modify_frozen_evaluator_sources=True,
    latest_sensor_correlation_audit=str(REPORT/'encoder_feedback_chain_audit001/result.json'),
    prepared_three_height_initialization=str(REPORT/'three_height_static_prior002/result.json'),
    latest_full_suite_preflight=str(REPORT/'full_suite_preflight005_current_requested'),
    full_suite_native_verified=False,qualified=False,hardware_ready=False,
    next_action=result['next_action'],observed_utc=stamp,last_observed_utc=stamp)
assert ledger_file.read_bytes()==raw
temporary=ledger_file.with_suffix('.closed004.tmp')
write(temporary,ledger)
temporary.replace(ledger_file)
print(json.dumps(dict(closed_fresh_updates=512,total_closed=3457,nominal_mid_pass=0,
    nominal_mid_max_XY=mid(nominal)['max_xy_excursion_m'],queue_child=queue['active_child']['label'])))

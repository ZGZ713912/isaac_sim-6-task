"""Record one immutable live observation without changing the active experiment."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

WORK=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT=BASE/'reports/robust_policy_repair_20261024'
RUN=BASE/'runs/scratch_standing_20261026/pilot011_joint_tail_requested001'
OUT=REPORT/'joint_tail_live001.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


assert not OUT.exists()
progress=read(RUN/'progress.json')
assert progress['status']=='training' and progress['requested_updates']==128
pid=progress['pid']
assert int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])==progress['start_ticks']
assert progress['checkpoint_reads']==[]
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert str(pid) in gpu.splitlines()
manifest=read(RUN/'source_hashes.json')
assert len(manifest)==265 and all(sha(WORK/name)==value and sha(RUN/'source_snapshot'/name)==value for name,value in manifest.items())
first=read(REPORT/'joint_tail_first_verification001.json')
assert first['all_checks_pass'] and len(first['checks'])==32 and first['RSL_config_scalars_verified']==54
comparison=read(REPORT/'joint_tail_first64_002.json')
assert len(comparison['windows'])==2
queue_path=REPORT/'requested_v2_evaluation_queue003'
queue=read(queue_path/'receipt.json')
assert queue['status']=='waiting_for_exact_training_process_exit' and queue['active_child'] is None
assert all(sha(WORK/name)==value for name,value in queue['source_sha256'].items())
assert int(Path(f"/proc/{queue['pid']}/stat").read_text().rsplit(')',1)[1].split()[19])==queue['start_ticks']
stamp=datetime.now(timezone.utc).isoformat()
result=dict(schema='joint-tail-reward-only-live-observation-v1',observed_utc=stamp,
    run=str(RUN),pid=pid,start_ticks=progress['start_ticks'],completed_updates=progress['completed_updates'],
    requested_updates=128,prior_closed_fresh_updates=3457,total_observed_fresh_updates=3457+progress['completed_updates'],
    actual_GPU_process_present=True,all265_live_and_archived_sources_unchanged=True,
    committed_source=progress['committed_source'],first_verification_sha256=sha(REPORT/'joint_tail_first_verification001.json'),
    first64_comparison_sha256=sha(REPORT/'joint_tail_first64_002.json'),first64_windows=comparison['windows'],
    queue=dict(path=str(queue_path),pid=queue['pid'],start_ticks=queue['start_ticks'],
        frozen_source_count=len(queue['source_sha256']),status=queue['status']),
    conclusion='First64 exploratory updates do not show a standing-speed improvement. Finish the fixed128 budget, preserve the same weights for cold nominal/requested evaluation, and do not call this a successful reward repair.',
    next_candidate_if_failed='Fresh nominal-sample learning warmup followed by gradual admission of the already sampled full-width randomized lanes, with matching normalization and PPO masks; retain final requested domains and all permanent capabilities. Not implemented or adopted.',
    full_goal_remaining=['three-height robust standing','all initial falls and stable followup','actual moving fall recovery',
        'low/high forward/reverse straight tracking','four-way arcs and transitions','both80/100rpm spins',
        'standing/moving/spinning height changes','high-speed braking','smooth leg/wheel coordination','held-out domains/seeds',
        'near-full integrated qualification','fresh long training','ONNX parity','RMCS offline integration and deployment contract'],
    qualification=False,hardware_ready=False,long_training_started=False,ONNX_exported=False,RMCS_deployed=False,
    producer_sha256=sha(Path(__file__)))
write(OUT,result)
(REPORT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
ledger_file=REPORT/'scratch_campaign001.json'
raw=ledger_file.read_bytes();ledger=json.loads(raw)
assert ledger['active_pid']==pid and ledger['prior_closed_fresh_updates']==3457
ledger.update(completed_updates_observed=progress['completed_updates'],active_completed_updates_observed=progress['completed_updates'],
    total_from_zero_completed=result['total_observed_fresh_updates'],total_from_zero_completed_observed=result['total_observed_fresh_updates'],
    latest_live_read=str(OUT),latest_event_comparison=str(REPORT/'joint_tail_first64_002.json'),
    latest_observation=result['conclusion'],observed_utc=stamp,last_observed_utc=stamp)
assert ledger_file.read_bytes()==raw
temporary=ledger_file.with_suffix('.jointlive001.tmp');write(temporary,ledger);temporary.replace(ledger_file)
print(json.dumps(dict(completed_updates=progress['completed_updates'],total_observed=result['total_observed_fresh_updates'],
    sources_unchanged=len(manifest),queue_waiting=True)))

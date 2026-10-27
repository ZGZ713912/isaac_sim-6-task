"""Immutable live observation; no source, learner, model or optimizer mutation."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

WORK=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT=BASE/'reports/robust_policy_repair_20261024'
RUN=BASE/'runs/scratch_standing_20261026/pilot012_isolated_sensor_requested001'
OUT=REPORT/'isolated_sensor_live001.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


assert not OUT.exists()
progress=read(RUN/'progress.json');pid=progress['pid']
assert progress['status']=='training' and progress['completed_updates']>=32 and progress['requested_updates']==128
assert int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])==progress['start_ticks']
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert str(pid) in gpu.splitlines() and progress['checkpoint_reads']==[]
manifest=read(RUN/'source_hashes.json')
assert len(manifest)==265 and all(sha(WORK/name)==digest and sha(RUN/'source_snapshot'/name)==digest for name,digest in manifest.items())
queue_path=REPORT/'requested_v2_evaluation_queue004';queue=read(queue_path/'receipt.json')
assert queue['status']=='waiting_for_exact_training_process_exit' and queue['active_child'] is None
assert int(Path(f"/proc/{queue['pid']}/stat").read_text().rsplit(')',1)[1].split()[19])==queue['start_ticks']
assert all(sha(WORK/name)==digest for name,digest in queue['source_sha256'].items())
comparison=REPORT/'isolated_sensor_first32_001.json';windows=read(comparison)['windows']
assert len(windows)==1 and windows[0]['PPO_update_numbers_inclusive']==[1,32]
stamp=datetime.now(timezone.utc).isoformat()
result=dict(schema='isolated-sensor-live-observation-v1',observed_utc=stamp,run=str(RUN),pid=pid,start_ticks=progress['start_ticks'],
    completed_updates=progress['completed_updates'],requested_updates=128,prior_closed_fresh_updates=3585,
    total_observed_fresh_updates=3585+progress['completed_updates'],all265_live_and_archived_sources_unchanged=True,
    committed_source=progress['committed_source'],first32_event_comparison=str(comparison),
    first32_event_comparison_sha256=sha(comparison),first32_windows=windows,
    first_verification=str(REPORT/'isolated_sensor_first_verification001.json'),
    native_sensor_reset_proof=str(REPORT/'isolated_sensor_native_verification001.json'),
    queue=dict(path=str(queue_path),pid=queue['pid'],start_ticks=queue['start_ticks'],status=queue['status'],frozen_source_count=len(queue['source_sha256'])),
    conclusion='The sensor-cache implementation leak is fixed; common first32 training speed/tilt metrics still do not show a balance improvement. Complete the bounded run and assess no-autoreset native behavior before any robustness claim.',
    next_if_failed='Fresh nominal-population PPO warmup followed by gradual inclusion of full-width randomized lanes, with the same native PPO and normalizer masks; CPU candidate only, not used by012.',
    full_goal_remaining=['three-height robust standing','all initial falls with stable followup','actual moving fall recovery and resume',
        'low/high forward/reverse straight tracking','four-way arcs and transitions','both80/100rpm spins',
        'standing/moving/spinning height changes','high-speed braking','smooth leg/wheel coordination','held-out domains/seeds',
        'near-full integrated qualification','fresh long training','ONNX parity','RMCS deployment contract/integration'],
    long_training_started=False,ONNX_exported=False,qualified=False,hardware_ready=False,producer_sha256=sha(__file__))
write(OUT,result);(REPORT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
ledger_path=REPORT/'scratch_campaign001.json';raw=ledger_path.read_bytes();ledger=json.loads(raw)
assert ledger['active_pid']==pid and ledger['prior_closed_fresh_updates']==3585
ledger.update(completed_updates_observed=progress['completed_updates'],active_completed_updates_observed=progress['completed_updates'],
    total_from_zero_completed=result['total_observed_fresh_updates'],total_from_zero_completed_observed=result['total_observed_fresh_updates'],
    latest_live_read=str(OUT),latest_event_comparison=str(comparison),latest_observation=result['conclusion'],
    serial_evaluation_queue=str(queue_path),serial_evaluation_queue_pid=queue['pid'],serial_evaluation_queue_start_ticks=queue['start_ticks'],
    do_not_modify_frozen_evaluator_sources=True,current_generation_middle_height_qualification=False,
    observed_utc=stamp,last_observed_utc=stamp)
assert ledger_path.read_bytes()==raw
temporary=ledger_path.with_suffix('.isolatedlive001.tmp');write(temporary,ledger);temporary.replace(ledger_path)
print(json.dumps(dict(completed_updates=progress['completed_updates'],total_observed_fresh_updates=result['total_observed_fresh_updates'],
    original_GPU_process_live=True,frozen_source_count=265,queue_waiting=True)))

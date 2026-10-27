"""Authenticate the live trainer and its independent serial evaluation queue."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from xml.etree import ElementTree

WORK=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT=BASE/'reports/robust_policy_repair_20261024'
RUN=BASE/'runs/scratch_standing_20261026/pilot010_requested_domains_retry001'
QUEUE=REPORT/'requested_v2_evaluation_queue002'
NAME='requested_domains_iteration_read003'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def live(pid,ticks,required):
    proc=Path('/proc')/str(pid)
    assert int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])==ticks
    command=(proc/'cmdline').read_bytes()
    assert all(value.encode() in command for value in required)
    return command.replace(b'\0',b' ').decode()

output=REPORT/(NAME+'.json')
assert not output.exists()
raw=(RUN/'progress.json').read_bytes()
progress=json.loads(raw)
trainer_command=live(280546,1391023,['scripts/train_scratch_standing.py',str(RUN)])
assert progress['pid']==280546 and progress['start_ticks']==1391023
assert progress['status']=='training' and progress['checkpoint_reads']==[]
hashes=read(RUN/'source_hashes.json')
assert len(hashes)==265
assert all(sha(WORK/name)==value==sha(RUN/'source_snapshot'/name) for name,value in hashes.items())
queue_raw=(QUEUE/'receipt.json').read_bytes()
queue=json.loads(queue_raw)
queue_command=live(queue['pid'],queue['start_ticks'],['scripts/queue_requested_domain_evaluation.py',str(QUEUE)])
assert queue['status']=='waiting_for_exact_training_process_exit'
assert queue['training_identity']==dict(pid=280546,start_ticks=1391023)
assert queue['new_PPO']==0 and queue['active_child'] is None
assert all(sha(WORK/name)==value==sha(QUEUE/'source_snapshot'/name)
           for name,value in queue['source_sha256'].items())
failed=REPORT/'requested_v2_evaluation_queue001/receipt.json'
assert read(failed)['status']=='failed' and read(failed)['profiles']==[]
tests_file=REPORT/'requested_evaluation_queue_CPU004.xml'
tests=ElementTree.parse(tests_file).getroot().find('testsuite')
assert tests.attrib['tests']=='8' and tests.attrib['failures']=='0' and tests.attrib['errors']=='0'
sensor=REPORT/'sensor_functional_response002'
verification=read(sensor/'verification.json')
assert verification['all_functional_deltas_recomputed'] and verification['all_scalar_values_exact_within_event_float32_tolerance']
moment=datetime.now(timezone.utc).isoformat()
snapshot=REPORT/(NAME+'.progress.json')
snapshot.write_bytes(raw)
queue_snapshot=REPORT/(NAME+'.queue.json')
queue_snapshot.write_bytes(queue_raw)
report=dict(schema='requested-domain-v2-live-training-and-serial-queue-v1',observed_utc=moment,
    training=dict(run=str(RUN),pid=280546,start_ticks=1391023,command=trainer_command,
        completed_updates=progress['completed_updates'],requested_updates=512,checkpoint_reads=[],
        frozen_training_commit=progress['committed_source']['commit'],all265_sources_unchanged=True,
        progress_snapshot_sha256=sha(snapshot)),
    serial_evaluation_queue=dict(folder=str(QUEUE),pid=queue['pid'],start_ticks=queue['start_ticks'],
        command=queue_command,tool_session_id=30051,source_commit='0092cc6b',
        receipt_snapshot_sha256=sha(queue_snapshot),all12_helper_contract_sources_unchanged=True,
        current_child=None,new_PPO=0,training_restarts_allowed=False),
    failed_queue_preserved=dict(path=str(failed),sha256=sha(failed),native_steps=0,new_PPO=0,
        cause='Two evaluator helpers absent from managed worktree; restored exact authenticated prior diagnostic bytes'),
    queue_CPU_tests=8,queue_CPU_test_evidence_sha256=sha(tests_file),
    CPU_sensor_response_result=str(sensor/'result.json'),CPU_sensor_response_result_sha256=sha(sensor/'result.json'),
    CPU_sensor_response_verification=str(sensor/'verification.json'),
    interpretation='One-frame CPU feedback sensitivity only; weaker true pitch/gyro response needs native closed-loop confirmation.',
    full_goal_remaining=True,long_training_started=False,ONNX_exported=False,RMCS_deployed=False,
    native_behavior_qualified=False,hardware_ready=False,verifier_sha256=sha(Path(__file__)))
write(output,report)
shutil.copyfile(Path(__file__),REPORT/(NAME+'.py'))
ledger_file=REPORT/'scratch_campaign001.json'
ledger_raw=ledger_file.read_bytes()
ledger=json.loads(ledger_raw)
assert ledger['active_run']==str(RUN) and ledger['active_pid']==280546
assert ledger['prior_closed_fresh_updates']==2945
ledger.update(completed_updates_observed=progress['completed_updates'],
    active_completed_updates_observed=progress['completed_updates'],
    total_from_zero_completed_observed=2945+progress['completed_updates'],
    latest_live_read=str(output),last_observed_utc=moment,observed_utc=moment,
    latest_sensor_response=str(sensor/'result.json'),latest_sensor_response_verification=str(sensor/'verification.json'),
    serial_evaluation_queue=str(QUEUE),serial_evaluation_queue_pid=queue['pid'],
    serial_evaluation_queue_start_ticks=queue['start_ticks'],qualified=False,hardware_ready=False)
assert ledger_file.read_bytes()==ledger_raw
temporary=ledger_file.with_suffix('.read003.tmp')
write(temporary,ledger)
temporary.replace(ledger_file)
print(json.dumps(dict(completed_updates=progress['completed_updates'],queue_pid=queue['pid'],
    frozen_training_sources_verified=265,queue_sources_verified=12,queue_CPU_tests=8)))

"""Persist the actual live learner and queue identity with their first window."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT = BASE / 'reports/robust_policy_repair_20261024'
RUN = BASE / 'runs/scratch_standing_20261026/pilot013_gradual_admission_requested001'
OUT = REPORT / 'domain_admission_live001.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not OUT.exists()
progress = read(RUN / 'progress.json')
assert progress['status'] == 'training' and progress['completed_updates'] >= 16 and progress['requested_updates'] == 512
assert progress['pid'] == 599642 and progress['start_ticks'] == 3112307 and progress['checkpoint_reads'] == []
assert int(Path('/proc/599642/stat').read_text().rsplit(')', 1)[1].split()[19]) == 3112307
gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
assert '599642' in gpu.splitlines()
manifest = read(RUN / 'source_hashes.json')
assert len(manifest) == 266 and all(sha(ROOT / name) == value and sha(RUN / 'source_snapshot' / name) == value for name, value in manifest.items())
queue_path = REPORT / 'requested_v2_evaluation_queue005'
queue = read(queue_path / 'receipt.json')
assert queue['status'] == 'waiting_for_exact_training_process_exit' and queue['active_child'] is None
assert queue['training_identity'] == dict(pid=599642, start_ticks=3112307)
assert int(Path(f"/proc/{queue['pid']}/stat").read_text().rsplit(')', 1)[1].split()[19]) == queue['start_ticks']
assert all(sha(ROOT / name) == value for name, value in queue['source_sha256'].items())
comparison = REPORT / 'domain_admission_first16_002.json'
observed = read(comparison)
assert observed['RSL_schedule_matches_every_complete_update'] and len(observed['windows']) == 1
assert observed['windows'][0]['fresh_PPO_update_numbers_inclusive'] == [1, 16]
stamp = datetime.now(timezone.utc).isoformat()
result = dict(schema='domain-admission-live-observation-v1', observed_utc=stamp, run=str(RUN),
    pid=599642, start_ticks=3112307, exec_session_id=1230,
    completed_updates=progress['completed_updates'], requested_updates=512,
    prior_closed_fresh_updates=3713, total_observed_fresh_updates=3713 + progress['completed_updates'],
    all266_training_sources_unchanged=True, training_source_commit=progress['committed_source']['commit'],
    first_real_native_PPO_verification=str(REPORT / 'domain_admission_first_verification001.json'),
    first16_event_comparison=str(comparison), first16_event_comparison_sha256=sha(comparison),
    first16_windows=observed['windows'],
    queue=dict(path=str(queue_path), pid=queue['pid'], start_ticks=queue['start_ticks'], exec_session_id=46155,
        status=queue['status'], all12_evaluation_sources_unchanged=True),
    conclusion='The optional admission and shared PPO/normalizer population are verified in actual native learning. First16 stochastic nominal/randomized metrics are not stable standing; no claimed robustness improvement.',
    remaining='Finish warmup, ramp and full-population phases; cold tests, all permanent capabilities, near-full qualification, fresh long training, ONNX parity and RMCS deployment integration.',
    long_training_started=False, ONNX_exported=False, qualified=False, hardware_ready=False,
    producer_sha256=sha(__file__))
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
(REPORT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
ledger_path = REPORT / 'scratch_campaign001.json'
raw = ledger_path.read_bytes()
ledger = json.loads(raw)
assert ledger['active_pid'] == 599642 and ledger['prior_closed_fresh_updates'] == 3713
ledger.update(completed_updates_observed=progress['completed_updates'], active_completed_updates_observed=progress['completed_updates'],
    total_from_zero_completed=result['total_observed_fresh_updates'], total_from_zero_completed_observed=result['total_observed_fresh_updates'],
    latest_live_read=str(OUT), latest_event_comparison=str(comparison),
    latest_observation=result['conclusion'], do_not_modify_live_src=True, do_not_modify_training_entrypoint=True,
    do_not_modify_domain_admission_helper=True, do_not_modify_frozen_evaluator_sources=True,
    observed_utc=stamp, last_observed_utc=stamp)
assert ledger_path.read_bytes() == raw
temporary = ledger_path.with_suffix('.admissionlive001.tmp')
temporary.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
temporary.replace(ledger_path)
print(json.dumps(dict(completed_updates=progress['completed_updates'], closed_fresh_updates=3713,
    all266_training_sources_unchanged=True, queue_waiting=True, queue_pid=queue['pid'])))

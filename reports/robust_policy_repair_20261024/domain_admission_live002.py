"""Record actual013 progress, immutable96-update windows and frozen sources."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT = BASE / 'reports/robust_policy_repair_20261024'
RUN = BASE / 'runs/scratch_standing_20261026/pilot013_gradual_admission_requested001'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = REPORT / 'domain_admission_live002.json'
    if output.exists():
        raise ValueError('Preserve this live observation')
    progress = read(RUN / 'progress.json')
    if (progress['pid'] != 599642 or progress['start_ticks'] != 3112307
            or progress['status'] != 'training' or progress['completed_updates'] < 96
            or progress['requested_updates'] != 512 or progress['checkpoint_reads']):
        raise ValueError('Expected the same fresh original learner')
    if int(Path('/proc/599642/stat').read_text().rsplit(')', 1)[1].split()[19]) != 3112307:
        raise ValueError('Original learner exited')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if '599642' not in gpu.splitlines():
        raise ValueError('Original learner absent from actual GPU processes')
    manifest = read(RUN / 'source_hashes.json')
    for name, expected in manifest.items():
        if sha(ROOT / name) != expected or sha(RUN / 'source_snapshot' / name) != expected:
            raise ValueError('Frozen learner bytes changed: ' + name)
    queue_path = REPORT / 'requested_v2_evaluation_queue005'
    queue = read(queue_path / 'receipt.json')
    if (queue['status'] != 'waiting_for_exact_training_process_exit' or queue['active_child'] is not None
            or queue['training_identity'] != dict(pid=599642, start_ticks=3112307)):
        raise ValueError('Queue no longer waits for the exact learner')
    if int(Path(f"/proc/{queue['pid']}/stat").read_text().rsplit(')', 1)[1].split()[19]) != queue['start_ticks']:
        raise ValueError('Original queue exited')
    for name, expected in queue['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen evaluator bytes changed: ' + name)
    curve = REPORT / 'domain_admission_first96_005.json'
    windows = read(curve)
    if (not windows['RSL_schedule_matches_every_complete_update']
            or [window['fresh_PPO_update_numbers_inclusive'] for window in windows['windows']] != [[1, 32], [33, 64], [65, 96]]):
        raise ValueError('Need exactly three complete native windows')
    for suffix in ('.json', '.source.py'):
        shutil.copyfile(curve.with_suffix(suffix), ROOT / 'reports/robust_policy_repair_20261024' / curve.with_suffix(suffix).name)
    stamp = datetime.now(timezone.utc).isoformat()
    result = dict(schema='domain-admission-live-observation-v2', observed_utc=stamp, run=str(RUN),
        pid=599642, start_ticks=3112307, exec_session_id=1230,
        completed_updates=progress['completed_updates'], requested_updates=512,
        prior_closed_fresh_updates=3713, total_observed_fresh_updates=3713+progress['completed_updates'],
        all_training_sources_unchanged=True, frozen_training_files=len(manifest),
        training_source_commit=progress['committed_source']['commit'],
        event_comparison=str(curve), event_comparison_sha256=sha(curve), windows=windows['windows'],
        queue=dict(path=str(queue_path), pid=queue['pid'], start_ticks=queue['start_ticks'],
            status=queue['status'], frozen_files=len(queue['source_sha256']), all_sources_unchanged=True),
        candidate_preparations=['standing_phase_prepared001.json', 'unified_training_cases_prepared001.json'],
        candidates_adopted=False, new_PPO_by_observer=0,
        conclusion='First96 nominal/randomized stochastic windows remain unstable. Same learner continues its declared512 updates; no robustness improvement claimed.',
        long_training_started=False, ONNX_exported=False, qualified=False, hardware_ready=False,
        producer_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    shutil.copyfile(Path(__file__), REPORT / Path(__file__).name)
    shutil.copyfile(output, ROOT / 'reports/robust_policy_repair_20261024' / output.name)
    ledger_path = REPORT / 'scratch_campaign001.json'
    raw = ledger_path.read_bytes()
    ledger = json.loads(raw)
    if ledger['active_pid'] != 599642 or ledger['prior_closed_fresh_updates'] != 3713:
        raise ValueError('Campaign identity changed')
    ledger.update(completed_updates_observed=progress['completed_updates'], active_completed_updates_observed=progress['completed_updates'],
        total_from_zero_completed=result['total_observed_fresh_updates'], total_from_zero_completed_observed=result['total_observed_fresh_updates'],
        total_closed_from_zero_completed=3713, latest_live_read=str(output), latest_event_comparison=str(curve),
        latest_observation=result['conclusion'], observed_utc=stamp, last_observed_utc=stamp,
        prepared_reward_phase_candidate=str(ROOT / 'reports/robust_policy_repair_20261024/standing_phase_prepared001.json'),
        prepared_unified_training_cases=str(ROOT / 'reports/robust_policy_repair_20261024/unified_training_cases_prepared001.json'))
    if ledger_path.read_bytes() != raw:
        raise ValueError('Campaign changed during observation')
    temporary = ledger_path.with_suffix('.live002.tmp')
    temporary.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(ledger_path)
    print(json.dumps(dict(completed_updates=progress['completed_updates'], closed_fresh_updates=3713,
        frozen_training_files=len(manifest), frozen_queue_files=len(queue['source_sha256']), queue_waiting=True)))


if __name__ == '__main__':
    main()

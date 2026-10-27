"""Close the exact serial cold queue and preserve each paired negative result."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT = BASE / 'reports/robust_policy_repair_20261024'
QUEUE = REPORT / 'requested_v2_evaluation_queue004'
OUT = REPORT / 'isolated_sensor_cold_comparison001.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def live(pid, start):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]) == start
    except FileNotFoundError:
        return False


assert not OUT.exists()
queue = read(QUEUE / 'receipt.json')
assert queue['status'] == 'completed_unqualified' and queue['active_child'] is None
assert not live(queue['pid'], queue['start_ticks'])
assert all(sha(ROOT / name) == value for name, value in queue['source_sha256'].items())
assert [row['profile'] for row in queue['profiles']] == [
    'nominal', 'requested', 'without_encoder_bias', 'without_policy_delay', 'without_alignment']
native = [read(Path(row['run']) / 'receipt.json') for row in queue['profiles']]
gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
checks = dict(exact_queue_exited=True, no_active_child=True, same_five_native_physics=all(
    value['physics_runtime'] == native[0]['physics_runtime'] for value in native),
    frozen12_sources_unchanged=True, same_checkpoint=all(value['checkpoint_sha256'] == queue['checkpoint_sha256'] for value in native),
    no_PPO_or_optimizer=all(value['new_PPO'] == 0 and value['optimizer_constructed'] is False for value in native),
    same_seed=all(value['seed'] == 261201 for value in native),
    all_native_completed=all(value['status'] == 'completed_unqualified' for value in native),
    native_GPU_processes_exited=all(str(value['pid']) not in gpu for value in native),
    actual_nonintervened_draws_paired=all(all(row['paired_nonintervened_draw_checks'].values())
        for row in queue['profiles'] if row['profile'].startswith('without_')),
    no_middle_height_pass=all(row['middle_height']['standing_pass_v2_count'] == 0 for row in queue['profiles']))
assert all(checks.values()), checks
result = dict(schema='isolated-sensor-cold-comparison-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
    checks=checks, all_checks_pass=True, queue=str(QUEUE), queue_sha256=sha(QUEUE / 'receipt.json'),
    final_checkpoint_sha256=queue['checkpoint_sha256'],
    profiles=[dict(profile=row['profile'], middle_height=row['middle_height'],
        paired_nonintervened_draw_checks=row['paired_nonintervened_draw_checks'],
        native_receipt_sha256=row['receipt_sha256'], review_sha256=row['review_sha256']) for row in queue['profiles']],
    interpretation='This128-update sensor-cache-fix policy fails even with nominal evaluation. Single-group removal does not restore the learned controller; not proof of one unique training cause or of eventual512-update failure.',
    next_experiment='Fresh512 with nominal-population PPO warmup192 then admission ramp128; same complete requested physical widths, architecture, Gaussian reward and domain samples. No resume or teacher.',
    new_PPO_by_reader=0, long_training_started=False, ONNX_exported=False, qualified=False, hardware_ready=False,
    producer_sha256=sha(__file__))
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
(REPORT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
ledger_path = REPORT / 'scratch_campaign001.json'
raw = ledger_path.read_bytes()
ledger = json.loads(raw)
assert ledger['active_pid'] is None and ledger['prior_closed_fresh_updates'] == 3713
ledger.update(status='fresh_admission_prepared_unqualified', active_evaluation=None,
    active_evaluation_pid=None, active_evaluation_start_ticks=None,
    do_not_modify_frozen_evaluator_sources=False,
    latest_controlled_comparison=str(OUT), latest_randomization_comparison=str(OUT),
    latest_observation=result['interpretation'], next_action=result['next_experiment'],
    latest_preflight=str(REPORT / 'domain_admission_integrated_preflight001.json'),
    observed_utc=result['observed_utc'], last_observed_utc=result['observed_utc'])
assert ledger_path.read_bytes() == raw
temporary = ledger_path.with_suffix('.cold004.tmp')
temporary.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
temporary.replace(ledger_path)
print(json.dumps(dict(checks=len(checks), all_pass=True,
    middle=[dict(profile=row['profile'], passes=row['middle_height']['standing_pass_v2_count'],
        xy_m=row['middle_height']['max_xy_excursion_m'], retained_ms=row['middle_height']['retained_loss_ms']) for row in queue['profiles']])))

"""Close actual013 cold comparisons; do not infer a unique training cause."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
REPORT = BASE/REL
QUEUE = REPORT/'requested_v2_evaluation_queue005'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def live(identity):
    try:
        return int(Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(') ', 1)[1].split()[19]) == identity['start_ticks']
    except FileNotFoundError:
        return False


def main():
    output = REPORT/'domain_admission_cold_comparison001.json'
    if output.exists():
        raise ValueError('Preserve the original completed queue observation')
    queue = read(QUEUE/'receipt.json')
    seal_file = REPORT/'scratch_pilot013_closed_seal001.json'
    seal = read(seal_file)
    if (queue['status'] != 'completed_unqualified' or queue['active_child'] is not None
            or live(queue) or live(queue['training_identity'])
            or not seal['all_checks_pass'] or seal['completed_updates'] != 512):
        raise ValueError('Wait for the exact previous learner seal and entire cold queue exit')
    native = [read(Path(row['run'])/'receipt.json') for row in queue['profiles']]
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
    checks = dict(exact_queue_exited=True, original_training_exited=True,
        all5_profiles_completed=[row['profile'] for row in queue['profiles']] ==
            ['nominal','requested','without_encoder_bias','without_policy_delay','without_alignment'],
        same_checkpoint=all(row['checkpoint_sha256'] == queue['checkpoint_sha256'] == seal['final_checkpoint_sha256'] for row in native),
        same_actual_native_physics=all(row['physics_runtime'] == native[0]['physics_runtime'] for row in native),
        same_seed=all(row['seed'] == 261201 for row in native),
        no_optimizer_or_PPO=all(row['new_PPO'] == 0 and row['optimizer_constructed'] is False for row in native),
        every_native_process_exited=all(not live(row) and str(row['pid']) not in gpu for row in native),
        frozen12_evaluation_sources_unchanged=all(sha(ROOT/name) == value for name,value in queue['source_sha256'].items()),
        every_result_authenticated=all(sha(Path(row['run'])/'receipt.json') == row['receipt_sha256']
            and sha(row['review']) == row['review_sha256'] for row in queue['profiles']),
        actual_nonintervened_draws_paired=all(all(row['paired_nonintervened_draw_checks'].values())
            for row in queue['profiles'] if row['profile'].startswith('without_')))
    if not all(checks.values()):
        raise ValueError(checks)
    result = dict(schema='domain-admission-completed-cold-comparison-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(), checks=checks, all_checks_pass=True,
        completed_fresh_updates=512, total_closed_fresh_updates=4225,
        queue=str(QUEUE), queue_sha256=sha(QUEUE/'receipt.json'),
        checkpoint_sha256=queue['checkpoint_sha256'], training_seal_sha256=sha(seal_file),
        profiles=queue['profiles'], qualified=False, hardware_ready=False,
        interpretation='The same fresh013 PT restores middle-height nominal and full-requested-domain standing without reducing requested widths. Requested drift remains larger; untrained low/high heights fail. Sensor-cache repair plus changed learning admission/512 budget differ from prior experiments, so do not assign a unique causal share or claim full13 robustness.',
        next_action='Run the committed zero-PPO actual full-task runtime checks, then a fully fresh mixed-task experiment. Do not resume013.',
        new_native_steps_by_reader=0, new_PPO_by_reader=0, long_training_started=False, ONNX_exported=False,
        producer_sha256=sha(__file__))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    shutil.copyfile(output, ROOT/REL/output.name)
    shutil.copyfile(Path(__file__), REPORT/Path(__file__).name)
    ledger_file = REPORT/'scratch_campaign001.json'
    raw = ledger_file.read_bytes()
    ledger = json.loads(raw)
    if ledger['active_pid'] is not None or ledger['total_closed_from_zero_completed'] != 4225:
        raise ValueError('Unexpected fresh campaign owner/accounting')
    ledger.update(status='full_task_native_verification_prepared_unqualified',
        active_evaluation=None, active_evaluation_pid=None, active_evaluation_start_ticks=None,
        serial_evaluation_queue_pid=None, serial_evaluation_queue_start_ticks=None,
        do_not_modify_frozen_evaluator_sources=False, latest_randomization_comparison=str(output),
        latest_controlled_comparison=str(output), latest_observation=result['interpretation'],
        next_action=result['next_action'], observed_utc=result['observed_utc'])
    if ledger_file.read_bytes() != raw:
        raise ValueError('Another writer changed campaign accounting')
    temporary = ledger_file.with_suffix('.closed013.tmp')
    temporary.write_text(json.dumps(ledger,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(ledger_file)
    print(json.dumps(dict(checks=len(checks), all_pass=True,
        middle=[dict(profile=row['profile'], passes=row['middle_height']['standing_pass_v2_count'],
            XY_m=row['middle_height']['max_xy_excursion_m'], retained_ms=row['middle_height']['retained_loss_ms']) for row in queue['profiles']])))


if __name__ == '__main__':
    main()

"""Record actual013 closure and source adoption without claiming native proof."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT/REL/'fresh_native_adoption001.json'
    if output.exists():
        raise ValueError('Preserve this actual boundary observation')
    seal_file = BASE/REL/'scratch_pilot013_closed_seal001.json'
    seal = read(seal_file)
    if not seal['all_checks_pass'] or not seal['checks']['current_source_matches'] or seal['completed_updates'] != 512:
        raise ValueError('Need the previous exact512 learner closed while all266 original sources still matched')
    source = {}
    for folder in ('standing_phase_candidate001', 'unified_training_runtime_candidate001'):
        for name, row in read(ROOT/REL/folder/'receipt.json')['source'].items():
            if sha(ROOT/name) != row['candidate_sha256']:
                raise ValueError('Declared source not adopted: '+name)
            source[name] = row
    queue_file = BASE/REL/'requested_v2_evaluation_queue005/receipt.json'
    queue = read(queue_file)
    if any(sha(ROOT/name) != value for name, value in queue['source_sha256'].items()):
        raise ValueError('The active cold evaluation queue still owns all12 frozen files')
    xml = BASE/REL/'fresh_native_adoption_CPU001.xml'
    suite = ET.parse(xml).getroot().find('testsuite')
    if int(suite.attrib['tests']) != 54 or any(int(suite.attrib[key]) for key in ('errors', 'failures', 'skipped')):
        raise ValueError('Need all54 targeted CPU checks')
    preflight = BASE/REL/'fresh_unified_entry_preflight005'
    pre = read(preflight/'preflight.json')
    if pre['source_count'] != 272 or pre['new_PPO'] or pre['native_steps']:
        raise ValueError('Only the actual272-source configuration preflight is recorded here')
    for name, value in read(preflight/'source_hashes.json').items():
        if sha(ROOT/name) != value:
            raise ValueError('Configuration preflight differs from current source: '+name)
    review_file = BASE/REL/'domain_admission_terminal_review001.json'
    review = read(review_file)
    curve_file = BASE/REL/'domain_admission_final512_016.json'
    curve = read(curve_file)
    result = dict(schema='fresh-native-source-adoption-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
        preceding_closed_seal=str(seal_file), preceding_closed_seal_sha256=sha(seal_file),
        preceding_native_updates=512, preceding_checks_passed=len(seal['checks']),
        total_closed_fresh_updates=3713+512, source_adoption=source,
        CPU_tests_passed=54, CPU_xml_sha256=sha(xml),
        full_entry_preflight=str(preflight), full_entry_preflight_sha256=sha(preflight/'preflight.json'),
        terminal_v2_review=str(review_file), terminal_v2_review_sha256=sha(review_file),
        terminal_height_groups=review['groups'],
        RSL_final_observation=str(curve_file), RSL_final_observation_sha256=sha(curve_file),
        final_complete_training_window=curve['windows'][-1],
        queue_status_at_adoption=queue['status'], queue12_sources_unchanged=True,
        native_verification_entry='scripts/verify_fresh_unified_runtime.py',
        native_verification_entry_sha256=sha(ROOT/'scripts/verify_fresh_unified_runtime.py'),
        launch_guard='Wait for exact prior seal and cold queue exit; pass actual native runtime and stable-hold wiring; source/bank-bound exited receipt required before any fresh full-task PPO',
        new_native_steps=0, new_PPO=0, source_adopted=True, native_wiring_verified=False,
        behavior_qualified=False, hardware_ready=False)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    shutil.copyfile(output, BASE/REL/output.name)
    shutil.copyfile(Path(__file__), BASE/REL/Path(__file__).name)
    shutil.copyfile(xml, ROOT/REL/xml.name)
    stored = ROOT/REL/preflight.name
    stored.mkdir(exist_ok=False)
    for name in ('preflight.json', 'mechanism_preflight.json', 'floor_geometry.json'):
        shutil.copyfile(preflight/name, stored/name)
    ledger_file = BASE/REL/'scratch_campaign001.json'
    ledger = read(ledger_file)
    ledger.update(status='serial_evaluating_unqualified', active_run=None, active_pid=None,
        active_start_ticks=None, completed_updates_observed=0, active_completed_updates_observed=0,
        prior_from_zero_completed=4225, prior_completed_fresh_updates=4225, prior_closed_fresh_updates=4225,
        total_closed_from_zero_completed=4225, total_from_zero_completed=4225,
        total_from_zero_completed_observed=4225, last_run_seal=str(seal_file),
        last_completed_run=str(BASE/'runs/scratch_standing_20261026/pilot013_gradual_admission_requested001'),
        latest_user_acceptance_review=str(review_file), latest_event_comparison=str(curve_file),
        latest_observed_utc=result['observed_utc'], do_not_modify_live_src=False,
        do_not_modify_training_entrypoint=False, do_not_modify_domain_admission_helper=False,
        current_generation_middle_height_qualification=False,
        latest_observation='013 actual512 and20 closure checks; warm nominal middle7/7; cold nominal7/7. Requested cold queue remains active. Full13,3heights and hardware unqualified.',
        prepared_full_task_native_verification=str(ROOT/'scripts/verify_fresh_unified_runtime.py'),
        next_action='Finish same013 cold requested domains, then zero-PPO native wiring checks before a fully fresh mixed-task experiment')
    ledger['prepared_next_fresh_experiment']['status'] = 'completed_unqualified'
    ledger_file.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('preceding_checks_passed', 'total_closed_fresh_updates', 'CPU_tests_passed', 'new_native_steps', 'new_PPO')}))


if __name__ == '__main__':
    main()

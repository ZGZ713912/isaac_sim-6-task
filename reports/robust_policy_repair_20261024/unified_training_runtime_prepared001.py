"""Preserve unadopted full-task runtime and the actual013 full-population start."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
RUN = BASE / 'runs/scratch_standing_20261026/pilot013_gradual_admission_requested001'
sys.path.insert(0, str(ROOT / 'scripts'))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / REL / 'unified_training_runtime_prepared001.json'
    if output.exists():
        raise ValueError('Preserve earlier observations')
    progress = read(RUN / 'progress.json')
    if (progress['pid'] != 599642 or progress['start_ticks'] != 3112307
            or progress['status'] != 'training' or progress['completed_updates'] < 322
            or progress['requested_updates'] != 512 or progress['checkpoint_reads']):
        raise ValueError('Expected the original fresh learner after real full admission')
    if int(Path('/proc/599642/stat').read_text().rsplit(')', 1)[1].split()[19]) != 3112307:
        raise ValueError('Original learner exited')
    manifest = read(RUN / 'source_hashes.json')
    for name, expected in manifest.items():
        if sha(ROOT / name) != expected or sha(RUN / 'source_snapshot' / name) != expected:
            raise ValueError('Frozen learner bytes changed: '+name)
    queue = read(BASE / REL / 'requested_v2_evaluation_queue005/receipt.json')
    if (queue['status'] != 'waiting_for_exact_training_process_exit' or queue['active_child'] is not None
            or queue['training_identity'] != dict(pid=599642, start_ticks=3112307)):
        raise ValueError('Expected the exact evaluator still waiting')
    if int(Path(f"/proc/{queue['pid']}/stat").read_text().rsplit(')', 1)[1].split()[19]) != queue['start_ticks']:
        raise ValueError('Original evaluator queue exited')
    for name, expected in queue['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen evaluator changed: '+name)
    xml = BASE / REL / 'unified_cases_and_runtime_CPU003.xml'
    suite = ET.parse(xml).getroot().find('testsuite')
    if int(suite.attrib['tests']) != 16 or any(int(suite.attrib[k]) for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Need16 passing task/native-block/clock counterexamples')
    shutil.copyfile(xml, ROOT / REL / xml.name)
    candidate = BASE / REL / 'unified_training_runtime_candidate001'
    subprocess.run(['git', 'apply', '--check',
        str(ROOT / REL / 'standing_phase_candidate001/integration.patch'),
        str(candidate / 'integration.patch')], cwd=ROOT, check=True)
    stored = ROOT / REL / candidate.name
    stored.mkdir(exist_ok=False)
    for name in ('integration.patch', 'receipt.json'):
        shutil.copyfile(candidate / name, stored / name)
    curves = []
    for name in ('domain_admission_ramp256_011', 'domain_admission_ramp288_012'):
        curve = BASE / REL / (name+'.json')
        evidence = read(curve)
        if not evidence['RSL_schedule_matches_every_complete_update'] or evidence['current_source_mismatches']:
            raise ValueError('Actual RSL admission evidence rejected')
        for suffix in ('.json', '.source.py'):
            shutil.copyfile(curve.with_suffix(suffix), ROOT / REL / (name+suffix))
        curves.append(dict(file=str(curve), sha256=sha(curve),
            updates=evidence['common_complete_RSL_updates'],
            complete_window_endpoint=evidence['windows'][-1]['fresh_PPO_update_numbers_inclusive'][1]))
    latest = read(BASE / REL / 'domain_admission_ramp288_012.json')
    if latest['latest_complete_native_population']['population_fraction'] != 1.:
        raise ValueError('All256 must actually appear in the real RSL population')
    from unified_training_cases_candidate import TrainingCases
    bank = ROOT / REL / 'pose_bank_height_matched073.json'
    catalogue = TrainingCases(read(bank), 256, 271701)
    files = ('scripts/unified_training_cases_candidate.py', 'scripts/unified_training_runtime_candidate.py',
        'tests/test_unified_training_cases_candidate.py', 'tests/test_unified_training_runtime_candidate.py')
    result = dict(schema='fresh-unified-native-runtime-preparation-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(),
        live_learner=dict(pid=599642, start_ticks=3112307, completed_updates=progress['completed_updates'],
            requested_updates=512, source_commit=progress['committed_source']['commit'],
            frozen_training_files=len(manifest), all_sources_unchanged=True),
        waiting_queue=dict(pid=queue['pid'], start_ticks=queue['start_ticks'],
            frozen_files=len(queue['source_sha256']), all_sources_unchanged=True),
        actual_RSL_observations=curves,
        actual_latest_population=latest['latest_complete_native_population'],
        CPU_tests_passed=16, CPU_xml_sha256=sha(xml),
        CPU_scope='Actual native sequence/timeout/autoreset blocks and grounding selector executed on CPU; environment factory uses synthetic protocol feedback, not Isaac physics.',
        candidate=str(candidate), candidate_receipt=read(candidate / 'receipt.json'),
        source_sha256={name: sha(ROOT / name) for name in files},
        phases=[catalogue.summary(update) for update in (0, 192, 480, 768)],
        command_curriculum='Requests and transition amplitudes ramp only in reset data; final80/100rpm and3/-2.5m/s remain unchanged. Native reward at rest otherwise has negligible tracking signal.',
        combined_reward_and_runtime_patch_check_passed=True,
        candidate_applied=False, actual_native_class_constructed=False,
        new_native_steps=0, new_PPO=0, qualified=False, hardware_ready=False,
        integration_remaining=['Actual013 exit and integrity seal before any frozen source change',
            'Fresh full-task entry, height coefficient/source authentication and extended integrity seal',
            'Real native sequence/autoreset/phase/grounded-birth verification before fresh PPO',
            'Same-policy full13 capabilities, independent seeds and held-out domains before long training/ONNX'])
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    shutil.copyfile(output, BASE / REL / output.name)
    shutil.copyfile(Path(__file__), BASE / REL / Path(__file__).name)
    ledger_file = BASE / REL / 'scratch_campaign001.json'
    raw = ledger_file.read_bytes()
    ledger = json.loads(raw)
    if ledger['active_pid'] != 599642 or ledger['prior_closed_fresh_updates'] != 3713:
        raise ValueError('Campaign identity changed')
    ledger.update(completed_updates_observed=progress['completed_updates'],
        active_completed_updates_observed=progress['completed_updates'],
        total_from_zero_completed=3713+progress['completed_updates'],
        total_from_zero_completed_observed=3713+progress['completed_updates'],
        total_closed_from_zero_completed=3713, latest_event_comparison=curves[-1]['file'],
        prepared_unified_runtime=str(output), latest_live_read=str(output),
        latest_observation='All256 actual PPO lanes admitted after clock320; nominal balance improved but randomized drift remains. Same512 continues; no native standing qualification.',
        observed_utc=result['observed_utc'], last_observed_utc=result['observed_utc'])
    if ledger_file.read_bytes() != raw:
        raise ValueError('Campaign changed during observation')
    temporary = ledger_file.with_suffix('.runtime-prepared.tmp')
    temporary.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(ledger_file)
    print(json.dumps({key: result[key] for key in ('live_learner', 'actual_latest_population', 'CPU_tests_passed', 'candidate_applied')}))


if __name__ == '__main__':
    main()

"""Record actual CPU construction, backward seal and untouched live learner."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
RUN = BASE/'runs/scratch_standing_20261026/pilot013_gradual_admission_requested001'
sys.path.insert(0, str(ROOT/'scripts'))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT/REL/'fresh_unified_entry_prepared001.json'
    if output.exists():
        raise ValueError('Preserve earlier preparation')
    preflight = BASE/REL/'fresh_unified_entry_preflight004'
    manifest = read(preflight/'source_hashes.json')
    for name, expected in manifest.items():
        if sha(ROOT/name) != expected or sha(preflight/'source_snapshot'/name) != expected:
            raise ValueError('Preflight source changed: '+name)
    if read(preflight/'mechanism_preflight.json')['status'] != 'static_passed':
        raise ValueError('Need actual independent CPU mechanism preflight')
    import train_fresh_unified as entry
    rejected = False
    try:
        entry.require_adopted_native_sources()
    except ValueError as error:
        rejected = 'declared adopted source' in str(error)
    if not rejected:
        raise ValueError('Candidate source gate unexpectedly permitted a native launch')
    tests = {}
    for name, expected in (('fresh_unified_entry_CPU003.xml', 7), ('fresh_integrity_contract_CPU001.xml', 2)):
        path = BASE/REL/name
        suite = ET.parse(path).getroot().find('testsuite')
        if int(suite.attrib['tests']) != expected or any(int(suite.attrib[k]) for k in ('failures', 'errors', 'skipped')):
            raise ValueError('CPU validation rejected: '+name)
        shutil.copyfile(path, ROOT/REL/name)
        tests[name] = dict(passed=expected, sha256=sha(path))
    failed = BASE/REL/'fresh_unified_entry_CPU001.xml'
    shutil.copyfile(failed, ROOT/REL/failed.name)
    backward = BASE/REL/'scratch_pilot012_seal_backward_check002.json'
    if not read(backward)['all_checks_pass']:
        raise ValueError('Old actual128-update integrity seal regressed')
    for path in (backward, backward.with_suffix('.source.py')):
        shutil.copyfile(path, ROOT/REL/path.name)
    stored = ROOT/REL/'fresh_unified_entry_preflight004'
    stored.mkdir(exist_ok=False)
    for name in ('preflight.json', 'mechanism_preflight.json', 'floor_geometry.json'):
        shutil.copyfile(preflight/name, stored/name)
    curve = BASE/REL/'domain_admission_full416_014.json'
    evidence = read(curve)
    if evidence['windows'][-1]['fresh_PPO_update_numbers_inclusive'][1] != 416:
        raise ValueError('Need actual complete416-update evidence')
    for path in (curve, curve.with_suffix('.source.py')):
        shutil.copyfile(path, ROOT/REL/path.name)
    progress = read(RUN/'progress.json')
    if (progress['pid'] != 599642 or progress['start_ticks'] != 3112307
            or progress['status'] != 'training' or progress['checkpoint_reads']
            or progress['requested_updates'] != 512):
        raise ValueError('Actual013 identity changed')
    frozen = read(RUN/'source_hashes.json')
    for name, expected in frozen.items():
        if sha(ROOT/name) != expected or sha(RUN/'source_snapshot'/name) != expected:
            raise ValueError('Live training bytes changed: '+name)
    for pid, ticks in ((599642, 3112307), (601306, 3119635)):
        if int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]) != ticks:
            raise ValueError('Exact native/queue process exited')
    queue = read(BASE/REL/'requested_v2_evaluation_queue005/receipt.json')
    for name, expected in queue['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen evaluator changed: '+name)
    if queue['active_child'] is not None or queue['status'] != 'waiting_for_exact_training_process_exit':
        raise ValueError('Queue no longer waiting')
    checked_files = ('config/contract.json', 'config/design.json', 'task_contract.json',
        'source_hashes.json', 'height_initialization_reference.json')
    result = dict(schema='fresh-unified-entry-preparation-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(),
        entry='scripts/train_fresh_unified.py', preflight=str(preflight), source_count=len(manifest),
        preflight_sha256={name: sha(preflight/name) for name in checked_files},
        entry_sha256=sha(ROOT/'scripts/train_fresh_unified.py'), CPU_tests=tests,
        prior_rejected_test=dict(path=str(failed), sha256=sha(failed),
            reason='Test incorrectly expected a passed boolean; actual independent preflight returns status=static_passed. Corrected before successful runs.'),
        unadopted_native_source_gate_rejected=True, old128_update_seal_backward_check=read(backward),
        fresh_plan=dict(updates=1536, curriculum='three heights from start; task population and request amplitudes192..768; permanent full tasks thereafter',
            source_archive='272 committed files required before any native PPO; every actor/critic parameter trainable; empty Adam and no checkpoint reads',
            initialization='ordinary trainable static-mechanics height coefficients; no policy donor or teacher',
            randomization='full requested v2; same192+128 domain admission candidate',
            actual_episode_clock=True, standing_failure_resets=False,
            true_leg_speed_extrema='every five1ms samples per policy interval; direct RSL writer once per completed rollout',
            terminal_evaluation='separate cold full13 same-policy suite; no standing-only success inference'),
        active013=dict(pid=599642, start_ticks=3112307, completed_updates=progress['completed_updates'],
            requested_updates=512, frozen_source_count=len(frozen), all_sources_unchanged=True,
            waiting_queue_pid=601306, frozen_queue_count=len(queue['source_sha256']), queue_sources_unchanged=True),
        RSL_observation=str(curve), RSL_observation_sha256=sha(curve),
        new_entry_native_steps=0, new_entry_PPO=0, training_started=False,
        native_full_task_verification=False, qualified=False, hardware_ready=False,
        remaining=['Actual013 exit/seal and its cold queue results',
            'Apply verified phase/runtime patches, then native sequence/partial-reset/grounded-birth/phase checks',
            'Cold evaluator height-actor support and actual full13 same-checkpoint evaluation',
            'Behavior-based selection before long training, ONNX parity and deployment'])
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    shutil.copyfile(output, BASE/REL/output.name)
    shutil.copyfile(Path(__file__), BASE/REL/Path(__file__).name)
    print(json.dumps({key: result[key] for key in ('source_count', 'active013', 'training_started', 'native_full_task_verification')}))


if __name__ == '__main__':
    main()

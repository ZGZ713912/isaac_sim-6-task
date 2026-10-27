"""Archive full-suite CPU migration and actual448-update RSL diagnostics."""
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
    output = ROOT/REL/'full_suite_height_grounding_prepared001.json'
    if output.exists():
        raise ValueError('Preserve earlier preparation')
    xml = BASE/REL/'full_suite_height_grounding_CPU001.xml'
    suite = ET.parse(xml).getroot().find('testsuite')
    if int(suite.attrib['tests']) != 22 or any(int(suite.attrib[k]) for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Need22 passing actor/grounding and full capability checks')
    preflight = BASE/REL/'full_suite009_height_grounding_preflight002'
    pre = read(preflight/'receipt.json')
    if (pre['status'] != 'CPU_preflight_passed_unqualified' or pre['case_count'] != 198
            or pre['native_policy_steps'] != 0 or pre['new_PPO'] != 0):
        raise ValueError('Need actual198-case CPU preflight, never a native-pass claim')
    if sha(ROOT/'scripts/evaluate_fresh_full_suite.py') != pre['diagnostic_source_hashes']['evaluate_fresh_full_suite.py']:
        raise ValueError('Evaluator changed after actual preflight')
    stored = ROOT/REL/preflight.name
    stored.mkdir(exist_ok=False)
    for name in ('receipt.json', 'preflight.json', 'floor_geometry.json', 'diagnostic_sources/full_suite_grounded_reset.py'):
        destination = stored/name
        destination.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(preflight/name, destination)
    curve = BASE/REL/'domain_admission_full448_015.json'
    figure = BASE/REL/'domain_admission_full448_002.json'
    curve_data, figure_data = read(curve), read(figure)
    if curve_data['windows'][-1]['fresh_PPO_update_numbers_inclusive'] != [417,448]:
        raise ValueError('Need the actual complete448-update window')
    if (figure_data['source_reports'][str(curve)] != sha(curve)
            or not figure_data['actual_event_prefixes_verified']):
        raise ValueError('Figure does not match the real RSL observation')
    for path in (curve, curve.with_suffix('.source.py'), figure, figure.with_suffix('.png'), figure.with_suffix('.svg'), xml):
        if path.suffix in ('.png', '.svg') and sha(path) != figure_data['figure_sha256'][path.suffix]:
            raise ValueError('Figure changed')
        shutil.copyfile(path, ROOT/REL/path.name)
    run = BASE/'runs/scratch_standing_20261026/pilot013_gradual_admission_requested001'
    progress = read(run/'progress.json')
    if (progress['pid'], progress['start_ticks'], progress['status']) != (599642,3112307,'training'):
        raise ValueError('Expected the same active original013 learner')
    frozen = read(run/'source_hashes.json')
    for name, expected in frozen.items():
        if sha(ROOT/name) != expected or sha(run/'source_snapshot'/name) != expected:
            raise ValueError('Active source changed: '+name)
    queue = read(BASE/REL/'requested_v2_evaluation_queue005/receipt.json')
    for name, expected in queue['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen queue source changed: '+name)
    result = dict(schema='full-suite-height-grounding-preparation-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(), CPU_tests_passed=22,
        CPU_xml_sha256=sha(xml), full_suite_preflight=str(preflight), preflight_sha256=sha(preflight/'receipt.json'),
        evaluator_sha256=sha(ROOT/'scripts/evaluate_fresh_full_suite.py'),
        height_actor='Authenticate mechanical coefficient reference and all three actual frozen inheritance sources',
        grounding='Direct native RobustGetupEnv case initialization; ground all upright stationary/moving/spinning births. Legacy source copy changes eligibility only; new archive uses its explicit argument. No writes during evaluation steps.',
        actual_legacy_initialization_migration=pre['initial_grounding'],
        RSL_curve=str(curve), RSL_curve_sha256=sha(curve), latest_complete_window=curve_data['windows'][-1],
        figure=str(figure.with_suffix('.png')), figure_sha256=figure_data['figure_sha256'], figure_visually_checked=True,
        active013_completed_updates=progress['completed_updates'], prior_closed_fresh_updates=3713,
        frozen_266_training_and_12_queue_sources_unchanged=True,
        new_native_steps=0, new_PPO=0, actual_height_checkpoint_evaluated=False,
        native_full_suite_verified=False, qualified=False, hardware_ready=False,
        scope='CPU preparation and actual stochastic training diagnostics only; no full-task or deployment pass')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    shutil.copyfile(output, BASE/REL/output.name)
    shutil.copyfile(Path(__file__), BASE/REL/Path(__file__).name)
    shutil.copyfile(ROOT/REL/'domain_admission_training_plot002.py', BASE/REL/'domain_admission_training_plot002.py')
    print(json.dumps({key: result[key] for key in ('CPU_tests_passed', 'active013_completed_updates', 'new_native_steps', 'qualified')}))


if __name__ == '__main__':
    main()

"""Promote reviewed CPU sources only after the four native replays are sealed."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_tasks.chassis.flat_base_continuation import digest, encode

CANDIDATE = HERE / 'motion_grip_candidate001'
target = CANDIDATE / 'adoption001.json'
if target.exists():
    raise SystemExit('Preserve existing source adoption')
queue_path = ROOT / 'runs/flat_base_20261019/static_grip1338_envelopes_queue001.json'
queue = json.loads(queue_path.read_text())
if queue['status'] != 'complete' or len(queue['history']) != 4:
    raise SystemExit('Wait for all four native replays')
if any(Path('/proc/' + str(queue[key])).exists() for key in ('pid', 'worker_pid')):
    raise SystemExit('Wait for both replay owners to exit')
seal_path = HERE / 'static_grip_candidate002/envelopes_completion001.json'
seal = json.loads(seal_path.read_text())
if (not seal['execution_verified'] or not all(seal['checks'].values())
        or not seal['no_lost_parent_checks']):
    raise SystemExit('Require the reviewed full replay seal')
if any(digest(ROOT / name) != value for name, value in queue['source_sha256'].items()):
    raise SystemExit('The frozen old replay sources changed before adoption')
origin = json.loads((CANDIDATE / 'staging_origin001.json').read_text())
cpu = json.loads((CANDIDATE / 'cpu_preparation001.json').read_text())
for name, expected_tests in (('cpu_proposal002.xml', 4), ('cpu_staged002.xml', 9)):
    suites = ET.parse(CANDIDATE / name).getroot()
    if (sum(int(s.attrib['tests']) for s in suites) != expected_tests
            or any(int(s.attrib.get('failures', 0)) or int(s.attrib.get('errors', 0)) for s in suites)):
        raise SystemExit('Require the exact passing isolated CPU checks: ' + name)
staged = CANDIDATE / 'staged_root'
files = list(origin['base_sha256']) + ['src/wheeled_algo/motion_grip.py']
if digest(staged / 'src/wheeled_algo/motion_grip.py') != cpu['proposal_sha256']:
    raise SystemExit('The tested motion module changed')
for name, old_hash in origin['base_sha256'].items():
    if digest(ROOT / name) != old_hash:
        raise SystemExit('Current source differs from the staging origin: ' + name)
if (ROOT / 'src/wheeled_algo/motion_grip.py').exists():
    raise SystemExit('Preserve an existing motion adaptation module')
test_path = ROOT / 'tests/test_motion_grip_adaptation.py'
if test_path.exists():
    raise SystemExit('Preserve an existing motion adaptation test')
new_hashes = {name:digest(staged / name) for name in files}
cpu_seal_path = CANDIDATE / 'staged_cpu_seal001.json'
cpu_seal = json.loads(cpu_seal_path.read_text())
if (new_hashes != cpu_seal['source_sha256'] or cpu_seal['passed_tests'] != 9
        or digest(CANDIDATE / 'proposed_tests.py') != cpu_seal['tests_sha256']
        or digest(CANDIDATE / 'cpu_staged002.xml') != cpu_seal['junit_sha256']):
    raise SystemExit('The isolated CPU-tested source or test bytes changed')
for name in files:
    (ROOT / name).write_bytes((staged / name).read_bytes())
test_path.write_bytes((CANDIDATE / 'proposed_tests.py').read_bytes())
proposal_test = ROOT / 'tests/test_motion_grip_candidate.py'
snapshot = CANDIDATE / 'cpu_proposal_test_snapshot.py'
if proposal_test.exists() and not snapshot.exists():
    proposal_test.rename(snapshot)
assert all(digest(ROOT / name) == sha for name, sha in new_hashes.items())
target.write_bytes(encode(dict(schema='flat-motion-grip-source-adoption-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), adopted=True,
    full_replay_seal_sha256=digest(seal_path), preceding_queue_sha256=digest(queue_path),
    old_source_sha256=origin['base_sha256'], new_source_sha256=new_hashes,
    tests_file=str(test_path.relative_to(ROOT)), tests_sha256=digest(test_path),
    isolated_CPU_checks=9, isolated_CPU_source_seal_sha256=digest(cpu_seal_path), reader_sha256=digest(Path(__file__)),
    native_training_started=False, new_optimizer_updates=0, qualified=False)))
print(json.dumps(dict(adopted=True, files=files, tests=str(test_path.relative_to(ROOT)))))

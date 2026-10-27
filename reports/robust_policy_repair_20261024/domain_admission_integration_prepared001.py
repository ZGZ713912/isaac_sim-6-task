"""Archive the unapplied optional-admission patch and its CPU evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT = BASE / 'reports/robust_policy_repair_20261024'
OUT = REPORT / 'domain_admission_integration_prepared001.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not OUT.exists()
candidate = REPORT / 'domain_admission_integration_candidate001'
receipt = read(candidate / 'receipt.json')
assert receipt['candidate_applied'] is False
assert sha(ROOT / 'scripts/train_scratch_standing.py') == receipt['original_sha256']
assert sha(ROOT / 'scripts/domain_learning_admission_candidate.py') == receipt['helper_sha256']
suite = ET.parse(REPORT / 'domain_admission_integration_CPU002.xml').getroot().find('testsuite')
assert int(suite.attrib['tests']) == 12 and int(suite.attrib['failures']) == int(suite.attrib['errors']) == 0
queue = read(REPORT / 'requested_v2_evaluation_queue004/receipt.json')
assert len(queue['source_sha256']) == 12
assert all(sha(ROOT / name) == digest for name, digest in queue['source_sha256'].items())
run = BASE / 'runs/scratch_standing_20261026/pilot012_isolated_sensor_requested001'
manifest = read(run / 'source_hashes.json')
assert len(manifest) == 265 and all(sha(ROOT / name) == digest for name, digest in manifest.items())
from sys import path
path.insert(0, str(ROOT / 'scripts'))
from domain_learning_admission_candidate import plan

result = dict(schema='domain-admission-integration-prepared-v1',
    observed_utc=datetime.now(timezone.utc).isoformat(), candidate_directory=str(candidate),
    candidate_receipt=receipt, candidate_applied=False, live012_source_count=265,
    all_live012_and_queue004_source_bytes_unchanged=True,
    CPU_tests=12, CPU_test_receipt=str(REPORT / 'domain_admission_integration_CPU002.xml'),
    CPU_test_receipt_sha256=sha(REPORT / 'domain_admission_integration_CPU002.xml'),
    proposed_fresh_experiment=plan(192, 128, 512, 261501),
    training_changes='Optional PPO/normalizer sample admission only; full physical widths throughout.',
    population='The existing nominal sensor/actuator lanes retain the requested ground-friction distribution; this is not an exact replay of009.',
    schedule='At completed actor clock192 no randomized lanes are admitted; full population at clock320, first full update321, last192 updates full.',
    adopted_in012=False, new_native_PPO=0, native_behavior_verified=False,
    qualified=False, hardware_ready=False, producer_sha256=sha(__file__))
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
(REPORT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print(json.dumps({key: result[key] for key in ('candidate_applied', 'CPU_tests', 'live012_source_count', 'new_native_PPO')}))

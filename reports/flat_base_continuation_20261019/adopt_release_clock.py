"""Adopt the reviewed release-clock reader only after the frozen trial is sealed."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / 'release_clock_candidate001'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
out = HERE / 'release_clock_adoption024.json'
if out.exists():
    raise SystemExit('Preserve the completed adoption receipt')
queue_path = ROOT / 'runs/flat_base_20261019/wheel_from1280_queue001.json'
queue = read(queue_path)
seal_path = HERE / 'wheel_from1280_completion001.json'
seal = read(seal_path)
binding = read(CANDIDATE / 'parent_binding.json')
proof_path = CANDIDATE / 'cpu001.json'
proof = read(proof_path)
production = ROOT / 'scripts/export_flat_base.py'
candidate = CANDIDATE / 'export_flat_base.py'
backup = CANDIDATE / 'parent_export_flat_base.py'
temporary = production.with_name('.export_flat_base.clock_adoption.py')
checks = dict(
    all_frozen_trial_owners_exited=queue['status'] == 'complete'
        and all(not Path('/proc/' + str(queue[k])).exists() for k in ('pid', 'worker_pid')),
    completed_trial_sealed=seal['passed'] and all(seal['checks'].values())
        and seal['queue_sha256'] == sha(queue_path),
    exact_reviewed_parent=sha(production) == binding['parent_export_sha256']
        == proof['parent_export_sha256'] == queue['source_sha256']['scripts/export_flat_base.py'],
    exact_reviewed_candidate=sha(candidate) == binding['candidate_export_sha256']
        == proof['candidate_export_sha256'],
    full_twenty_test_proof=proof['passed'] and proof['exit_code'] == 0
        and proof['tests_passed'] == 20 and not proof['failed_tests'],
    exact_proof_inputs=sha(CANDIDATE / 'test_release_clock.py') == proof['new_test_sha256']
        and sha(CANDIDATE / 'verify_cpu.py') == proof['verifier_sha256']
        and sha(ROOT / 'tests/test_flat_release_evidence.py') == proof['existing_release_test_sha256'],
    no_prior_backup=not backup.exists(),
    no_existing_temporary_source=not temporary.exists(),
)
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
old_bytes, new_bytes = production.read_bytes(), candidate.read_bytes()
backup.write_bytes(old_bytes)
temporary.write_bytes(new_bytes)
temporary.chmod(stat.S_IMODE(production.stat().st_mode))
temporary.replace(production)
checks['adopted_source_matches_reviewed_candidate'] = sha(production) == binding['candidate_export_sha256']
checks['parent_source_preserved_exactly'] = sha(backup) == binding['parent_export_sha256']
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
out.write_text(json.dumps(dict(schema='flat-release-clock-adoption-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
    trial_seal_sha256=sha(seal_path), candidate_CPU_sha256=sha(proof_path),
    old_export_sha256=sha(backup), new_export_sha256=sha(production),
    parent_backup=str(backup.relative_to(ROOT)), adopter_sha256=sha(Path(__file__)),
    change='Read training frequency from the authenticated plan; retain all eleven cohorts and numerical gates.',
    new_optimizer_updates=0, new_native_steps=0, behavior_qualified=False,
    hardware_motion_started=False), indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=True, checks=len(checks), new_export_sha256=sha(production)), indent=2))

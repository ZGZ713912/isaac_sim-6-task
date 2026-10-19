"""Seal the completed six-child native run before adopting another source."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'runs/flat_base_20261019'
output = HERE/'feedback832_queue_completion001.json'
if output.exists():
    raise SystemExit('Preserve the existing immutable completion receipt')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
queue_path = RUN/'feedback832_queue001.json'
queue = read(queue_path)
review_path = HERE/'feedback832_behavior_review001.json'
review = read(review_path)
native = read(HERE/'feedback832_native_milestones.json')
checkpoint = RUN/'pilot016/model_000832.pt'
checks = dict(
    queue_complete=queue['status']=='complete',
    all_six_children_exit0=len(queue['history'])==6 and all(r['exit_code']==0 for r in queue['history']),
    both_owners_exited=not any(Path('/proc/'+str(queue[k])).exists() for k in ('pid','worker_pid')),
    all121_frozen_sources_unchanged=len(queue['source_sha256'])==121 and all(sha(ROOT/n)==v for n,v in queue['source_sha256'].items()),
    exactly64_completed_optimizer_updates=queue['new_optimizer_updates']==64,
    native_learning_milestones_passed=native['passed'] and native['all_required_milestones_present'],
    all_four_behavior_cohorts_complete=review['all_four_cohorts_complete'],
    exact_checkpoint_in_queue_and_review=sha(checkpoint)==queue['checkpoint_sha256']==review['checkpoint_sha256'])
receipt = dict(schema='flat-feedback832-completed-training-replay-freeze-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), checks=checks, passed=all(checks.values()),
    source_count=len(queue['source_sha256']), queue_sha256=sha(queue_path), checkpoint_sha256=sha(checkpoint),
    behavior_review_sha256=sha(review_path), native_verification_sha256=sha(HERE/'feedback832_native_milestones.json'),
    optimizer_updates_performed=64, behavior_qualified=False, hardware_motion_started=False)
if not receipt['passed']:
    print(json.dumps(receipt, indent=2)); raise SystemExit('The native run is not ready to seal')
output.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))

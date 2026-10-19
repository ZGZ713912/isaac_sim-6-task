"""Freeze the clock pilot only after all native owners have exited."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
output = HERE / 'feedback1248_queue_completion001.json'
if output.exists():
    raise SystemExit('Preserve the completed clock-pilot receipt')
queuepath = RUN / 'feedback1248_queue001.json'
queue = read(queuepath)
reviewpath = HERE / 'feedback1248_behavior_review001.json'
review = read(reviewpath)
nativepath = HERE / 'feedback1248_native_milestones.json'
native = read(nativepath)
eventpath = HERE / 'feedback1248_training_event_review001.json'
events = read(eventpath)
revisionpath = HERE / 'continuation1248_contract001.json'
revision = read(revisionpath)
checkpoint = RUN / 'pilot023/model_001248.pt'
checks = dict(queue_complete=queue['status']=='complete',
    all_five_children_exit0=len(queue['history'])==5 and all(r['exit_code']==0 for r in queue['history']),
    both_owners_exited=not any(Path('/proc/'+str(queue[k])).exists() for k in ('pid','worker_pid')),
    all128_frozen_sources_unchanged=len(queue['source_sha256'])==128 and all(sha(ROOT/n)==v for n,v in queue['source_sha256'].items()),
    exactly64_completed_optimizer_updates=queue['new_optimizer_updates']==64,
    exact_native_restore_clock_and_learning=native['passed'] and native['all_required_milestones_present'],
    all_three_screening_cohorts_complete=review['all_three_cohorts_complete'],
    actual_same_checkpoint_in_all_reports=sha(checkpoint)==queue['checkpoint_sha256']==review['checkpoint_sha256']==events['checkpoint_sha256'],
    unchanged_revision_and_prerequisites=sha(revisionpath)==queue['continuation_contract_sha256'] and all(
        sha(HERE/n)==v for n,v in revision['prerequisite_sha256'].items()),
    all_history_including0_critic64_actor_preserved=tuple(review[k] for k in ('successful_updates_total',
        'actor_updates_total','feedback_branch_actor_updates','post_split_actor_updates',
        'inherited_shared_feedback_updates','consumed_optimizer_updates_total')) == (1248,1068,690,386,304,1249))
receipt = dict(schema='flat-feedback1248-completed-unchanged-v16-pilot-freeze-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    passed=all(checks.values()),checks=checks,queue_sha256=sha(queuepath),checkpoint_sha256=sha(checkpoint),
    behavior_review_sha256=sha(reviewpath),native_verification_sha256=sha(nativepath),
    event_review_sha256=sha(eventpath),continuation_contract_sha256=sha(revisionpath),
    optimizer_updates_performed=64,critic_only_updates=0,actor_updates=64,source_count=128,
    behavior_qualified=False,hardware_motion_started=False)
if not receipt['passed']:
    print(json.dumps(receipt,indent=2));raise SystemExit('Same-plan pilot is not ready to seal')
output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

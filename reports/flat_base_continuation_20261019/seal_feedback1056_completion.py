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
output = HERE / 'feedback1056_queue_completion001.json'
if output.exists():
    raise SystemExit('Preserve the completed clock-pilot receipt')
queuepath = RUN / 'feedback1056_queue001.json'
queue = read(queuepath)
reviewpath = HERE / 'feedback1056_behavior_review001.json'
review = read(reviewpath)
nativepath = HERE / 'feedback1056_native_milestones.json'
native = read(nativepath)
eventpath = HERE / 'feedback1056_training_event_review001.json'
events = read(eventpath)
revisionpath = HERE / 'training_plan_revision017.json'
revision = read(revisionpath)
checkpoint = RUN / 'pilot020/model_001056.pt'
checks = dict(queue_complete=queue['status']=='complete',
    all_five_children_exit0=len(queue['history'])==5 and all(r['exit_code']==0 for r in queue['history']),
    both_owners_exited=not any(Path('/proc/'+str(queue[k])).exists() for k in ('pid','worker_pid')),
    all128_frozen_sources_unchanged=len(queue['source_sha256'])==128 and all(sha(ROOT/n)==v for n,v in queue['source_sha256'].items()),
    exactly32_completed_optimizer_updates=queue['new_optimizer_updates']==32,
    exact_native_restore_clock_and_learning=native['passed'] and native['all_required_milestones_present'],
    all_three_screening_cohorts_complete=review['all_three_cohorts_complete'],
    actual_same_checkpoint_in_all_reports=sha(checkpoint)==queue['checkpoint_sha256']==review['checkpoint_sha256']==events['checkpoint_sha256'],
    unchanged_revision_and_prerequisites=sha(revisionpath)==queue['plan_revision_sha256'] and all(
        sha(HERE/n)==v for n,v in revision['prerequisite_sha256'].items()),
    all_history_including10_critic22_actor_preserved=tuple(review[k] for k in ('successful_updates_total',
        'actor_updates_total','feedback_branch_actor_updates','post_split_actor_updates',
        'inherited_shared_feedback_updates','consumed_optimizer_updates_total')) == (1056,886,508,204,304,1057))
receipt = dict(schema='flat-feedback1056-completed-clock-pilot-freeze-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    passed=all(checks.values()),checks=checks,queue_sha256=sha(queuepath),checkpoint_sha256=sha(checkpoint),
    behavior_review_sha256=sha(reviewpath),native_verification_sha256=sha(nativepath),
    event_review_sha256=sha(eventpath),plan_revision_sha256=sha(revisionpath),
    optimizer_updates_performed=32,critic_only_updates=10,actor_updates=22,source_count=128,
    behavior_qualified=False,hardware_motion_started=False)
if not receipt['passed']:
    print(json.dumps(receipt,indent=2));raise SystemExit('Clock pilot is not ready to seal')
output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

"""Seal the completed bounded continuation and its four same-checkpoint replays."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
output = HERE / 'feedback960_queue_completion001.json'
if output.exists():
    raise SystemExit('Preserve the existing immutable completion receipt')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
queue_path = RUN / 'feedback960_queue001.json'
queue = read(queue_path)
review_path = HERE / 'feedback960_behavior_review001.json'
review = read(review_path)
native_path = HERE / 'feedback960_native_milestones.json'
native = read(native_path)
contract_path = HERE / 'continuation960_contract001.json'
contract = read(contract_path)
checkpoint = RUN / 'pilot018/model_000960.pt'
checks = dict(
    queue_complete=queue['status'] == 'complete',
    all_six_children_exit0=len(queue['history']) == 6 and all(r['exit_code'] == 0 for r in queue['history']),
    both_owners_exited=not any(Path('/proc/' + str(queue[k])).exists() for k in ('pid', 'worker_pid')),
    all123_frozen_sources_unchanged=len(queue['source_sha256']) == 123
        and all(sha(ROOT / n) == v for n, v in queue['source_sha256'].items()),
    exactly64_completed_optimizer_updates=queue['new_optimizer_updates'] == 64,
    native_restore_and_learning_milestones_passed=native['passed'] and native['all_required_milestones_present'],
    all_four_behavior_cohorts_complete=review['all_four_cohorts_complete'],
    exact_checkpoint_in_queue_and_review=sha(checkpoint) == queue['checkpoint_sha256'] == review['checkpoint_sha256'],
    unchanged_contract_and_prerequisites=sha(contract_path) == queue['continuation_contract_sha256']
        and all(sha(HERE / n) == v for n, v in contract['prerequisite_sha256'].items()),
    no_history_reset=(review['successful_updates_total'], review['actor_updates_total'],
        review['feedback_branch_actor_updates'], review['post_split_actor_updates'],
        review['inherited_shared_feedback_updates'], review['consumed_optimizer_updates_total'])
        == (960, 810, 432, 128, 304, 961))
receipt = dict(schema='flat-feedback960-completed-training-replay-freeze-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), checks=checks, passed=all(checks.values()),
    source_count=len(queue['source_sha256']), queue_sha256=sha(queue_path), checkpoint_sha256=sha(checkpoint),
    behavior_review_sha256=sha(review_path), native_verification_sha256=sha(native_path),
    continuation_contract_sha256=sha(contract_path), optimizer_updates_performed=64,
    behavior_qualified=False, hardware_motion_started=False)
if not receipt['passed']:
    print(json.dumps(receipt, indent=2))
    raise SystemExit('The native continuation is not ready to seal')
output.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))

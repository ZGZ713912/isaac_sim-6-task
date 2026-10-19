"""Authenticate a second unchanged-v13 short block after all896 replays finish."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/flat_base_20261019'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
target=HERE/'continuation960_contract001.json'
if target.exists() or (RUN/'pilot018').exists() or (RUN/'feedback960_queue001.json').exists():
    raise SystemExit('Preserve the existing960 contract or native run')
queue=read(RUN/'feedback896_queue001.json')
if queue['status']!='complete' or any(Path('/proc/'+str(queue[k])).exists() for k in ('pid','worker_pid')):
    raise SystemExit('Wait for the896 controller and worker to finish and exit')
completion=read(HERE/'feedback896_queue_completion001.json')
review=read(HERE/'feedback896_behavior_review001.json')
native=read(HERE/'feedback896_native_milestones.json')
cpu=read(HERE/'same_v13_real896_resume_cpu002.json')
assert completion['passed'] and review['all_four_cohorts_complete']
assert native['passed'] and native['all_required_milestones_present'] and cpu['passed']
assert completion['queue_sha256']==sha(RUN/'feedback896_queue001.json')
assert completion['behavior_review_sha256']==sha(HERE/'feedback896_behavior_review001.json')
assert all(sha(ROOT/n)==v for n,v in queue['source_sha256'].items())
progress=read(RUN/'pilot017/progress.json')
assert progress['status']=='block_complete' and progress['successful_updates_total']==896
assert progress['post_split_actor_updates']==64 and progress['critic_warmup_remaining']==0
assert len(progress['source_sha256'])==117
assert all(sha(ROOT/n)==v for n,v in progress['source_sha256'].items())
parent=RUN/'pilot017/model_000896.pt'
assert sha(parent)==completion['checkpoint_sha256']==review['checkpoint_sha256']==cpu['checkpoint_sha256']
plan=ROOT/'contracts/flat_base_finetune_v13.json'
assert sha(plan)==progress['plan_sha256']
prerequisites=['feedback896_queue_completion001.json','feedback896_behavior_review001.json',
 'feedback896_native_milestones.json','feedback896_training_event_review001.json',
 'feedback896_motion_window_audit001.json','static832_896_window_audit001.json',
 'same_v13_real896_resume_cpu002.json','separated_v13_production_tests001.json']
receipt=dict(schema='flat-unchanged-v13-short-continuation-contract-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
 parent_checkpoint_sha256=sha(parent),plan_file=str(plan.relative_to(ROOT)),plan_sha256=sha(plan),
 source_sha256=progress['source_sha256'],changed_plan_fields=[],changed_source_files=[],
 prerequisite_sha256={n:sha(HERE/n) for n in prerequisites},maximum_additional_optimizer_updates=64,
 additional_critic_warmup_updates=0,prior_consumed_optimizer_updates=897,
 parent_counters={k:progress[k] for k in ('successful_updates_total','actor_updates_total','active_branch_actor_updates',
 'feedback_branch_actor_updates','post_split_actor_updates','inherited_shared_feedback_updates','consumed_optimizer_updates_total')},
 semantics='Full unchanged v13 parent896 mean/std/critic/Adam/RNG/counters; no new fork, rate change, reward change, sampling change or warmup. Next64 updates are another short convergence experiment, followed by four paired replays. Previous regressions remain failures and are not waived.',
 decision='Test a second small block after middle/high standing improvement and default23cm improvement from11/22 to12/22 with lower forward3/backward3/start-stop3 errors. Low-height first-second displacement, stop speed, original-policy high-speed preservation and nominal-height reverse steady error remain failures; this is not approval for long training or deployment.',
 long_training_started=False,subsequent_training_queued=False,behavior_qualified=False,hardware_motion_started=False)
target.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_sha256','prerequisite_sha256')},indent=2))

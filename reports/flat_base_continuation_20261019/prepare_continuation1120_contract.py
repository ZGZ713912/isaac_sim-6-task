"""Authenticate an unchanged64-update v15 block after all1056 screens finish."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
target = HERE / 'continuation1120_contract001.json'
if target.exists() or (RUN/'pilot021').exists() or (RUN/'feedback1120_queue001.json').exists():
    raise SystemExit('Preserve the existing1120 contract or native run')
queue = read(RUN/'feedback1056_queue001.json')
if queue['status'] != 'complete' or any(Path('/proc/'+str(queue[k])).exists() for k in ('pid','worker_pid')):
    raise SystemExit('Wait for1056 queue and final native worker to finish and exit')
completion = read(HERE/'feedback1056_queue_completion001.json')
review = read(HERE/'feedback1056_behavior_review001.json')
native = read(HERE/'feedback1056_native_milestones.json')
assert completion['passed'] and review['all_three_cohorts_complete']
assert native['passed'] and native['all_required_milestones_present']
assert completion['queue_sha256'] == sha(RUN/'feedback1056_queue001.json')
assert completion['behavior_review_sha256'] == sha(HERE/'feedback1056_behavior_review001.json')
assert all(sha(ROOT/n)==v for n,v in queue['source_sha256'].items())
progress = read(RUN/'pilot020/progress.json')
assert progress['status']=='block_complete' and progress['successful_updates_total']==1056
assert progress['post_split_actor_updates']==204 and progress['critic_warmup_remaining']==0
assert len(progress['source_sha256'])==118
assert all(sha(ROOT/n)==v for n,v in progress['source_sha256'].items())
parent = RUN/'pilot020/model_001056.pt'
assert sha(parent)==completion['checkpoint_sha256']==review['checkpoint_sha256']=='92783dd1adcfb7cca9239d2e26d8e6c228ca6ffb34a5d694d88cc5e06d787004'
plan = ROOT/'contracts/flat_base_finetune_v15.json'
assert sha(plan)==progress['plan_sha256']
basic = read(RUN/'deployment1056_basic001/evaluation.json')
assert basic['status']=='evaluated' and basic['physics_hz']==1000
assert sum(c['passed'] for c in basic['result']['cases'].values())==7
assert all(basic['result']['cases'][n]['checks']['stand_drift'] for n in ('stand_230mm','stand_305mm','stand_430mm'))
prerequisites = ['feedback1056_queue_completion001.json','feedback1056_behavior_review001.json',
    'feedback1056_native_milestones.json','feedback1056_training_event_review001.json',
    'clock_v15_production_cpu001.json','clock_v15_adoption001.json']
receipt = dict(schema='flat-unchanged-v15-short-continuation-contract-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    parent_checkpoint_sha256=sha(parent),plan_file=str(plan.relative_to(ROOT)),plan_sha256=sha(plan),
    source_sha256=progress['source_sha256'],changed_plan_fields=[],changed_source_files=[],
    prerequisite_sha256={n:sha(HERE/n) for n in prerequisites},maximum_additional_optimizer_updates=64,
    additional_critic_warmup_updates=0,prior_consumed_optimizer_updates=1057,
    parent_counters={k:progress[k] for k in ('successful_updates_total','actor_updates_total','active_branch_actor_updates',
        'feedback_branch_actor_updates','post_split_actor_updates','inherited_shared_feedback_updates','consumed_optimizer_updates_total')},
    semantics='Full unchanged v15 parent1056 mean/std/critic/Adam/RNG/counters; no new fork, rate, reward, sampling, clock or warmup. '
        'Next64 actor/critic updates are a bounded convergence experiment; all failures remain recorded.',
    decision='1000Hz basic screen passed7/8 including all three10s stationary3cm targets. Backward0.5m/s MAE.06208 remains above.05. '
        'Keep the aligned timing and all objective settings; test another small64-update block before considering scale-up. '
        'RMCS-default full range improved8 to11/22 with all three static3cm targets passing, but reverse-yaw4 translation.02943 to.05958m/s and reverse-curve.03893 to.05823m/s lost checks. Source forward3 yaw remains failed. '
        'The complete three-cohort1056 review is bound above; these regressions remain repair targets. No behavior qualification or long-run authorization inferred.',
    long_training_started=False,subsequent_training_queued=False,behavior_qualified=False,hardware_motion_started=False)
target.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_sha256','prerequisite_sha256')},indent=2))

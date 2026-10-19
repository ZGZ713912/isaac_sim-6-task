"""Freeze a declared sampling-only continuation after both native replay arms exit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import torch

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).parent
RUN = ROOT/'runs/flat_base_20261019'
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
revision_path = REPORT/'training_plan_revision012.json'
if revision_path.exists():
    raise SystemExit('Revision012 already exists')
queue=json.loads((RUN/'default_height640_queue001.json').read_text())
if queue['status']!='complete' or any(Path('/proc/'+str(queue[k])).exists() for k in ('pid','worker_pid')):
    raise SystemExit('Both default-height arms must complete and the queue must exit')
paired_path=REPORT/'feedback640_default_height_review001.json'
paired=json.loads(paired_path.read_text())
if not paired['pair_complete'] or not paired['all_source_and_checkpoint_bindings_valid']:
    raise SystemExit('Completed paired default-height review is missing')
checkpoint=RUN/'pilot013/model_000640.pt'
if sha(checkpoint)!='3d0806ceb77ea7e12de6fbcbcdc9c6ab0f39dd0fc8d53ac7340ca8d65043fbe2':
    raise SystemExit('Parent640 checkpoint changed')
info=torch.load(checkpoint,map_location='cpu',weights_only=True)['infos']
old_file='contracts/flat_base_finetune_v9.json'
new_file='contracts/flat_base_finetune_v10.json'
old,new=[json.loads((ROOT/n).read_text()) for n in (old_file,new_file)]
audit=json.loads((REPORT/'sampling_candidate_v10_audit001.json').read_text())
if sha(ROOT/new_file)!=audit['candidate_sha256'] or sha(ROOT/old_file)!=info['plan_sha256']:
    raise SystemExit('The reviewed candidate or parent plan changed')
changed=sorted(k for k in set(old)|set(new) if old.get(k)!=new.get(k))
if changed!=['group_episode_seconds','schema','train_groups','training_scope']:
    raise SystemExit('Expected only the declared sampling revision')
old_sources=info['source_sha256']
new_sources={n:sha(ROOT/n) for n in old_sources}
changed_sources=sorted(n for n in old_sources if old_sources[n]!=new_sources[n])
if changed_sources!=['src/wheeled_tasks/chassis/flat_base_continuation.py']:
    raise SystemExit('Unexpected training source change: '+str(changed_sources))
from wheeled_tasks.chassis.flat_base_continuation import read_plan
read_plan(ROOT,new_file)
archive=REPORT/'training_plan_revision012_source'
archive.mkdir(exist_ok=False)
for name in new_sources:
    target=archive/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/name,target)
report=dict(schema='flat-training-plan-revision-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    parent_checkpoint_sha256=sha(checkpoint),old_plan_file=old_file,new_plan_file=new_file,
    old_plan_sha256=sha(ROOT/old_file),new_plan_sha256=sha(ROOT/new_file),old_source_sha256=old_sources,
    new_source_sha256=new_sources,changed_plan_fields=changed,changed_source_files=changed_sources,
    semantics='Exact parent640 mean weights/Adam/std/critic/RNG/counters/consumption; no gate or command composition change; moving heights .305/.23/.42 with probabilities .5/.25/.25; +/-3 m/s split 5 long 20 s plus 5 short 2 s per direction; static population/reward unchanged',
    additional_critic_warmup_updates=20,maximum_additional_optimizer_updates=64,
    prior_consumed_optimizer_updates=info['consumed_optimizer_updates_total'],behavior_qualified=False,
    default_height_paired_review_sha256=sha(paired_path),sampling_candidate_audit_sha256=sha(REPORT/'sampling_candidate_v10_audit001.json'),
    native_restore_and_learning_verification='pending actual641/660/661/704 milestones; not claimed by this source receipt')
revision_path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps(dict(revision_sha256=sha(revision_path),new_plan_sha256=report['new_plan_sha256'],source_files=len(new_sources),changed_fields=changed,changed_sources=changed_sources,maximum_additional_updates=64)))

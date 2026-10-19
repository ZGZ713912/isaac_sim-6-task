"""After sealed height coverage, adopt the reviewed entry and probe two serial populations."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/flat_base_20261019'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
statepath=RUN/'population_capacity_queue001.json'
if statepath.exists():raise SystemExit('Preserve existing capacity queue')
qpath=RUN/'feedback1248_height_coverage_queue001.json'
sealpath=HERE/'feedback1248_height_coverage_completion001.json'
previous=read(qpath);seal=read(sealpath)
if (previous['status']!='complete' or not seal['passed'] or seal['queue_sha256']!=sha(qpath)
        or any(Path('/proc/'+str(previous[k])).exists() for k in ('pid','worker_pid'))):
    raise SystemExit('Require completed sealed height queue and all its owners out')
if not all(sha(ROOT/n)==v for n,v in previous['source_sha256'].items()):
    raise SystemExit('Do not adopt on top of an unreviewed source delta')
candidate=HERE/'population_transition_candidate002/train_flat_base.py'
expected='bd51d84077d3eff4405293f65f5da6b8986e95404408ae07438bc237fe8b481a'
if sha(candidate)!=expected or not read(HERE/'population_transition_candidate001/cpu001.json')['passed']:
    raise SystemExit('Candidate or CPU evidence changed')
checkpoint=RUN/'pilot023/model_001248.pt'
if sha(checkpoint)!=previous['checkpoint_sha256']:raise SystemExit('Checkpoint changed')
import torch
info=torch.load(checkpoint,map_location='cpu',weights_only=True)['infos']
entry='scripts/train_flat_base.py'
old_sources=info['source_sha256']
if any(sha(ROOT/n)!=v for n,v in old_sources.items()):raise SystemExit('Actual parent source mismatch')
new_sources={**old_sources,entry:expected}
old_plan=ROOT/info['plan_file']
plan=HERE/'population_coverage_candidate001/plan_4096.json'
old=read(old_plan);new=read(plan)
if {k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}!={'allowed_num_envs','default_num_envs'}:
    raise SystemExit('Capacity plan must change population only')
if any((RUN/name).exists() or (RUN/(name+'.log')).exists() for name in ('capacity512_001','capacity4096_001')):
    raise SystemExit('Preserve all previous capacity outputs')
entry_revision=HERE/'population_capacity_entry_revision019.json'
plan_revision=HERE/'population_capacity_plan_revision019.json'
if entry_revision.exists() or plan_revision.exists():raise SystemExit('Preserve exact revision receipts')
entry_revision.write_text(json.dumps(dict(schema='flat-trainer-entry-revision-v1',
    parent_checkpoint_sha256=sha(checkpoint),old_entry_sha256=old_sources[entry],new_entry_sha256=expected,
    reason='Height diagnostics, authenticated population-only migration with cumulative frames, isolated capacity mode',
    behavior_qualified=False),indent=2)+'\n')
plan_revision.write_text(json.dumps(dict(schema='flat-training-plan-revision-v1',
    parent_checkpoint_sha256=sha(checkpoint),old_plan_file=str(old_plan.relative_to(ROOT)),
    old_plan_sha256=sha(old_plan),new_plan_sha256=sha(plan),
    old_source_sha256=old_sources,new_source_sha256=new_sources,
    changed_plan_fields=['allowed_num_envs','default_num_envs'],
    population_transition=dict(schema='flat-training-population-transition-v1',old_num_envs=512,
        new_num_envs=4096,num_steps_per_env=96,inherited_successful_updates=1248,
        inherited_successful_training_frames=61341696,additional_critic_warmup_updates=0)),indent=2)+'\n')
# All prior native users are out; preserve the byte-identical old entry in the candidate evidence.
(ROOT/entry).write_bytes(candidate.read_bytes())
state=dict(schema='flat-native-population-capacity-queue-v1',pid=os.getpid(),status='starting',
    started_at=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=sha(checkpoint),
    prior_height_completion_sha256=sha(sealpath),queue_entry_sha256=sha(Path(__file__)),
    old_entry_sha256=old_sources[entry],new_entry_sha256=expected,
    plan_sha256=sha(plan),history=[],new_optimizer_updates=0,subsequent_training_queued=False,
    hardware_motion_started=False,scope='Two isolated engineering capacity probes; no learned parameters or new training checkpoint')
def save():
    tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');tmp.replace(statepath)
def resources():
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    used,free=map(int,result.stdout.splitlines()[0].split(','))
    available=next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    return dict(gpu_used_mib=used,gpu_free_mib=free,available_ram_kib=available)
try:
    save()
    for count,selected,revision,flag in ((512,old_plan,entry_revision,'--resume-entry-revision'),
                                       (4096,plan,plan_revision,'--resume-plan-revision')):
        name=f'capacity{count}_001';state.update(status='running',phase=name);save()
        command=[sys.executable,str(ROOT/entry),'--plan',str(selected.relative_to(ROOT)),
            '--output',str(RUN/name),'--updates','1','--num-envs',str(count),'--resume',str(checkpoint),
            flag,str(revision),'--capacity-probe']
        samples=[];started=time.monotonic()
        with (RUN/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state['worker_pid']=child.pid;save()
            while child.poll() is None:
                sample=resources();samples.append(sample)
                if sample['gpu_free_mib']<512 or sample['available_ram_kib']<2*1024**2 or time.monotonic()-started>600:
                    child.terminate();state['guard_stop']=sample;save();child.wait(timeout=60);break
                time.sleep(2)
        path=RUN/name/'progress.json';data=read(path) if path.exists() else {}
        state['history'].append(dict(phase=name,num_envs=count,exit_code=child.returncode,
            progress_sha256=sha(path) if path.exists() else None,
            status=data.get('status'),capacity=data.get('capacity_probe'),
            peak_gpu_used_mib=max(s['gpu_used_mib'] for s in samples),
            min_available_ram_kib=min(s['available_ram_kib'] for s in samples)))
        save()
        if child.returncode or data.get('status')!='capacity_complete' or not data.get('capacity_probe',{}).get('passed'):
            raise RuntimeError('Native capacity child failed; preserve its output and do not queue PPO')
    state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat())
except Exception:
    state.update(status='failed',error=traceback.format_exc(),finished_at=datetime.now(timezone.utc).isoformat());raise
finally:
    save()

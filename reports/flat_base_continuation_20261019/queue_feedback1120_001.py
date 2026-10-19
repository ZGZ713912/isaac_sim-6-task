"""One64-update unchanged-v15 pilot and three serial development replays."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/flat_base_20261019'
HERE = Path(__file__).resolve().parent
STATE = RUN / 'feedback1120_queue001.json'
PYTHON = '/home/noir/miniconda3/envs/isaacsim6/bin/python'
phases = ('pilot021','feedback1120_native_verification001','deployment1120_basic001',
    'rmcs_default1120_envelope001','feedback1120_basic001')
if STATE.exists() or any((RUN/n).exists() or (RUN/(n+'.log')).exists() for n in phases):
    raise SystemExit('Preserve every existing same-plan-pilot output and log')
lock = (RUN/'.feedback1120_queue001.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
revision_path = HERE/'continuation1120_contract001.json'
revision = read(revision_path)
hashes = dict(revision['source_sha256'])
for n in ('scripts/evaluate_flat_base.py','scripts/compare_flat_base.py','scripts/export_flat_base.py',
          'contracts/flat_base_finetune_v15.json','scripts/flat_default_height_profile.py',
          'scripts/flat_maximum_height_profile.py','reports/flat_base_continuation_20261019/verify_feedback1120_native.py'):
    hashes[n] = sha(ROOT/n)
hashes.update(read(HERE/'clock_candidate_v15_support001.json')['unchanged_dependency_sha256'])
state = dict(schema='flat-feedback1120-same-plan-v15-serial-queue-v1',pid=os.getpid(),status='starting',
    started_at=datetime.now(timezone.utc).isoformat(),source_sha256=hashes,
    parent_checkpoint_sha256=revision['parent_checkpoint_sha256'],continuation_contract_sha256=sha(revision_path),
    maximum_additional_optimizer_updates=64,expected_critic_only_updates=0,expected_actor_updates=64,
    new_optimizer_updates=0,subsequent_training_queued=False,hardware_motion_started=False,history=[],
    screening_scope='Unchanged64-update pilot: raw1000Hz eight cases, RMCS-default1000Hz full envelope, raw200Hz eight cases. '
        'Nominal-height full envelope and other release cohorts remain unassessed for this checkpoint.')


def save():
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n')
    temp.replace(STATE)


def run(phase,command):
    if any(sha(ROOT/n)!=v for n,v in hashes.items()):
        raise ValueError('Frozen unchanged-v15 pilot source changed')
    if phase not in phases or (RUN/(phase+'.log')).exists():
        raise ValueError('Unexpected phase or existing output log')
    state.update(status='running',phase=phase)
    with (RUN/(phase+'.log')).open('x') as log:
        worker = subprocess.Popen([PYTHON,*command],cwd=ROOT,
            env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES'},stdout=log,stderr=subprocess.STDOUT)
        state['worker_pid']=worker.pid
        save()
        code=worker.wait()
    state['history'].append(dict(phase=phase,exit_code=code))
    progress=RUN/'pilot021/progress.json'
    if progress.exists():
        state['new_optimizer_updates']=read(progress)['successful_updates_total']-1056
    save()
    if code:
        raise RuntimeError('Same-plan pilot child failed: '+phase)


try:
    save()
    old=read(RUN/'feedback1056_queue001.json')
    complete=read(HERE/'feedback1056_queue_completion001.json')
    if (old['status']!='complete' or not complete['passed'] or complete['queue_sha256']!=sha(RUN/'feedback1056_queue001.json')
            or any(Path('/proc/'+str(old[k])).exists() for k in ('pid','worker_pid'))):
        raise ValueError('The1056 native queue must finish, exit and be sealed')
    for n,v in revision['prerequisite_sha256'].items():
        if sha(HERE/n)!=v:
            raise ValueError('Authenticated prerequisite changed: '+n)
    parent=RUN/'pilot020/model_001056.pt'
    if sha(parent)!=revision['parent_checkpoint_sha256']:
        raise ValueError('Authenticated1056 parent changed')
    run('pilot021',['scripts/train_flat_base.py','--plan','contracts/flat_base_finetune_v15.json',
        '--output',str(RUN/'pilot021'),'--resume',str(parent),'--num-envs','512','--updates','64'])
    run('feedback1120_native_verification001',[str(HERE/'verify_feedback1120_native.py')])
    proof=read(HERE/'feedback1120_native_milestones.json')
    if not proof['passed'] or not proof['all_required_milestones_present']:
        raise ValueError('Actual native timing/full-state/learning checks failed')
    checkpoint=RUN/'pilot021/model_001120.pt'
    state['checkpoint_sha256']=sha(checkpoint)
    specs=[('deployment1120_basic001',['--deployment-pd'],[('source','deployment_source001'),('parent1056','deployment1056_basic001')]),
        ('rmcs_default1120_envelope001',['--deployment-pd','--rmcs-command-reference','--packaged-default-height','--envelope'],
            [('source','rmcs_default_source_envelope001'),('parent1056','rmcs_default1056_envelope001')]),
        ('feedback1120_basic001',[],[('source','baseline_native002'),('parent1056','feedback1056_basic001')])]
    sys.path.insert(0,str(ROOT/'scripts'))
    from compare_flat_base import compare
    for phase,flags,references in specs:
        run(phase,['scripts/evaluate_flat_base.py','--checkpoint',str(checkpoint),'--output',str(RUN/phase),*flags])
        path=RUN/phase/'evaluation.json'
        data=read(path)
        if data['status']!='evaluated' or data['checkpoint_sha256']!=state['checkpoint_sha256']:
            raise ValueError('Replay did not finish against the1120 checkpoint')
        for label,reference in references:
            ref=RUN/reference/'evaluation.json'
            result=compare(read(ref),data)
            result.update(baseline_sha256=sha(ref),candidate_sha256=sha(path))
            (HERE/('comparison_'+phase+'_'+label+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat())
    save()
except Exception:
    state.update(status='failed',error=traceback.format_exc())
    save()
    raise

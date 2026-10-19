"""Resume read-only verification/replay after the preserved midpoint-number error."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/flat_base_20261019'
REPORTS=ROOT/'reports/flat_base_continuation_20261019'
STATE=RUN/'feedback768_replay_queue002.json'
PYTHON='/home/noir/miniconda3/envs/isaacsim6/bin/python'
if STATE.exists():
    raise SystemExit('Replay already has an owner receipt')
lock=(RUN/'.feedback768_replay_queue002.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent_path=RUN/'feedback768_queue001.json'
parent=json.loads(parent_path.read_text())
hashes=dict(parent['source_sha256'])
for name in ('reports/flat_base_continuation_20261019/verify_feedback768_native_v2.py',
             'reports/flat_base_continuation_20261019/feedback768_midpoint_schedule_correction001.json'):
    hashes[name]=sha(ROOT/name)
state=dict(schema='flat-feedback768-replay-only-queue-v2',pid=os.getpid(),status='waiting',
    waiting_for_pid=parent['pid'],source_sha256=hashes,maximum_additional_optimizer_updates=0,
    new_optimizer_updates=0,inherited_pilot_optimizer_updates=64,subsequent_training_queued=False,
    started_at=datetime.now(timezone.utc).isoformat(),hardware_motion_started=False,history=[])


def save():
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(STATE)


def run(phase,command):
    if any(sha(ROOT/name)!=value for name,value in hashes.items()):
        raise ValueError('Frozen replay source changed')
    state.update(status='running',phase=phase)
    with (RUN/(phase+'.log')).open('w') as log:
        worker=subprocess.Popen([PYTHON,*command],cwd=ROOT,env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES'},
                                stdout=log,stderr=subprocess.STDOUT)
        state['worker_pid']=worker.pid;save();code=worker.wait()
    state['history'].append(dict(phase=phase,exit_code=code));save()
    if code:
        raise RuntimeError('Replay child failed: '+phase)


try:
    save()
    while Path('/proc/'+str(parent['pid'])).exists():
        time.sleep(2)
    parent=json.loads(parent_path.read_text())
    if (parent['status']!='failed' or parent.get('phase')!='feedback768_native_verification001'
            or parent['history']!=[dict(phase='pilot015',exit_code=0),
                                  dict(phase='feedback768_native_verification001',exit_code=1)]
            or any(Path('/proc/'+str(parent[k])).exists() for k in ('pid','worker_pid'))):
        raise ValueError('Replay may only follow the specific completed pilot/missing736 verification failure')
    old_receipt=REPORTS/'feedback768_native_milestones.json'
    old=json.loads(old_receipt.read_text())
    if (not old['passed'] or old['all_required_milestones_present']
            or set(old['milestones'])!={'705','768'}):
        raise ValueError('The prior failure is not isolated to the unavailable736 checkpoint')
    progress=json.loads((RUN/'pilot015/progress.json').read_text())
    if progress['status']!='block_complete' or progress['successful_updates_total']!=768:
        raise ValueError('Full64-update pilot has not completed')
    state.update(parent_failure_queue_sha256=sha(parent_path),
                 preserved_missing736_receipt_sha256=sha(old_receipt),
                 actual_midpoint_checkpoint_sha256=sha(RUN/'pilot015/model_000737.pt'))
    checkpoint=RUN/'pilot015/model_000768.pt'
    state['checkpoint_sha256']=sha(checkpoint);save()
    run('feedback768_native_verification002',[str(REPORTS/'verify_feedback768_native_v2.py')])
    proof=json.loads((REPORTS/'feedback768_native_milestones_v2.json').read_text())
    if not proof['passed'] or not proof['all_required_milestones_present']:
        raise ValueError('Corrected actual705/737/768 verification failed')
    phases=[('feedback768_basic001',[],'baseline_native002','feedback704_basic001'),
            ('feedback768_envelope001',['--envelope'],'envelope_source002','feedback704_envelope001'),
            ('rmcs768_envelope001',['--deployment-pd','--rmcs-command-reference','--envelope'],'rmcs_source_envelope001','rmcs704_envelope001'),
            ('rmcs_default768_envelope001',['--deployment-pd','--rmcs-command-reference','--packaged-default-height','--envelope'],'rmcs_default_source_envelope001','rmcs_default704_envelope001')]
    sys.path.insert(0,str(ROOT/'scripts'))
    from compare_flat_base import compare
    for phase,flags,original,previous in phases:
        if (RUN/phase).exists():
            raise ValueError('Preserve an existing replay directory: '+phase)
        run(phase,['scripts/evaluate_flat_base.py','--checkpoint',str(checkpoint),'--output',str(RUN/phase),*flags])
        candidate_path=RUN/phase/'evaluation.json';candidate=json.loads(candidate_path.read_text())
        if candidate.get('status')!='evaluated' or candidate['checkpoint_sha256']!=state['checkpoint_sha256']:
            raise ValueError('Replay did not complete with the authenticated768 checkpoint')
        for label,baseline in (('source',original),('parent704',previous)):
            path=RUN/baseline/'evaluation.json'
            result=compare(json.loads(path.read_text()),candidate)
            result.update(baseline_sha256=sha(path),candidate_sha256=sha(candidate_path))
            (REPORTS/('comparison_'+phase+'_'+label+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());save()
except Exception:
    state.update(status='failed',error=traceback.format_exc());save();raise

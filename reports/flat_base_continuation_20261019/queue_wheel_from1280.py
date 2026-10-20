"""Run only16 v6 wheel-head updates then the three unchanged native cohorts."""
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

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
STATE = RUN / 'wheel_from1280_queue001.json'
CONTRACT = HERE / 'wheel_from1280_contract001.json'
PHASES = ('pilot027_from1280','wheel_from1280_native_final','wheel_from1280_basic001',
    'wheel_from1280_default001','wheel_from1280_maximum001')
if STATE.exists() or CONTRACT.exists() or any((RUN/p).exists() or (RUN/(p+'.log')).exists() for p in PHASES):
    raise SystemExit('Preserve prior trial outputs')
prior_path = RUN / 'static_teacher0_1296_queue001.json'
seal_path = HERE / 'static_teacher0_1296_completion001.json'
prior, seal = read(prior_path), read(seal_path)
if (prior['status'] != 'complete' or not seal['passed'] or seal['queue_sha256'] != sha(prior_path)
        or any(Path('/proc/'+str(prior[k])).exists() for k in ('pid','worker_pid'))):
    raise SystemExit('Wait for all previous owners to exit')
adoption_path = HERE / 'wheel_authority_from1280_adoption023.json'
adoption = read(adoption_path)
parent, plan = ROOT/adoption['parent_checkpoint'], ROOT/adoption['plan_file']
if not adoption['passed'] or sha(parent) != adoption['parent_checkpoint_sha256'] or sha(plan) != adoption['plan_sha256']:
    raise SystemExit('Exact source-bound composition is required')
function_path = HERE / 'wheel_authority_candidate001/composed002_function_cpu003.json'
function = read(function_path)
preflight_path = HERE / 'wheel_authority_candidate001/evaluator_preflight003/evaluation.json'
if (not function['passed'] or not all(function['checks'].values())
        or function['parent_checkpoint_sha256'] != sha(parent)
        or read(preflight_path)['status'] != 'static_preflight_passed'):
    raise SystemExit('Actual-parent CPU function and evaluator preflight are required')
verifier = HERE / 'verify_wheel_from1280_native.py'
hashes = dict(adoption['new_source_sha256'])
hashes.update(adoption['integration_source_sha256'])
for p in (Path(__file__),verifier,plan,adoption_path):
    hashes[str(p.relative_to(ROOT))] = sha(p)
if any(sha(ROOT/n) != value for n,value in hashes.items()):
    raise SystemExit('Frozen execution sources changed')
lock = (RUN/'.wheel_from1280.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
contract = dict(schema='flat-wheel-from1280-bounded-contract-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    parent_checkpoint_sha256=sha(parent),adoption_sha256=sha(adoption_path),source_sha256=hashes,
    actual_parent_function_CPU_sha256=sha(function_path),evaluator_preflight_sha256=sha(preflight_path),
    optimizer_updates=16,actor_updates=16,extra_critic_warmup=0,
    expected_final_successful_updates=1296,expected_final_lineage_consumption=1297,
    lineage_consumption_at_entry=1281,prior_sibling1296_trial_consumption=16,
    sibling1296_trial_used_as_learning_parent=False,prior_sibling_consumption_retained=True,
    origin1280_and_both_subsequent_trial_consumption_at_exit=1313,
    inherited_successful_frames=73924608,additional_successful_frames=6291456,expected_final_successful_frames=80216064,
    changed='Only994 newly appended high-demand/braking wheel parameters learn; existing actor branches and Adam stay frozen.',
    evaluation='Serial basic,23cm full22,42cm full22; original thresholds/seeds/horizons. No subsequent PPO queued.',
    regression_reference='Original1280 native replays and static-recall1280 native composition;1296 remains rejected.',
    behavior_qualified=False,long_training_started=False,heldout_seed_used=False,hardware_motion_started=False)
CONTRACT.write_text(json.dumps(contract,indent=2,allow_nan=False)+'\n')
state = dict(schema='flat-wheel-from1280-serial-queue-v1',status='starting',pid=os.getpid(),
    started_at=datetime.now(timezone.utc).isoformat(),contract_sha256=sha(CONTRACT),source_sha256=hashes,
    parent_checkpoint_sha256=sha(parent),history=[],additional_optimizer_updates=16,
    prior_sibling_consumption_retained=True,subsequent_training_queued=False,
    heldout_seed_used=False,hardware_motion_started=False)

def save():
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    tmp.replace(STATE)

def frozen():
    if any(sha(ROOT/n) != value for n,value in hashes.items()):
        raise ValueError('Frozen execution source changed')

def child(name, command, watch=False):
    frozen()
    state.update(status='running',phase=name)
    with (RUN/(name+'.log')).open('x') as log:
        worker = subprocess.Popen(command,cwd=ROOT,env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES'},
            stdout=log,stderr=subprocess.STDOUT)
        state['worker_pid'] = worker.pid
        save()
        checked = False
        while worker.poll() is None:
            if watch and not checked and (RUN/'pilot027_from1280/model_001281.pt').exists():
                with (RUN/'wheel_from1280_native_first.log').open('x') as first:
                    code = subprocess.run([sys.executable,str(verifier),'--first-only'],cwd=ROOT,
                        stdout=first,stderr=subprocess.STDOUT).returncode
                state['first_native_verification_exit_code'] = code
                save()
                if code:
                    worker.terminate()
                    worker.wait(timeout=60)
                    raise RuntimeError('First native audit failed; stopped only this worker')
                checked = True
            time.sleep(2)
        state['history'].append(dict(phase=name,exit_code=worker.returncode))
        save()
    if worker.returncode:
        raise RuntimeError('Child failed; no later work launched: '+name)
    frozen()

try:
    save()
    child('pilot027_from1280',[sys.executable,str(ROOT/'scripts/train_flat_base.py'),'--plan',str(plan.relative_to(ROOT)),
        '--output',str(RUN/'pilot027_from1280'),'--updates','16','--num-envs','4096','--resume',str(parent)],watch=True)
    final = RUN/'pilot027_from1280/model_001296.pt'
    state['checkpoint_sha256'] = sha(final)
    child('wheel_from1280_native_final',[sys.executable,str(verifier)])
    for name,flags in (('wheel_from1280_basic001',[]),
        ('wheel_from1280_default001',['--rmcs-command-reference','--packaged-default-height','--envelope']),
        ('wheel_from1280_maximum001',['--rmcs-command-reference','--packaged-maximum-height','--envelope'])):
        child(name,[sys.executable,str(ROOT/'scripts/evaluate_flat_base.py'),'--seed','191029','--checkpoint',str(final),
            '--deployment-pd',*flags,'--output',str(RUN/name)])
        evaluation = read(RUN/name/'evaluation.json')
        cases = evaluation['result']['cases']
        state['history'][-1].update(evaluation_sha256=sha(RUN/name/'evaluation.json'),
            passed_cases=sum(c['passed'] for c in cases.values()),requested_cases=len(cases),
            full_horizon_episodes=sum(c['full_horizon_episodes'] for c in cases.values()),
            requested_episodes=sum(c['requested_episodes'] for c in cases.values()))
        save()
    state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat())
except Exception:
    state.update(status='failed',error=traceback.format_exc(),finished_at=datetime.now(timezone.utc).isoformat())
    raise
finally:
    save()

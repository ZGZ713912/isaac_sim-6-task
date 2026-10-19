"""Sixteen full-state population-only updates, then three matched development replays."""
from datetime import datetime,timezone
import fcntl,hashlib,json,os
from pathlib import Path
import subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;RUN=ROOT/'runs/flat_base_20261019'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
statepath=RUN/'feedback1264_population_queue001.json';contractpath=HERE/'continuation1264_population_contract001.json'
if statepath.exists() or contractpath.exists() or (RUN/'pilot024').exists():
    raise SystemExit('Preserve all prior training attempts and receipts')
capacity_path=HERE/'population_capacity_completion002.json';capacity=read(capacity_path)
capacity_queue_path=RUN/'population_capacity_queue002.json';capacity_queue=read(capacity_queue_path)
if (not capacity['passed'] or capacity_queue['status']!='complete' or capacity['queue_sha256']!=sha(capacity_queue_path)
        or any(Path('/proc/'+str(capacity_queue[k])).exists() for k in ('pid','worker_pid'))):
    raise SystemExit('Both isolated capacity probes must be sealed and their owners out')
lock=(RUN/'.feedback1264_population_queue001.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
height_path=HERE/'feedback1248_height_coverage_completion001.json';height=read(height_path)
if not height['passed']:raise SystemExit('The complete parent height development coverage is missing')
checkpoint=RUN/'pilot023/model_001248.pt';plan=HERE/'population_coverage_candidate001/plan_4096.json'
revision=HERE/'population_capacity_plan_revision019.json';verifier=HERE/'verify_feedback1264_population_native.py'
hashes=dict(read(RUN/'feedback1248_height_coverage_queue001.json')['source_sha256'])
hashes['scripts/train_flat_base.py']=capacity['entry_sha256']
for p in (plan,verifier):hashes[str(p.relative_to(ROOT))]=sha(p)
if any(sha(ROOT/n)!=v for n,v in hashes.items()) or sha(checkpoint)!=capacity['checkpoint_sha256']:
    raise SystemExit('The exact reviewed sources, plan and parent must be present')
specs=(
    ('deployment1264_basic001',['--deployment-pd'],'deployment_source001','deployment1248_basic001'),
    ('rmcs_default1264_envelope001',['--deployment-pd','--rmcs-command-reference','--packaged-default-height','--envelope'],
        'rmcs_default_source_envelope001','rmcs_default1248_envelope001'),
    ('rmcs_maximum1264_envelope001',['--deployment-pd','--rmcs-command-reference','--packaged-maximum-height','--envelope'],
        'rmcs_maximum_source_envelope001','rmcs_maximum1248_envelope001'))
for name,flags,baseline,parent in specs:
    if (RUN/name).exists() or (RUN/(name+'.log')).exists():raise SystemExit('Preserve every existing replay')
    if not all((RUN/n/'evaluation.json').is_file() for n in (baseline,parent)):
        raise SystemExit('The exact source and parent fixed cohorts must already exist')
contract=dict(schema='flat-population-only-continuation-contract-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),parent_checkpoint_sha256=sha(checkpoint),
    capacity_completion_sha256=sha(capacity_path),parent_height_completion_sha256=sha(height_path),
    parent_behavior_review_sha256=sha(HERE/'feedback1248_behavior_review001.json'),
    parent_maximum_height_readout_sha256=sha(HERE/'feedback1248_maximum_height_readout001.json'),
    plan_file=str(plan.relative_to(ROOT)),plan_sha256=sha(plan),revision_sha256=sha(revision),
    source_sha256=hashes,additional_optimizer_updates=16,additional_actor_updates=16,
    additional_critic_only_updates=0,num_envs=4096,num_steps_per_env=96,
    inherited_successful_training_frames=61341696,additional_successful_training_frames=6291456,
    expected_final_successful_training_frames=67633152,expected_final_successful_updates=1264,
    expected_final_consumed_optimizer_updates=1265,
    unchanged='Actor architecture/weights at entry, Adam/std/critic/RNG restoration, reward, groups/shares, episode horizons, 50Hz policy/1000Hz physics/fresh-feedback PD, rates, PPO epochs/minibatches and acceptance thresholds',
    reason='Measure a larger, less sparse population with height-stratified training observations; this is a bounded trial, not an established causal fix',
    subsequent_training_queued=False,large_5000_or10000_started=False,heldout_seed_used=False,hardware_motion_started=False)
contractpath.write_text(json.dumps(contract,indent=2,allow_nan=False)+'\n')
state=dict(schema='flat-feedback1264-population-queue-v1',pid=os.getpid(),status='starting',
    started_at=datetime.now(timezone.utc).isoformat(),parent_checkpoint_sha256=sha(checkpoint),
    contract_sha256=sha(contractpath),queue_entry_sha256=sha(Path(__file__)),source_sha256=hashes,history=[],
    additional_optimizer_updates=16,subsequent_training_queued=False,heldout_seed_used=False,hardware_motion_started=False)
def save():
    tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');tmp.replace(statepath)
def frozen():
    if any(sha(ROOT/n)!=v for n,v in hashes.items()):raise RuntimeError('Frozen execution file changed')
def child(name,command,watch_first=False):
    frozen();state.update(status='running',phase=name);save()
    with (RUN/(name+'.log')).open('x') as log:
        worker=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES'},stdout=log,stderr=subprocess.STDOUT)
        state['worker_pid']=worker.pid;save();first_done=False
        while worker.poll() is None:
            if watch_first and not first_done and (RUN/'pilot024/model_001249.pt').exists():
                with (RUN/'feedback1264_native_first.log').open('x') as firstlog:
                    result=subprocess.run([sys.executable,str(verifier),'--first-only'],cwd=ROOT,stdout=firstlog,stderr=subprocess.STDOUT)
                state['first_native_verification_exit_code']=result.returncode;save()
                if result.returncode:
                    worker.terminate();worker.wait(timeout=60);raise RuntimeError('First native state proof failed; stopped this worker')
                first_done=True
            time.sleep(2)
        state['history'].append(dict(phase=name,exit_code=worker.returncode));save()
    if worker.returncode:raise RuntimeError(name+' failed; preserve all data and do not queue later work')
    frozen()
sys.path.insert(0,str(ROOT/'scripts'));from compare_flat_base import compare
try:
    save()
    child('pilot024',[sys.executable,str(ROOT/'scripts/train_flat_base.py'),'--plan',str(plan.relative_to(ROOT)),
        '--output',str(RUN/'pilot024'),'--updates','16','--num-envs','4096','--resume',str(checkpoint),
        '--resume-plan-revision',str(revision)],watch_first=True)
    final=RUN/'pilot024/model_001264.pt'
    state['checkpoint_sha256']=sha(final);save()
    child('feedback1264_native_final',[sys.executable,str(verifier)])
    for name,flags,source,parent in specs:
        child(name,[sys.executable,str(ROOT/'scripts/evaluate_flat_base.py'),'--seed','191029','--output',str(RUN/name),
            '--checkpoint',str(final),*flags])
        path=RUN/name/'evaluation.json';data=read(path);comparisons={}
        for label,reference in (('source',source),('parent1248',parent)):
            old=read(RUN/reference/'evaluation.json');result=compare(old,data);relative=[]
            for case_name,new_case in data['result']['cases'].items():
                vx,yaw,_=new_case['command'];prior=old['result']['cases'][case_name]
                if abs(vx)>=.75 or abs(yaw)>=1.5:
                    for metric,tolerance in (('vx_mae_m_s',.02),('yaw_mae_rad_s',.05)):
                        if new_case[metric]>prior[metric]+tolerance:
                            relative.append(dict(case=case_name,metric=metric,before=prior[metric],after=new_case[metric],tolerance=tolerance))
            output=HERE/(f'comparison_{name}_{label}.json');output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            comparisons[label]=dict(path=str(output.relative_to(ROOT)),sha256=sha(output),
                lost_checks=result['lost_baseline_checks'],relative_regressions=relative)
        cases=data['result']['cases'];state['history'][-1].update(evaluation_sha256=sha(path),
            passed_cases=sum(c['passed'] for c in cases.values()),requested_cases=len(cases),
            full_horizon_episodes=sum(c['full_horizon_episodes'] for c in cases.values()),
            requested_episodes=sum(c['requested_episodes'] for c in cases.values()),comparisons=comparisons);save()
    state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat())
except Exception:
    state.update(status='failed',error=traceback.format_exc(),finished_at=datetime.now(timezone.utc).isoformat());raise
finally:
    save()

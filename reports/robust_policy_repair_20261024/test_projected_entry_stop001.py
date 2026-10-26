"""CPU failure fixture exercising the actual native outer update bookkeeping.

No simulator, PPO optimizer step or real process signal is used.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace

sys.path[:0]=['src','scripts',str(Path(__file__).parent)]
import torch
from export_robust_getup import load_mean
from train_robust_getup import TrainingStopped,digest,write_json
from entry_common_guard001 import PhaseGuard

torch.set_num_threads(1)
R=Path('reports/robust_policy_repair_20261024')
cp=Path('runs/robust_policy_repair_20261024/function_root_entry002/model_initial.pt')
mean,_=load_mean(cp)
original=deepcopy(mean.state_dict())
guard=PhaseGuard(mean,R/'entry_phase_path_dataset001.npz')
neutral,neutral_checks=guard.measure()
native_tree=ast.parse(Path('scripts/train_robust_getup.py').read_text())
outer_node=next(n for n in ast.walk(native_tree) if isinstance(n,ast.FunctionDef) and n.name=='update')
stop_node=deepcopy(next(n for n in ast.walk(native_tree) if isinstance(n,ast.FunctionDef) and n.name=='request_stop'))
stop_node.body[0]=ast.Global(names=['stop_requested'])
wrapper=R/'run_projected_entry_pilot001.py'
tree=ast.parse(wrapper.read_text())
inner_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='update')
tensor_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tensor_receipt')
class Writer:
    def __init__(self): self.rows=[]
    def add_scalar(self,*args): self.rows.append(args)
    def flush(self): pass

with tempfile.TemporaryDirectory(prefix='entry-common-stop-fixture-') as directory:
    out=Path(directory)
    report=dict(new_optimizer_attempts=0,new_optimizer_updates=0,completed_updates=1501)
    saved=[];signals=[]
    actor=torch.nn.Module();actor.add_module('mean',mean)
    runner=SimpleNamespace(alg=SimpleNamespace(actor=actor,critic=torch.nn.Linear(1,1)))
    env=SimpleNamespace(actor_updates=1501)
    st=SimpleNamespace(observations={'policy':torch.zeros(1,1,35)},
        distribution_params=[torch.zeros(1,1,6),torch.ones(1,1,6)])
    for name in ('actions','dones','rewards','values','actions_log_prob'):
        setattr(st,name,torch.zeros(1,1))
    mask=SimpleNamespace(storage=st,learning=torch.ones(1,1),bootstrap=torch.ones(1,1),
        projected_mean=mean,phase_guard=guard,projected_rows=[],
        projector=SimpleNamespace(project=lambda model:(_ for _ in ()).throw(RuntimeError('fixture solver failure'))),
        projected_initial=dict(metrics=neutral,checks=neutral_checks),metrics={'fixture':True})
    def save(path): saved.append(dict(path=Path(path).name,report=deepcopy(report)))
    outer=dict(report=report,args=SimpleNamespace(output=out),write_json=write_json,
        parent_updates=1501,runner=runner,env=env,mask=mask,save=save,
        TrainingStopped=TrainingStopped,torch=torch,time=time,start=time.monotonic(),stop_requested=False)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stop_node],type_ignores=[])),'native-request-stop','exec'),outer)
    def kill(pid,number):
        signals.append(number)
        outer['request_stop'](number,None)
    def native_update(self):
        # Deliberately violate the stopped-phase constraint in one completed
        # fixture update while keeping the feature basis and wheel rows exact.
        with torch.no_grad(): mean.entry_recovery[-1].bias[:4].add_(.3)
        return {'fixture_loss':.5}
    import hashlib
    writer=Writer()
    inner=dict(latest={},native_update=native_update,native_write=write_json,output=out,
        source=cp,train=SimpleNamespace(digest=digest),Path=Path,time=time,hashlib=hashlib,
        __file__=str(wrapper.resolve()),os=SimpleNamespace(kill=kill,getpid=lambda:0),
        signal=signal,logger=SimpleNamespace(writer=writer),deepcopy=deepcopy,VERSION='fixed-basis-common-projected-Adam-v1')
    exec(compile(ast.Module(body=[tensor_node,inner_node],type_ignores=[]),'actual-common-guard-wrapper','exec'),inner)
    outer['original_update']=lambda:inner['update'](mask)
    exec(compile(ast.Module(body=[outer_node],type_ignores=[]),'actual-native-outer-update','exec'),outer)
    stopped=False
    try: outer['update']()
    except TrainingStopped: stopped=True
    receipt=json.loads((out/'entry_phase_guard.json').read_text())
    failure=receipt['updates'][0]
    checks=dict(neutral_passed=all(neutral_checks.values()),
        projection_failure_rejected=not failure['checks']['projection_solved'],
        fixed_basis_preserved=failure['checks']['frozen_feature_basis_exact'],
        wheel_rows_preserved=failure['checks']['wheel_rows_exact'],
        one_stop_signal=signals==[signal.SIGTERM],stop_requested=receipt['stop_requested'],
        actual_outer_raised_training_stopped=stopped,attempt_charged=report['new_optimizer_attempts']==1,
        completed_charged=report['new_optimizer_updates']==1,
        clock_incremented=report['completed_updates']==1502 and env.actor_updates==1502,
        in_flight_cleared=report['optimizer_update_in_flight'] is False,
        checkpoint_save_invoked_once=len(saved)==1 and saved[0]['path']=='model_stopped.pt',
        saved_counter_complete=len(saved)==1 and saved[0]['report']['completed_updates']==1502,
        stop_event_logged=any(x[0]=='EntryProjection/passed' and x[1]==0 for x in writer.rows))
    mean.load_state_dict(original)
    checks['all_fixture_weights_restored']=all(torch.equal(v,mean.state_dict()[k]) for k,v in original.items())
    result=dict(schema='projected-Adam-negative-stop-fixture-v1',checks=checks,
        failure_metrics=failure['metrics'],failure_checks=failure['checks'],
        native_outer_sha256=digest(Path('scripts/train_robust_getup.py')),wrapper_sha256=digest(wrapper),
        fixture_attempts=1,fixture_completed=1,actual_PPO_updates=0,optimizer_steps=0,
        scope='AST-executed actual wrapper and actual trainer update/request_stop; synthetic completed update, fake signal and save callback. Not native simulator evidence.',
        source_checkpoint_sha256=digest(cp),created_unix=time.time())
    write_json(R/'projected_entry_negative_stop001.json',result)
    print(json.dumps(result,indent=2))
    assert all(checks.values()),checks

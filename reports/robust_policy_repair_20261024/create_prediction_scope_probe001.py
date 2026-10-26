"""Explicit existing 5 ms causal encoder prediction scope intervention."""
from pathlib import Path
import sys,copy,json,hashlib,time
sys.path[:0]=['src','scripts']
import torch,numpy as np
from export_robust_getup import load_mean
from wheeled_algo.robust_getup import quiet_gate

torch.set_num_threads(1)
base=Path('runs/robust_policy_repair_20261024');source=base/'low_wheel_diagonal_restoring_probe001/model_probe.pt';old,info=load_mean(source);state=torch.load(source,map_location='cpu',weights_only=False);out=base/'quiet_prediction_all005_probe001';out.mkdir(exist_ok=False);cp=out/'model_probe.pt';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
state['infos']['settings']['quiet_leg_prediction_all_heights']=True
state['infos']['quiet_prediction_scope_probe']=dict(source=str(source.resolve()),source_sha256=sha(source),previous_scope='high_only',scope='all_quiet_heights',prediction_s=.005,new_PPO=0,tensor_changes=0,hypothesis='Test whether the existing 5 ms causal leg encoder extrapolation reduces low stance transient overshoot. No ground-truth, new state, actuator or reward change.')
torch.save(state,cp);new,_=load_mean(cp);checks=dict(all_weights_exact=all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()));changed=0;outside=0;highchanged=0
trace=base/'diagnosis013_low_diagonal_fixed/evaluation/trajectory.npz';frames=np.load(trace)['observations'].reshape(-1,35)
with torch.inference_mode():
 for obs in torch.tensor(frames).split(2048):
  a,b=old(obs),new(obs);diff=(a!=b).any(-1);changed+=int(diff.sum());outside+=int(diff[quiet_gate(obs).ravel()==0].sum());highchanged+=int(diff[obs[:,3]/5.>=.4].sum())
checks.update(outside_quiet_exact=outside==0,high_stance_exact=highchanged==0)
assert all(checks.values()),checks
r=dict(schema='existing-quiet-prediction-scope-probe-v1',checkpoint=str(cp),sha256=sha(cp),source_sha256=sha(source),checks=checks,frames=len(frames),changed_frames=changed,new_PPO=0,actual_experiment_consumed=2587,source_clock=info['completed_updates'],gate={'fixed_standing_at_least':21,'low_all7_required':True,'motion_MAE_at_most':.135,'high_and_recovery_outputs_preserved':True,'no_automatic_PPO':True},qualification=None,hardware_started=False,created_unix=time.time());(out/'model_probe.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

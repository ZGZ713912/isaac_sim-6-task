"""Actual checkpoint decoder, sensor replay and static-policy equivalence."""
from pathlib import Path
from copy import deepcopy
import sys,json,time,tempfile
sys.path[:0]=['src','scripts',str(Path(__file__).resolve().parent)]
import numpy as np,torch
from export_robust_getup import load_mean,digest
from train_robust_getup import write_json
from wheeled_algo.quiet_stance_authority import attach_stance_authority,MODULE
from wheeled_algo.periodic_entry_actor import exact_state_equal
from wheeled_algo import robust_getup,quiet_output_actor
from quiet_gyro_stance_authority_probe001 import policy_authority,NATIVE_GATE
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024');torch.set_num_threads(1)
cp=B/'quiet_stance050003/model_probe.pt';source=B/'motion_damping080002/model_probe.pt'
mean,info=load_mean(cp);old,prior=load_mean(source);neutral=attach_stance_authority(deepcopy(old),0.)
a,b=[torch.load(x,map_location='cpu',weights_only=False) for x in (cp,source)]
checks=dict(payload_weights_critic_optimizer_rng_exact=exact_state_equal({k:v for k,v in a.items() if k!='infos'},{k:v for k,v in b.items() if k!='infos'}),
 namespaces_exact=list(mean.state_dict())==list(old.state_dict()),actor_tensors_exact=exact_state_equal(mean.state_dict(),old.state_dict()),
 learning_clock_exact=info['completed_updates']==prior['completed_updates']==1501,
 static_policy_no_global_patch=True)
frames=zero_different=probe_different=changed=0;delta=np.zeros(6);traces=[]
methods=(robust_getup.RobustGetupActor.forward,quiet_output_actor.QuietOutputGetupActor.forward)
with torch.inference_mode():
 for name in ('diagnosis035_current_source_control','diagnosis035_stance_authority050_fixed'):
  path=B/name/'evaluation/trajectory.npz';x=np.load(path)['observations'].reshape(-1,35);traces.append(dict(path=str(path),sha256=digest(path),frames=len(x)))
  for values in np.array_split(x,(len(x)+511)//512):
   v=torch.from_numpy(values.copy());before=old(v);zero=neutral(v)
   with policy_authority(.5):expected=old(v)
   after=mean(v)
   zero_different+=int((before!=zero).any(-1).sum());probe_different+=int((expected!=after).any(-1).sum());changed+=int((before!=after).any(-1).sum());frames+=len(v)
   delta=np.maximum(delta,(after-expected).abs().amax(0).numpy())
checks.update(zero_strength_full_function_exact=zero_different==0,diagnostic_output_bit_exact=probe_different==0,
 active_function_change_nonempty=changed>0,native_methods_restored=methods==(robust_getup.RobustGetupActor.forward,quiet_output_actor.QuietOutputGetupActor.forward),
 shared_observer_and_std_gate_unchanged=robust_getup.quiet_gate is NATIVE_GATE and quiet_output_actor.quiet_gate is NATIVE_GATE)
rejections={}
with tempfile.TemporaryDirectory(prefix='quiet_stance_decoder_',dir=R) as folder:
 for kind in ('weights','optimizer','source_sha','module_sha','schema_downgrade','self_cycle'):
  state=deepcopy(a);p=Path(folder)/(kind+'.pt')
  if kind=='weights':next(iter(state['actor_state_dict'].values())).add_(1)
  elif kind=='optimizer':state['optimizer_state_dict']['param_groups'][0]['lr']*=2
  elif kind=='source_sha':state['infos']['quiet_stance_authority_migration']['source_sha256']='0'*64
  elif kind=='module_sha':state['infos']['source_sha256'][MODULE]='0'*64
  elif kind=='schema_downgrade':state['infos']['schema']='robust-getup-checkpoint-v2'
  else:state['infos']['quiet_stance_authority_migration']['source_checkpoint']=str(p.resolve())
  torch.save(state,p)
  try:load_mean(p)
  except (ValueError,KeyError) as error:rejections[kind]=str(error)
  else:raise AssertionError('Decoder accepted '+kind)
checks['six_actual_decoder_negative_controls']=len(rejections)==6
out=dict(schema='quiet-stance-implementation-replay-v1',checks={k:bool(v) for k,v in checks.items()},
 checkpoint=str(cp),checkpoint_sha256=digest(cp),source_checkpoint=str(source),source_sha256=digest(source),module_sha256=digest(MODULE),
 frames=frames,zero_strength_different=zero_different,diagnostic_output_different=probe_different,active_changed_frames=changed,diagnostic_delta_peak=delta.tolist(),
 traces=traces,rejections=rejections,new_PPO=0,actual_experiment_consumed=2630,repair_optimizer_updates=250,qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'quiet_stance_implementation_preflight001.json',out);print(json.dumps(out,indent=2));assert all(checks.values()),checks

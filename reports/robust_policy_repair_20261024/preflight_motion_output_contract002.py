"""CPU-only test of a draft contract using the real unchanged V4 checkpoint."""
from pathlib import Path
import sys,json,time,importlib.util,tempfile
from copy import deepcopy
sys.path[:0]=['src','scripts']
import torch
from export_robust_getup import load_mean,digest
from train_robust_getup import write_json
from wheeled_algo.periodic_entry_actor import exact_state_equal
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024');torch.set_num_threads(1)
path=Path('src/wheeled_algo/motion_output_checkpoint.py');spec=importlib.util.spec_from_file_location('motion_output_checkpoint_draft',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cp=B/'quiet_stance000003/model_probe.pt';root=torch.load(cp,map_location='cpu',weights_only=False);mean,_=load_mean(cp)
names=m.optimizer_actor_names(mean);initial=m.initialize_state(cp,root,digest(path),names);m.validate_state(initial,root,root,names)
checks=dict(actual_source28_parameter_namespace=len(names)==28,
 actual_optimizer_tensor_shapes=all(root['optimizer_state_dict']['state'][i]['exp_avg'].shape==root['actor_state_dict'][k].shape for i,k in zip(root['optimizer_state_dict']['param_groups'][0]['params'],names,strict=True)),
 initial_payload_exact=exact_state_equal({k:v for k,v in root.items() if k!='infos'},{k:v for k,v in initial.items() if k!='infos'}))
rejected={}
for kind in ('entry','quiet','basis','std','critic_nan','Adam_frozen','clock','source_function','threshold','module_source','scope','saved_rng'):
 state=deepcopy(initial)
 if kind=='entry':state['actor_state_dict']['mlp.mean.entry_recovery.4.bias'].add_(.01)
 elif kind=='quiet':state['actor_state_dict']['mlp.mean.quiet.2.bias'].add_(.01)
 elif kind=='basis':state['actor_state_dict']['mlp.mean.recovery.0.bias'].add_(.01)
 elif kind=='std':state['actor_state_dict']['mlp.log_std'].add_(.01)
 elif kind=='critic_nan':next(iter(state['critic_state_dict'].values())).fill_(float('nan'))
 elif kind=='Adam_frozen':state['optimizer_state_dict']['state'][0]['exp_avg'].add_(.01)
 elif kind=='clock':state['infos']['completed_updates']+=1
 elif kind=='source_function':state['infos']['settings']['quiet_stance_retained_authority']=.75
 elif kind=='threshold':state['infos']['settings']['quiet_stance_recovery_threshold']=.051
 elif kind=='module_source':state['infos']['source_sha256']['src/wheeled_algo/quiet_stance_authority.py']='0'*64
 elif kind=='scope':state['infos']['policy_function_learning']['scope']='all-actor'
 else:state['infos']['torch_rng'][0]^=1
 try:m.validate_state(state,root,root,names)
 except ValueError as error:rejected[kind]=str(error)
 else:raise AssertionError('Draft accepted '+kind)
checks['twelve_initial_tamper_controls_rejected']=len(rejected)==12
with tempfile.TemporaryDirectory(prefix='motion_output_cpu_',dir=R) as folder:
 initial_path=Path(folder)/'initial.pt';torch.save(initial,initial_path)
 state=deepcopy(initial)
 # Exercise actual native torch.Adam with its original two groups and saved
 # state. Only the final2 motion tensors and all8 critic tensors receive grads.
 actor_parameters=[torch.nn.Parameter(state['actor_state_dict'][k].clone()) for k in names]
 critic_names=list(state['critic_state_dict']);critic_parameters=[torch.nn.Parameter(state['critic_state_dict'][k].clone()) for k in critic_names]
 opt=torch.optim.Adam([dict(params=actor_parameters,name='actor'),dict(params=critic_parameters,name='critic')]);opt.load_state_dict(state['optimizer_state_dict'])
 selected=[names.index(k) for k in m.LEARNED_KEYS]
 before=deepcopy(opt.state_dict())
 for tick in range(12):
  opt.zero_grad(set_to_none=True)
  for i in selected:actor_parameters[i].grad=torch.full_like(actor_parameters[i],.001)
  for p in critic_parameters:p.grad=torch.full_like(p,.001)
  opt.step()
 state['actor_state_dict'].update({k:p.detach().clone() for k,p in zip(names,actor_parameters,strict=True)})
 state['critic_state_dict'].update({k:p.detach().clone() for k,p in zip(critic_names,critic_parameters,strict=True)})
 state['optimizer_state_dict']=opt.state_dict();info=state['infos'];info['completed_updates']+=1;info['policy_function_stage']='native-training';info['policy_function_learning']=m.make_progress(initial_path,initial,state['actor_state_dict'],1)
 m.validate_state(state,root,initial,names)
 checks['twelve_real_Adam_steps_accepted']=True
 checks['all26_frozen_actor_Adam_states_exact']=all(exact_state_equal(before['state'][i],opt.state_dict()['state'][i]) for i in range(28) if i not in selected)
 for kind in ('partial11steps','too_many13steps','native_frozen_Adam','native_frozen_weights'):
  bad=deepcopy(state)
  if kind=='partial11steps':bad['optimizer_state_dict']['state'][selected[0]]['step'].sub_(1.)
  elif kind=='too_many13steps':bad['optimizer_state_dict']['state'][selected[0]]['step'].add_(1.)
  elif kind=='native_frozen_Adam':bad['optimizer_state_dict']['state'][0]['exp_avg'].add_(1.)
  else:bad['actor_state_dict']['mlp.mean.entry_recovery.4.bias'].add_(.01)
  try:m.validate_state(bad,root,initial,names)
  except ValueError as error:rejected[kind]=str(error)
  else:raise AssertionError('Native draft accepted '+kind)
checks['four_native_transition_negative_controls_rejected']=len(rejected)==16
out=dict(schema='motion-output-contract-CPU-preflight-v1',checks={k:bool(v) for k,v in checks.items()},source_checkpoint=str(cp),source_sha256=digest(cp),
 contract_module=str(path),contract_sha256=digest(path),optimizer_actor_parameter_names=names,selected_parameter_count=sum(root['actor_state_dict'][k].numel() for k in m.LEARNED_KEYS),
 rejections=rejected,cpu_synthetic_Adam_steps=12,new_native_PPO=0,new_training_admitted=False,
 interpretation='CPU synthetic gradients validate saved-state restrictions and step accounting only; no simulator learning or behavior claim.',qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'motion_output_contract_CPU_preflight002.json',out);print(json.dumps(out,indent=2));assert all(checks.values()),checks

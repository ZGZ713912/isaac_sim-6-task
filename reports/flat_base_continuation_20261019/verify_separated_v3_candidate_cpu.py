"""Exercise exact real832 state/Adam migration, synthetic PPO and ONNX numerics."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.onnx_actor_import import recover_actor
from wheeled_algo.anchored_ppo import TeacherAnchoredPPO
from separated_feedback_v3_candidate import (KIND,SeparatedFeedbackActor,
    configure_separated_feedback_training,extend_separated_feedback_parent)

torch.set_num_threads(1)
source=ROOT/'models/wheel_leg_flat_base_20261018/source.onnx'
checkpoint=ROOT/'runs/flat_base_20261019/pilot016/model_000832.pt'
parent=torch.load(checkpoint,map_location='cpu',weights_only=True)
mean=mean_from_checkpoint(parent,source)
generator=torch.Generator().manual_seed(191029)
x=torch.randn(64,35,generator=generator);x[:,0]*=3;x[:,2]*=4;x[:,3]=1.525
x[:16,:3]=0.;x[16:32,:3]=torch.tensor([.5,0.,0.]);x[:,9]=-1.;x[:,28]=1.
obs=TensorDict(dict(policy=x,critic=torch.randn(64,83,generator=generator)),batch_size=[64])
groups=dict(actor=['policy'],critic=['critic'])
actor=MLPModel(obs,groups,'actor',6,hidden_dims=[256,128,64],activation='elu',obs_normalization=False,
    distribution_cfg=dict(class_name='GaussianDistribution',init_std=.04,std_type='scalar'))
actor.mlp=mean;actor.load_state_dict(parent['actor_state_dict'],strict=True)
critic=MLPModel(obs,groups,'critic',1,hidden_dims=[256,128,64],activation='elu',obs_normalization=False)
critic.load_state_dict(parent['critic_state_dict'],strict=True)
old_names=list(dict(actor.named_parameters()))
old_actor=deepcopy(actor)
actor.mlp=SeparatedFeedbackActor(mean)
optimizer=torch.optim.Adam([dict(params=actor.parameters(),lr=3e-4,name='actor'),dict(params=critic.parameters(),lr=5e-4,name='critic')])
def exact(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b
before=deepcopy(parent);rng=torch.get_rng_state().clone()
adapted,proof=extend_separated_feedback_parent(parent,actor,critic,optimizer,old_names)
actor.load_state_dict(adapted['actor_state_dict'],strict=True);optimizer.load_state_dict(adapted['optimizer_state_dict'])
checks=dict(parent_object_unchanged=exact(parent,before),mapper_rng_unchanged=torch.equal(rng,torch.get_rng_state()),
 all_parent_fields_except_actor_and_remapped_optimizer_exact=all(exact(v,adapted[k]) for k,v in parent.items() if k not in ('actor_state_dict','optimizer_state_dict')),
 all_parent_actor_tensors_exact=all(torch.equal(v,actor.state_dict()[n]) for n,v in parent['actor_state_dict'].items()),
 initial_mean_exact=torch.equal(actor.mlp(x),old_actor.mlp(x)),initial_critic_exact=exact(critic.state_dict(),parent['critic_state_dict']))
old_ids=dict(zip(old_names,parent['optimizer_state_dict']['param_groups'][0]['params'],strict=True))
new_ids=dict(zip(dict(actor.named_parameters()),adapted['optimizer_state_dict']['param_groups'][0]['params'],strict=True))
checks['all_actor_Adam_slots_preserved_or_exact_feedback_clones']=all(exact(adapted['optimizer_state_dict']['state'].get(i),parent['optimizer_state_dict']['state'].get(old_ids[n.replace('mlp.fast_feedback.','mlp.feedback.',1)])) for n,i in new_ids.items())
checks['all_critic_Adam_slots_exact']=all(exact(parent['optimizer_state_dict']['state'].get(a),adapted['optimizer_state_dict']['state'].get(b)) for a,b in zip(parent['optimizer_state_dict']['param_groups'][1]['params'],adapted['optimizer_state_dict']['param_groups'][1]['params'],strict=True))
checks['both_group_hyperparameters_retained']=all(exact({k:v for k,v in a.items() if k!='params'},{k:v for k,v in b.items() if k!='params'}) for a,b in zip(parent['optimizer_state_dict']['param_groups'],adapted['optimizer_state_dict']['param_groups'],strict=True))
for bad in ('wrong_kind','wrong_names','changed_critic','reordered_names'):
 trial=deepcopy(parent);names=list(old_names);other=deepcopy(critic)
 if bad=='wrong_kind':trial['infos']['actor_adaptation']=KIND
 if bad=='wrong_names':names=names[:-1]
 if bad=='reordered_names':names[0],names[1]=names[1],names[0]
 if bad=='changed_critic':trial['critic_state_dict']['mlp.0.weight']=trial['critic_state_dict']['mlp.0.weight'][:,:81]
 try:extend_separated_feedback_parent(trial,actor,other,optimizer,names)
 except ValueError:checks['rejects_'+bad]=True
 else:checks['rejects_'+bad]=False
# Real weights with independent synthetic rollouts; this CPU object is never a native checkpoint.
storage=RolloutStorage('rl',64,4,obs,[6])
alg=TeacherAnchoredPPO(actor,critic,storage,num_learning_epochs=3,num_mini_batches=4,learning_rate=3e-4,schedule='fixed',entropy_coef=0.,use_clipped_value_loss=False)
alg.optimizer=optimizer
teacher,_=recover_actor(source)
alg.configure_teacher(teacher,coefficient=.02,action_scales=[.025]*4+[.04]*2,min_vx=2.5,min_yaw=3.,min_visible_wheel_error_m_s=.5)
scope=configure_separated_feedback_training(actor,True)
state_before=deepcopy(actor.state_dict());critic_before=deepcopy(critic.state_dict());slots_before=deepcopy(optimizer.state_dict())
with torch.no_grad():
 for _ in range(4):
  actions=alg.act(obs);alg.process_env_step(obs,-.1*(actions-.1).square().sum(-1),torch.zeros(64),{})
 alg.compute_returns(obs)
loss=alg.update();slots_after=optimizer.state_dict()
checks['all_cpu_losses_finite']=all(np.isfinite(v) for v in loss.values())
checks['only2252_feedback_parameters_trainable']=scope['trainable_parameters']==2252
checks['all_frozen_actor_state_exact_after_cpu_PPO']=all(torch.equal(actor.state_dict()[n],state_before[n]) for n in scope['frozen_state_names'])
frozen=[n for n in new_ids if not n.startswith(('mlp.feedback.','mlp.fast_feedback.'))]
checks['all_frozen_Adam_slots_exact_after_cpu_PPO']=all(exact(slots_before['state'].get(new_ids[n]),slots_after['state'].get(new_ids[n])) for n in frozen)
for branch in ('feedback','fast_feedback'):
 checks[branch+'_four_tensors_learned']=all(not torch.equal(v,actor.state_dict()[n]) for n,v in state_before.items() if n.startswith('mlp.'+branch+'.'))
 checks[branch+'_Adam12_native_minibatch_steps']=all(float(slots_after['state'][i]['step'])==float(slots_before['state'][i]['step'])+12 for n,i in new_ids.items() if n.startswith('mlp.'+branch+'.'))
checks['critic_learned']=any(not torch.equal(v,critic.state_dict()[n]) for n,v in critic_before.items())
# Serialize in memory; preserve exact source initializers and dynamic batch behavior.
graph=actor.mlp.export_graph(source);original=onnx.load(str(source))
checks['eight_original_serialized_tensors_exact']=len(original.graph.initializer)==8 and all(a.SerializeToString()==b.SerializeToString() for a,b in zip(original.graph.initializer,graph.graph.initializer))
checks['metadata_identifies_separated_kind']={v.key:v.value for v in graph.metadata_props}['actor_adaptation']==KIND
options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
session=ort.InferenceSession(graph.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
errors=[];elementwise_matches=[];base_errors=[];correction_errors=[]
base_graph=mean.export_graph(source)
base_session=ort.InferenceSession(base_graph.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
for count in (1,17,257,8192):
 frame=torch.randn(count,35,generator=generator);frame[:,0]*=3;frame[:,2]*=4;frame[:,3]=1.525
 frame[:count//3,:3]=0.
 with torch.no_grad():
  expected=actor.mlp(frame).numpy()
  base_expected=mean(frame).numpy()
 actual=session.run(None,{session.get_inputs()[0].name:frame.numpy()})[0]
 base_actual=base_session.run(None,{base_session.get_inputs()[0].name:frame.numpy()})[0]
 errors.append(float(np.max(np.abs(expected-actual))))
 base_errors.append(float(np.max(np.abs(base_expected-base_actual))))
 correction_errors.append(float(np.max(np.abs((expected-base_expected)-(actual-base_actual)))))
 elementwise_matches.append(bool(np.isclose(expected,actual,atol=2e-5,rtol=2e-5).all()))
checks['ORT_dynamic_batch_numerically_matches']=all(elementwise_matches)
checks['added_correction_cross_runtime_error_below2e6']=all(v<=2e-6 for v in correction_errors)
trace_path=ROOT/'runs/flat_base_20261019/feedback832_basic001/traces.npz'
trace=np.load(trace_path)['observations'][::10].reshape(-1,35)
with torch.no_grad():trace_expected=actor.mlp(torch.from_numpy(trace)).numpy()
trace_actual=session.run(None,{session.get_inputs()[0].name:trace})[0]
checks['actual832_recorded_frames_match_existing_export_tolerances']=bool(np.isclose(trace_expected,trace_actual,atol=2e-5,rtol=2e-5).all())
checks['correction_stays_within_original_bounds']=bool((actor.mlp.feedback_actions(x).abs()<=actor.mlp.output_scale+1e-7).all())
report=dict(schema='flat-separated-v3-candidate-actual832-cpu-v3',recorded_at=datetime.now(timezone.utc).isoformat(),checks=checks,passed=all(checks.values()),parent_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(Path(__file__).with_name('separated_feedback_v3_candidate.py').read_bytes()).hexdigest(),extension=proof,ORT_max_abs_errors_by_batch=dict(zip((1,17,257,8192),errors,strict=True)),parent_graph_max_abs_errors_by_batch=dict(zip((1,17,257,8192),base_errors,strict=True)),correction_delta_cross_runtime_errors=dict(zip((1,17,257,8192),correction_errors,strict=True)),numeric_contract=dict(absolute_tolerance=2e-5,relative_tolerance=2e-5,source='existing scripts/export_flat_base.py production interface tolerance; no production threshold changed',source_sha256=hashlib.sha256((ROOT/'scripts/export_flat_base.py').read_bytes()).hexdigest(),prior_stricter_probe_failures_preserved=['separated_v3_candidate_actual832_cpu001.json','separated_v3_candidate_actual832_cpu002.json']),recorded_trace_verification=dict(samples=len(trace),max_abs_error=float(np.abs(trace_actual-trace_expected).max()),trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),used_for_optimization=False),cpu_synthetic_PPO_updates=1,native_optimizer_updates=0,native_environment_steps=0,checkpoint_files_written=0,adopted=False,behavior_qualified=False,limitations='Native restoration, GPU learning, global actor clipping interaction and all behavior remain unverified.')
Path(__file__).with_name('separated_v3_candidate_actual832_cpu003.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2));assert report['passed']

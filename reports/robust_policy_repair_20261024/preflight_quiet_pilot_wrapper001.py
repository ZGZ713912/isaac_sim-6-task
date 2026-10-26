"""CPU test of actual pilot callbacks using real legacy Adam state and log tags."""
from pathlib import Path
import ast,json,sys,time,types,inspect,textwrap
sys.path[:0]=['src','scripts']
import torch
from torch import nn
from export_robust_getup import load_mean,digest
from train_robust_getup import write_json
from wheeled_algo.gap_recovery_exploration import GapRecoveryNoiseMean
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024');torch.set_num_threads(1)
wrapper=R/'run_quiet_output_pilot001.py';output=B/'quiet_pilot_wrapper_cpu001';output.mkdir(exist_ok=False)
source=B/'quiet_output_root001/model_initial.pt';reference=output/'reference.json'
reference.write_text(json.dumps(dict(optimizer_attempts=0,new_PPO=0)))
sys.argv=[str(wrapper),'--zero-rollout-reference',str(reference),'--output',str(output),'--resume',str(source),'--updates','8','--quiet-only','--quiet-output-only']
tree=ast.parse(wrapper.read_text());tree.body=tree.body[:-1];namespace={'__file__':str(wrapper.resolve()),'__name__':'fallen_wrapper_cpu_fixture'}
exec(compile(tree,str(wrapper),'exec'),namespace)
state=torch.load(source,map_location='cpu',weights_only=False);s=state['infos']['settings'];mean,_=load_mean(source)
actor=nn.Module();actor.mlp=GapRecoveryNoiseMean(mean,s['quiet_exploration_fraction'],s.get('quiet_wheel_exploration_fraction'),s.get('gap_recovery_leg_exploration_fraction',1.))
actor.load_state_dict(state['actor_state_dict'],strict=True)
parameters=[p for p in actor.parameters() if p.requires_grad];critic=[nn.Parameter(v.clone()) for v in state['critic_state_dict'].values()]
optimizer=torch.optim.Adam([dict(params=parameters,name='actor'),dict(params=critic,name='critic')]);optimizer.load_state_dict(state['optimizer_state_dict'])
actor.requires_grad_(False)
for head in (mean.quiet,mean.quiet_low,mean.quiet_high):head[-1].requires_grad_(True)
alg=types.SimpleNamespace(actor=actor,optimizer=optimizer)
self=types.SimpleNamespace()
def init(obj,algorithm,*,num_envs):
 obj.algorithm=algorithm;obj.storage=types.SimpleNamespace(device='cpu',observations={'policy':torch.zeros(2,num_envs,35)})
 obj.learning=torch.ones(2,num_envs,dtype=torch.bool)
namespace['native_init']=init
namespace['initialize'](self,alg,num_envs=256)
steps=[]
def update(obj):
 for tick in range(12):
  optimizer.zero_grad(set_to_none=True)
  for n,p in actor.named_parameters():
   if p.requires_grad:p.grad=torch.full_like(p,.001)
  for p in critic:p.grad=torch.full_like(p,.001)
  optimizer.step()
 optimizer.zero_grad(set_to_none=True) #Native may clear grads; this must be safe.
 steps.append(12);return dict(surrogate=0.)
namespace['native_update']=update
class Writer:
 def __init__(self):self.scalars=[]
 def add_scalar(self,*args):self.scalars.append(args)
 def flush(self):pass
writer=Writer();namespace['logger']=types.SimpleNamespace(writer=writer)
loss=namespace['update'](self)
row=self.quiet_rows[0]
checks=dict(main_not_called=not (output/'model_initial.pt').exists(),
 native_update_loss_preserved=loss==dict(surrogate=0.),twelve_synthetic_Adam_steps=steps==[12],
 all_step_checks_pass=all(row['checks'].values()),correct_logging_clock=all(x[2]==1501 for x in writer.scalars),
 absent_final_grad_safe=row['metrics']['output_gradient_L2']==0.,
 TensorBoard_stable_tags=all(x[0].startswith('QuietOutputTraining/') for x in writer.scalars))
out=dict(schema='quiet-output-pilot-callback-CPU-preflight-v1',checks={k:bool(v) for k,v in checks.items()},wrapper_sha256=digest(wrapper),
 synthetic_cpu_Adam_steps=12,native_PPO_updates=0,training_admitted=False,logged_tags=[x[0] for x in writer.scalars],qualification=None,created_unix=time.time())
write_json(R/'quiet_pilot_wrapper_preflight001.json',out);print(json.dumps(out,indent=2));assert all(checks.values()),checks

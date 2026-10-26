"""At most8 native PPO updates on the existing fallen output layer.

Sampling, likelihood, native GAE/PPO/Adam and runtime forward are unchanged.
Counters read native pre-reset logs; they do not change transitions. The first
real rollout must reproduce the preceding zero-PPO collector before attempts.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,sys,time,os,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
if '--zero-rollout-reference' not in sys.argv:raise ValueError('Require preceding real zero-PPO rollout')
i=sys.argv.index('--zero-rollout-reference');reference=Path(sys.argv[i+1]);del sys.argv[i:i+2]
if int(sys.argv[sys.argv.index('--updates')+1])!=8:raise ValueError('This bounded trial admits exactly8 updates')
if '--recovery-only' not in sys.argv or '--motion-recovery-focus' not in sys.argv or '--entry-recovery-only' in sys.argv:
    raise ValueError('Only existing fallen output learning with moving push exposure')
output=Path(sys.argv[sys.argv.index('--output')+1]);source=Path(sys.argv[sys.argv.index('--resume')+1])
import json,torch
import train_robust_getup as train
from wheeled_algo.recovery_rollout import RecoveryRolloutMask
from wheeled_algo import exploration_init
native_init=RecoveryRolloutMask.__init__;native_returns=RecoveryRolloutMask.compute_returns
native_process=RecoveryRolloutMask.process_env_step;native_update=RecoveryRolloutMask.update
native_logging=exploration_init.install_axis_std_logging
logger=None

def receipt(value):
 a=value.detach().cpu().contiguous().numpy()
 return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())

def logging(instance):
 global logger
 logger=instance;return native_logging(instance)

def initialize(self,algorithm,*,num_envs):
 native_init(self,algorithm,num_envs=num_envs)
 state=torch.load(source,map_location='cpu',weights_only=False)
 if state['infos']['schema']!='robust-getup-checkpoint-v5' or state['infos']['completed_updates']!=1501:raise ValueError('Use authenticated1501 V5 initialization')
 names=[n for n,p in algorithm.actor.named_parameters() if p.requires_grad]
 if names!=['mlp.mean.recovery.4.weight','mlp.mean.recovery.4.bias']:raise ValueError('Unexpected trainable actor scope')
 self.fallen_counts=torch.zeros(3,device=self.storage.device);self.fallen_rows=[];self.fallen_reference=json.loads(reference.read_text())
 if self.fallen_reference['optimizer_attempts']!=0 or self.fallen_reference['new_PPO']!=0:raise ValueError('Reference includes optimizer work')
 self.fallen_source_state=state;self.fallen_num_envs=num_envs
 self.fallen_ids=[algorithm.optimizer.param_groups[0]['params'][state['infos']['policy_function_root']['optimizer_actor_parameter_names'].index(k)] for k in ['mlp.mean.recovery.4.weight','mlp.mean.recovery.4.bias']]
 self.fallen_initial_steps=[float(algorithm.optimizer.state[p]['step']) for p in self.fallen_ids]
 target=output/'source'/Path(__file__).resolve().relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(__file__,target)

def process(self,obs,rewards,dones,extras):
 result=native_process(self,obs,rewards,dones,extras)
 logs=extras['log']
 self.fallen_counts[0].add_(logs['UnifiedGetup/external_push_fraction']*self.fallen_num_envs)
 self.fallen_counts[1].add_(logs['UnifiedGetup/in_motion_fall_fraction']*self.fallen_num_envs)
 self.fallen_counts[2].add_(logs['UnifiedGetup/recovered_events'])
 return result

def returns(self,last_observations):
 result=native_returns(self,last_observations)
 if not self.fallen_rows:
  st=self.storage;values={'observations__'+str(k):receipt(v) for k,v in st.observations.items()}
  for name in ('actions','dones','rewards','values','actions_log_prob','returns','advantages'):values[name]=receipt(getattr(st,name))
  values['learning_mask']=receipt(self.learning);values['bootstrap_mask']=receipt(self.bootstrap)
  for i,v in enumerate(st.distribution_params):values['distribution_param_'+str(i)]=receipt(v)
  for k,v in last_observations.items():values['last_observations__'+str(k)]=receipt(v)
  values['last_critic_values']=receipt(self.algorithm.critic(last_observations))
  compared={k:values.get(k)==v for k,v in self.fallen_reference['tensors'].items()}
  all_exact=all(compared.values()) and values.keys()==self.fallen_reference['tensors'].keys()
  train.write_json(output/'first_rollout_native_comparison.json',dict(all_exact=all_exact,checks=compared,
    reference=str(reference),reference_sha256=train.digest(reference),phase='Native GAE complete before outer optimizer attempt',new_PPO=0))
  if not all_exact:
   shutil.copyfile(output/'model_initial.pt',output/'model_stopped.pt')
   raise train.TrainingStopped()
 return result

def update(self):
 mean=self.algorithm.actor.mlp.mean
 before={n:p.detach().clone() for n,p in mean.recovery[-1].named_parameters()}
 losses=native_update(self)
 completed=len(self.fallen_rows)+1;steps=[float(self.algorithm.optimizer.state[p]['step']) for p in self.fallen_ids]
 checks=dict(exact_native_Adam_steps=all(a==b+12*completed for a,b in zip(steps,self.fallen_initial_steps,strict=True)),
 finite_output_parameters=all(torch.isfinite(p).all() for p in mean.recovery[-1].parameters()))
 observed=self.storage.observations['policy'];gravity=observed[...,9];active=self.learning
 gradients=[p.grad.square().sum() for p in mean.recovery[-1].parameters() if p.grad is not None]
 metrics=dict(push_frames=float(self.fallen_counts[0]),in_motion_fall_history_frames=float(self.fallen_counts[1]),recovered_events=float(self.fallen_counts[2]),
  observed_gravity_fallen_frames=int(((gravity>-.7)&active).sum()),
  output_parameter_update_L2=float(torch.stack([(p-before[n]).square().sum() for n,p in mean.recovery[-1].named_parameters()]).sum().sqrt()),
  output_gradient_L2=float(torch.stack(gradients).sum().sqrt()) if gradients else 0.,
  native_Adam_step=steps[0])
 row=dict(completed_updates=completed,learning_clock=1501+completed,metrics=metrics,checks={k:bool(v) for k,v in checks.items()},native_update_completed=True,created_unix=time.time())
 self.fallen_rows.append(row);self.fallen_counts.zero_()
 train.write_json(output/'fallen_output_updates.json',dict(schema='native-fallen-output-updates-v1',updates=self.fallen_rows,
  source_checkpoint=str(source),source_sha256=train.digest(source),wrapper_sha256=train.digest(Path(__file__)),
  interpretation='Exact native logs; in_motion_fall_history is a sticky episode flag, not instantaneous tilt or successful recovery.'))
 if logger is not None and logger.writer is not None:
  for name,value in metrics.items():logger.writer.add_scalar('FallenOutputTraining/'+name,value,row['learning_clock']-1)
  logger.writer.add_scalar('FallenOutputTraining/passed',int(all(checks.values())),row['learning_clock']-1);logger.writer.flush()
 # Signal after the native update, then return to outer accounting and sealing.
 if not all(checks.values()):os.kill(os.getpid(),15)
 return losses

exploration_init.install_axis_std_logging=logging
RecoveryRolloutMask.__init__=initialize
RecoveryRolloutMask.process_env_step=process
RecoveryRolloutMask.compute_returns=returns
RecoveryRolloutMask.update=update
train.main()

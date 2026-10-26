"""Native strict-contact rollout/GAE and substep evidence, zero PPO attempts."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
if '--diagnostic-rollout-only' not in sys.argv:
    raise ValueError('This script is a zero-optimizer rollout diagnostic only')
sys.argv.remove('--diagnostic-rollout-only')
output=Path(sys.argv[sys.argv.index('--output')+1])
import numpy as np
import torch
import train_robust_getup as train
from wheeled_algo.recovery_rollout import RecoveryRolloutMask

from wheeled_tasks.chassis import standing_contact_reward, standing_contact_continuity
native_install = standing_contact_reward.install
native_continuity_install = standing_contact_continuity.install
break_trace = []
environment = None

def capture_environment(env, weight, *, grounded_reset=False):
    global environment
    spec = native_install(env, weight, grounded_reset=grounded_reset)
    env.standing_contact_monitor.record = True
    environment = env
    return spec

def capture_continuity(env, cost):
    spec = native_continuity_install(env, cost)
    native_step = env.step

    def step(actions):
        result = native_step(actions)
        values = env.standing_contact_continuity_last
        break_trace.append({k: v.detach().cpu().numpy().copy() for k, v in values.items()})
        return result

    env.step = step
    return spec

standing_contact_reward.install = capture_environment
standing_contact_continuity.install = capture_continuity
native_returns=RecoveryRolloutMask.compute_returns
native_update=RecoveryRolloutMask.update

def array(value):return value.detach().cpu().contiguous().numpy()

def collect(self,last_observations):
    native_returns(self,last_observations)
    st=self.storage
    values={'observations__'+str(k):array(v) for k,v in st.observations.items()}
    for name in ('actions','dones','rewards','values','actions_log_prob','returns','advantages'):
        values[name]=array(getattr(st,name))
    values['learning_mask']=array(self.learning);values['bootstrap_mask']=array(self.bootstrap)
    for i,v in enumerate(st.distribution_params):values['distribution_param_'+str(i)]=array(v)
    for k,v in last_observations.items():values['last_observations__'+str(k)]=array(v)
    values['last_critic_values']=array(self.algorithm.critic(last_observations))
    if environment is None:
        raise ValueError('Contact objective/reset were not installed')
    if not hasattr(environment, 'standing_contact_continuity_contract') or len(break_trace) != 512:
        raise ValueError('Need all native support-break penalties')
    break_path = output / 'native_support_break_penalties.npz'
    np.savez_compressed(break_path, **{k: np.stack([row[k] for row in break_trace]) for k in break_trace[0]})
    support_path = output / 'native_quiet_support_rollout.npz'
    np.savez_compressed(support_path, **environment.standing_contact_monitor.arrays())
    target=output/'native_quiet_rollout_and_GAE.npz'
    if target.exists():raise ValueError('Never overwrite a captured real rollout')
    np.savez_compressed(target,**values)
    from export_robust_getup import load_mean as load_verified_mean
    from wheeled_algo.gap_recovery_exploration import GapRecoveryNoiseMean
    resume=Path(sys.argv[sys.argv.index('--resume')+1])
    state=torch.load(resume,map_location='cpu',weights_only=False)
    origin=state['infos']['policy_function_root']
    settings=state['infos']['settings']
    # Reference construction may initialize then restore CPU NN weights. Protect
    # all sampling RNGs even though this diagnostic exits without more sampling.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        reference,root_info=load_verified_mean(origin['checkpoint'])
        reference=GapRecoveryNoiseMean(reference,settings['quiet_exploration_fraction'],
            settings.get('quiet_wheel_exploration_fraction'),settings.get('gap_recovery_leg_exploration_fraction',1.)).to(st.device)
        reference.log_std.data.copy_(state['actor_state_dict']['mlp.log_std'].to(st.device))
    frames=st.observations['policy'].reshape(-1,35)
    mean_different=std_different=0;mean_peak=std_peak=0.
    with torch.inference_mode():
        for obs in frames.split(512):
            expected=reference(obs);actual=self.algorithm.actor.mlp(obs)
            mean_different+=int((expected[:,0]!=actual[:,0]).any(-1).sum())
            std_different+=int((expected[:,1]!=actual[:,1]).any(-1).sum())
            mean_peak=max(mean_peak,float((expected[:,0]-actual[:,0]).abs().max()))
            std_peak=max(std_peak,float((expected[:,1]-actual[:,1]).abs().max()))
    checks=dict(original_mean_bit_exact=mean_different==0,original_conditioned_std_bit_exact=std_different==0,
        clock1501=state['infos']['completed_updates']==1501,
        exactly_twelve_trainable_actor_tensors=[n for n,p in self.algorithm.actor.named_parameters() if p.requires_grad]
            ==['mlp.mean.'+head+'.'+layer+'.'+suffix for head in ('quiet','quiet_low','quiet_high') for layer in ('0','2') for suffix in ('weight','bias')])
    train.write_json(output/'native_function_readback.json',dict(checks=checks,frames=len(frames),
        source_checkpoint=origin['checkpoint'],source_sha256=origin['sha256'],mean_different=mean_different,std_different=std_different,
        mean_delta_peak=mean_peak,std_delta_peak=std_peak,new_PPO=0))
    if not all(checks.values()):raise ValueError('Native quiet source function failed: '+repr(checks))
    # The actual runner calls compute_returns before its alg.update method.
    # Raising here precedes the trainer's outer attempted-PPO accounting.
    # Preserve the initial checkpoint as an explicitly zero-update artifact;
    # no advanced rollout RNG is claimed as a training-resume state.
    initial=output/'model_initial.pt';sealed=output/'model_stopped.pt'
    shutil.copyfile(initial,sealed)
    hashes={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in values.items()}
    train.write_json(output/'rollout_diagnosis.json',dict(schema='native-zero-PPO-strict-contact-rollout-v1',
        status='captured_before_optimizer_attempt',standing_contact_training_contract=environment.standing_contact_training_contract,
        standing_contact_continuity_contract=environment.standing_contact_continuity_contract,
        break_penalties_data=str(break_path),break_penalties_sha256=train.digest(break_path),
        support_data=str(support_path),support_data_sha256=train.digest(support_path),
        reset_state=environment.grounded_standing_reset,
        data=str(target),data_sha256=train.digest(target),tensors=hashes,
        valid_rollout=self.metrics,native_GAE=True,native_sampling_and_likelihood=True,
        gamma=self.algorithm.gamma,lam=self.algorithm.lam,
        normalize_advantage_per_mini_batch=self.algorithm.normalize_advantage_per_mini_batch,
        optimizer_attempts=0,optimizer_steps=0,new_PPO=0,
        source_clock=1501,source_function_checks=checks,checkpoint_note='Exact native initial checkpoint copied; no rollout-advanced RNG resume claim.',
        wrapper_sha256=train.digest(Path(__file__)),created_unix=time.time()))
    raise train.TrainingStopped()

def forbid_update(self):raise RuntimeError('Zero-PPO diagnosis may never enter native PPO update')

RecoveryRolloutMask.compute_returns=collect
RecoveryRolloutMask.update=forbid_update
train.main()

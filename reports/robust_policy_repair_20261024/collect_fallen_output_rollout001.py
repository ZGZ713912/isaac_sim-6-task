"""Collect one real training rollout/GAE, then leave before any PPO attempt."""
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
    target=output/'native_fallen_rollout_and_GAE.npz'
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
        exactly_two_trainable_actor_tensors=[n for n,p in self.algorithm.actor.named_parameters() if p.requires_grad]
            ==['mlp.mean.recovery.4.weight','mlp.mean.recovery.4.bias'])
    train.write_json(output/'native_function_readback.json',dict(checks=checks,frames=len(frames),
        source_checkpoint=origin['checkpoint'],source_sha256=origin['sha256'],mean_different=mean_different,std_different=std_different,
        mean_delta_peak=mean_peak,std_delta_peak=std_peak,new_PPO=0))
    if not all(checks.values()):raise ValueError('Native fallen source function failed: '+repr(checks))
    # The actual runner calls compute_returns before its alg.update method.
    # Raising here precedes the trainer's outer attempted-PPO accounting.
    # Preserve the initial checkpoint as an explicitly zero-update artifact;
    # no advanced rollout RNG is claimed as a training-resume state.
    initial=output/'model_initial.pt';sealed=output/'model_stopped.pt'
    shutil.copyfile(initial,sealed)
    hashes={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in values.items()}
    train.write_json(output/'rollout_diagnosis.json',dict(schema='native-zero-PPO-fallen-output-rollout-v1',
        status='captured_before_optimizer_attempt',data=str(target),data_sha256=train.digest(target),tensors=hashes,
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

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
    target=output/'native_rollout_and_GAE.npz'
    if target.exists():raise ValueError('Never overwrite a captured real rollout')
    np.savez_compressed(target,**values)
    # The actual runner calls compute_returns before its alg.update method.
    # Raising here precedes the trainer's outer attempted-PPO accounting.
    # Preserve the initial checkpoint as an explicitly zero-update artifact;
    # no advanced rollout RNG is claimed as a training-resume state.
    initial=output/'model_initial.pt';sealed=output/'model_stopped.pt'
    shutil.copyfile(initial,sealed)
    hashes={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in values.items()}
    train.write_json(output/'rollout_diagnosis.json',dict(schema='native-zero-PPO-credit-rollout-v1',
        status='captured_before_optimizer_attempt',data=str(target),data_sha256=train.digest(target),tensors=hashes,
        valid_rollout=self.metrics,native_GAE=True,native_sampling_and_likelihood=True,
        gamma=self.algorithm.gamma,lam=self.algorithm.lam,
        normalize_advantage_per_mini_batch=self.algorithm.normalize_advantage_per_mini_batch,
        optimizer_attempts=0,optimizer_steps=0,new_PPO=0,
        source_clock=1501,checkpoint_note='Exact native initial checkpoint copied; no rollout-advanced RNG resume claim.',
        wrapper_sha256=train.digest(Path(__file__)),created_unix=time.time()))
    raise train.TrainingStopped()

def forbid_update(self):raise RuntimeError('Zero-PPO diagnosis may never enter native PPO update')

RecoveryRolloutMask.compute_returns=collect
RecoveryRolloutMask.update=forbid_update
train.main()

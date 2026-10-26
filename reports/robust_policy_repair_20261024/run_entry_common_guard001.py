"""Native entry PPO restricted to258 common coefficients, with phase stops.

This file is an experimental training entry point, not an authorization to run.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
if '--entry-recovery-only' not in sys.argv or '--recovery-only' not in sys.argv:
    raise ValueError('This entry point is limited to the explicit entry-only profile')
output=Path(sys.argv[sys.argv.index('--output')+1])
source=Path(sys.argv[sys.argv.index('--resume')+1])
import torch
from entry_common_guard001 import attach_common_gradient_mask,PhaseGuard
from wheeled_algo.recovery_rollout import RecoveryRolloutMask
import train_robust_getup as train

native_init=RecoveryRolloutMask.__init__
native_update=RecoveryRolloutMask.update
native_write=train.write_json
native_logging=train.install_axis_std_logging
latest={}
active_logger=None

def tensor_receipt(value):
    array=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())

def write(path,value):
    if Path(path).resolve()==(output/'progress.json').resolve():
        value['effective_trainable_entry_parameters']=258
        value['entry_gradient_restriction']='frozen-basis-common-rows-v1'
        value['phase_guard_receipt']=str(output/'entry_phase_guard.json')
    return native_write(path,value)

def install_logging(logger):
    global active_logger
    native_logging(logger)
    active_logger=logger

def initialize(self,algorithm,*,num_envs):
    native_init(self,algorithm,num_envs=num_envs)
    state=torch.load(source,map_location='cpu',weights_only=False)
    if state['infos']['schema']!='robust-getup-checkpoint-v3' or state['infos']['completed_updates']!=1501:
        raise ValueError('Use the authenticated1501 function-root optimizer initialization')
    mean=algorithm.actor.mlp.mean
    for p in mean.entry_recovery.parameters():
        if algorithm.optimizer.state.get(p):
            raise ValueError('Common-only initialization requires fresh entry Adam slots')
    self.common_hooks=attach_common_gradient_mask(mean.entry_recovery)
    self.common_guard=PhaseGuard(mean,ROOT/'reports/robust_policy_repair_20261024/entry_phase_path_dataset001.npz')
    self.common_guard_rows=[]
    self.common_source_clock=state['infos']['completed_updates']
    metrics,checks=self.common_guard.measure()
    if not all(checks.values()):
        raise ValueError('Neutral phase guard failed before native PPO')
    self.common_guard_initial=dict(metrics=metrics,checks=checks)
    for file in (Path(__file__),Path(__file__).with_name('entry_common_guard001.py')):
        target=output/'source'/file.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(file.read_bytes())

def update(self):
    global latest
    if not self.common_guard_rows:
        st=self.storage
        values={'observations/'+str(k):tensor_receipt(v) for k,v in st.observations.items()}
        for name in ('actions','dones','rewards','values','actions_log_prob'):
            values[name]=tensor_receipt(getattr(st,name))
        values['learning_mask']=tensor_receipt(self.learning);values['bootstrap_mask']=tensor_receipt(self.bootstrap)
        for i,v in enumerate(st.distribution_params):
            values['distribution_param_'+str(i)]=tensor_receipt(v)
        native_write(output/'first_rollout_hashes.json',dict(schema='common-only-first-native-rollout-v1',
            tensors=values,phase='After native GAE before first optimizer step',
            likelihood_sampling_GAE_native=True,wrapper_sha256=train.digest(Path(__file__))))
    losses=native_update(self)
    metrics,checks=self.common_guard.measure()
    latest=dict(metrics=metrics,checks=checks,
        completed_learning_clock=self.common_source_clock+len(self.common_guard_rows)+1,
        native_iteration=self.common_source_clock+len(self.common_guard_rows),created_unix=time.time())
    self.common_guard_rows.append(latest)
    native_write(output/'entry_phase_guard.json',dict(schema='native-common-only-phase-guard-v1',
        initial=self.common_guard_initial,updates=self.common_guard_rows,
        effective_parameters=258,stop_requested=not all(checks.values()),
        scope='Original development constraints. Does not qualify native physical behavior.',
        source_sha256=train.digest(source),wrapper_sha256=train.digest(Path(__file__)),
        mask_sha256=train.digest(Path(__file__).with_name('entry_common_guard001.py'))))
    # Write through the existing native RSL logger, including an update that
    # is about to stop before the runner reaches its ordinary log callback.
    if active_logger is not None and active_logger.writer is not None:
        for key,value in metrics.items():
            active_logger.writer.add_scalar('EntryCommonGuard/'+key,value,latest['native_iteration'])
        active_logger.writer.add_scalar('EntryCommonGuard/passed',int(all(checks.values())),latest['native_iteration'])
        active_logger.writer.flush()
    if not all(checks.values()):
        # The installed native trainer handler marks stop_requested. Returning
        # normally lets its outer wrapper charge/count this completed update,
        # seal the checkpoint and stop before collecting another rollout.
        os.kill(os.getpid(),signal.SIGTERM)
    return losses

train.write_json=write
train.install_axis_std_logging=install_logging
RecoveryRolloutMask.__init__=initialize
RecoveryRolloutMask.update=update
train.main()

"""Native PPO/Adam followed by an explicit constrained common-row projection.

This is projected Adam, not plain native PPO or an equivalent Adam migration.
The runtime mean architecture and observation/action contracts are unchanged.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import os
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
if '--entry-recovery-only' not in sys.argv or '--recovery-only' not in sys.argv:
    raise ValueError('Projected trial is limited to existing entry-only parameters')
output=Path(sys.argv[sys.argv.index('--output')+1]);source=Path(sys.argv[sys.argv.index('--resume')+1])
import torch
from rsl_rl.runners import OnPolicyRunner
from wheeled_algo.recovery_rollout import RecoveryRolloutMask
from wheeled_algo import exploration_init
import train_robust_getup as train
from entry_common_guard001 import attach_common_gradient_mask,PhaseGuard
from common_phase_projector001 import CommonPhaseProjector,VERSION

native_init=RecoveryRolloutMask.__init__;native_update=RecoveryRolloutMask.update
native_save=OnPolicyRunner.save;native_write=train.write_json
native_logging=exploration_init.install_axis_std_logging
context=None;logger=None

def tensor_receipt(value):
    a=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())

def logging(instance):
    global logger
    native_logging(instance);logger=instance

def write(path,value):
    if Path(path).resolve()==(output/'progress.json').resolve():
        value.update(effective_trainable_entry_parameters=258,entry_optimizer_method=VERSION,
            optimizer_interpretation='Native PPO gradients/Adam then offline coefficient projection after each complete update; not unmodified native PPO.')
    return native_write(path,value)

def initialize(self,algorithm,*,num_envs):
    global context
    native_init(self,algorithm,num_envs=num_envs)
    state=torch.load(source,map_location='cpu',weights_only=False)
    if state['infos']['schema']!='robust-getup-checkpoint-v3' or state['infos']['completed_updates']!=1501:
        raise ValueError('Use the exact authenticated1501 optimizer initialization')
    mean=algorithm.actor.mlp.mean
    if any(algorithm.optimizer.state.get(p) for p in mean.entry_recovery.parameters()):
        raise ValueError('This new branch requires explicitly fresh six entry Adam slots')
    self.projected_mean=mean;self.common_hooks=attach_common_gradient_mask(mean.entry_recovery)
    self.phase_guard=PhaseGuard(mean,ROOT/'reports/robust_policy_repair_20261024/entry_phase_path_dataset001.npz')
    self.projector=CommonPhaseProjector(mean,ROOT);self.projected_rows=[]
    metrics,checks=self.phase_guard.measure()
    if not all(checks.values()):raise ValueError('Initial phase constraints failed')
    self.projected_initial=dict(metrics=metrics,checks=checks);context=self
    for file in (Path(__file__),Path(__file__).with_name('common_phase_projector001.py'),Path(__file__).with_name('entry_common_guard001.py')):
        target=output/'source'/file.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(file.read_bytes())

def save(self,path,infos=None):
    info=dict(infos or {})
    if context is not None:
        info['entry_projected_optimizer']=dict(schema=VERSION,source_checkpoint=str(source.resolve()),
            source_sha256=train.digest(source),source_clock=1501,
            completed_projected_PPO_updates=len(context.projected_rows),updates=deepcopy(context.projected_rows),
            effective_parameters=258,wrapper_sha256=train.digest(Path(__file__)),
            projector_sha256=train.digest(Path(__file__).with_name('common_phase_projector001.py')),
            phase_dataset_sha256=train.digest(ROOT/'reports/robust_policy_repair_20261024/entry_phase_path_dataset001.npz'),
            algorithm='Unchanged native sampling/likelihood/GAE/PPO gradient/Adam with fixed-basis common gradient mask; final258 coefficients projected after each full native update.',
            Adam='Native Adam moments retained as projected-Adam state; this is an explicit optimizer intervention, not an equivalent ordinary-Adam continuation.',
            runtime='Existing NN weights only; no projection, trajectory or pose table at inference')
    return native_save(self,path,infos=info)

def update(self):
    if not self.projected_rows:
        st=self.storage;values={'observations/'+str(k):tensor_receipt(v) for k,v in st.observations.items()}
        for name in ('actions','dones','rewards','values','actions_log_prob'):values[name]=tensor_receipt(getattr(st,name))
        values['learning_mask']=tensor_receipt(self.learning);values['bootstrap_mask']=tensor_receipt(self.bootstrap)
        for i,v in enumerate(st.distribution_params):values['distribution_param_'+str(i)]=tensor_receipt(v)
        native_write(output/'first_rollout_hashes.json',dict(schema='projected-entry-first-native-rollout-v1',tensors=values,
            phase='After native GAE before any optimizer step or projection',wrapper_sha256=train.digest(Path(__file__))))
    before=deepcopy(self.projected_mean.entry_recovery.state_dict())
    losses=native_update(self)
    error=None
    try:projection=self.projector.project(self.projected_mean)
    except Exception as failure:
        # The native optimizer update has completed and remains chargeable.
        # Keep the last feasible actor, record the failed projection, seal via
        # the native outer wrapper and stop before collecting another rollout.
        error=repr(failure);self.projected_mean.entry_recovery.load_state_dict(before)
        projection=dict(projection_solved=False,error=error,solver_seconds=0.,rounds=[])
    metrics,checks=self.phase_guard.measure()
    checks.update(projection_solved=projection['projection_solved'],
        recorded_step_bound=projection.get('recorded_step_target_peak_rad',0.)<=.005)
    row=dict(learning_clock=1502+len(self.projected_rows),metrics=metrics,checks=checks,
        projection=projection,created_unix=time.time())
    self.projected_rows.append(row)
    native_write(output/'entry_phase_guard.json',dict(schema='native-projected-common-phase-guard-v1',
        initial=self.projected_initial,updates=self.projected_rows,stop_requested=not all(checks.values()),
        effective_parameters=258,method=VERSION,source_sha256=train.digest(source),
        interpretation='Development constraints and sampled recorded step bound; physical behavior requires separate native review.'))
    if logger is not None and logger.writer is not None:
        iteration=row['learning_clock']-1
        for name,value in metrics.items():logger.writer.add_scalar('EntryProjection/'+name,value,iteration)
        for name in ('solver_seconds','recorded_step_target_peak_rad','retained_initial_change_rms_rad'):
            if name in projection:logger.writer.add_scalar('EntryProjection/'+name,projection[name],iteration)
        logger.writer.add_scalar('EntryProjection/cut_rounds',len(projection['rounds']),iteration)
        logger.writer.add_scalar('EntryProjection/passed',int(all(checks.values())),iteration);logger.writer.flush()
    if not all(checks.values()):os.kill(os.getpid(),signal.SIGTERM)
    return losses

train.write_json=write
exploration_init.install_axis_std_logging=logging
OnPolicyRunner.save=save
RecoveryRolloutMask.__init__=initialize
RecoveryRolloutMask.update=update
train.main()

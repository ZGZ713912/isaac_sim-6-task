"""Sixteen native PPO updates with explicit uninterrupted-support objective."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
if '--zero-rollout-reference' not in sys.argv:
    raise ValueError('Require a preceding actual zero-PPO rollout')
index = sys.argv.index('--zero-rollout-reference')
reference = Path(sys.argv[index + 1]); del sys.argv[index:index + 2]
if int(sys.argv[sys.argv.index('--updates') + 1]) != 16:
    raise ValueError('Only the declared sixteen-update contact pilot is admitted')
if not {'--quiet-only', '--quiet-output-only', '--grounded-standing-reset', '--standing-contact-weight'} <= set(sys.argv) or any(
        k in sys.argv for k in ('--quiet-velocity-only', '--recovery-only', '--motion-output-only', '--entry-recovery-only')):
    raise ValueError('Only the authenticated existing quiet output scope is allowed')
output = Path(sys.argv[sys.argv.index('--output') + 1])
source = Path(sys.argv[sys.argv.index('--resume') + 1])
import torch
import train_robust_getup as train
from wheeled_algo import exploration_init
from wheeled_algo.quiet_output_checkpoint import LEARNED_KEYS
from wheeled_algo.recovery_rollout import RecoveryRolloutMask

native_init = RecoveryRolloutMask.__init__
native_returns = RecoveryRolloutMask.compute_returns
native_update = RecoveryRolloutMask.update
native_logging = exploration_init.install_axis_std_logging
logger = None


def receipt(value):
    a = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest())


def logging(instance):
    global logger
    logger = instance
    return native_logging(instance)


def initialize(self, algorithm, *, num_envs):
    native_init(self, algorithm, num_envs=num_envs)
    state = torch.load(source, map_location='cpu', weights_only=False)
    if state['infos']['schema'] != 'robust-getup-checkpoint-v7' or state['infos']['completed_updates'] != 1501:
        raise ValueError('Use authenticated1501 V7 initialization')
    names = [n for n, p in algorithm.actor.named_parameters() if p.requires_grad]
    if names != list(LEARNED_KEYS):
        raise ValueError('Unexpected trainable actor scope')
    self.quiet_rows = []
    self.quiet_reference = json.loads(reference.read_text())
    if self.quiet_reference['optimizer_attempts'] != 0 or self.quiet_reference['new_PPO'] != 0:
        raise ValueError('Reference includes optimizer work')
    spec = json.loads((output / 'standing_contact_training_contract.json').read_text())
    if spec != self.quiet_reference['standing_contact_training_contract']:
        raise ValueError('Pilot objective/reset differ from actual reference rollout')
    self.quiet_parameters = {n: p for n, p in algorithm.actor.named_parameters() if n in LEARNED_KEYS}
    self.quiet_initial_steps = {n: float(algorithm.optimizer.state[p]['step']) for n, p in self.quiet_parameters.items()}
    target = output / 'source' / Path(__file__).resolve().relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, target)


def returns(self, last_observations):
    result = native_returns(self, last_observations)
    if not self.quiet_rows:
        st = self.storage
        values = {'observations__' + str(k): receipt(v) for k, v in st.observations.items()}
        for name in ('actions', 'dones', 'rewards', 'values', 'actions_log_prob', 'returns', 'advantages'):
            values[name] = receipt(getattr(st, name))
        values['learning_mask'] = receipt(self.learning)
        values['bootstrap_mask'] = receipt(self.bootstrap)
        for i, value in enumerate(st.distribution_params): values['distribution_param_' + str(i)] = receipt(value)
        for k, value in last_observations.items(): values['last_observations__' + str(k)] = receipt(value)
        values['last_critic_values'] = receipt(self.algorithm.critic(last_observations))
        compared = {k: values.get(k) == v for k, v in self.quiet_reference['tensors'].items()}
        exact = all(compared.values()) and values.keys() == self.quiet_reference['tensors'].keys()
        train.write_json(output / 'first_rollout_native_comparison.json', dict(all_exact=exact, checks=compared,
            reference=str(reference), reference_sha256=train.digest(reference),
            phase='Native GAE completed before the outer attempted-PPO counter', new_PPO=0))
        if not exact:
            shutil.copyfile(output / 'model_initial.pt', output / 'model_stopped.pt')
            raise train.TrainingStopped()
    return result


def update(self):
    before = {n: p.detach().clone() for n, p in self.quiet_parameters.items()}
    losses = native_update(self)
    completed = len(self.quiet_rows) + 1
    steps = {n: float(self.algorithm.optimizer.state[p]['step']) for n, p in self.quiet_parameters.items()}
    checks = dict(exact_native_Adam_steps=all(v == self.quiet_initial_steps[n] + 12 * completed for n, v in steps.items()),
        finite_output_parameters=all(torch.isfinite(p).all() for p in self.quiet_parameters.values()))
    obs = self.storage.observations['policy']
    active = self.learning & (obs[..., :1].abs().squeeze(-1) < .01) & (obs[..., 2].abs() < .01) & (obs[..., 9] < -.94)
    gradients = [p.grad.square().sum() for p in self.quiet_parameters.values() if p.grad is not None]
    metrics = dict(observed_upright_zero_request_frames=int(active.sum()),
        encoder_displacement_rms_m=float((obs[..., 14:16][active].square().mean().sqrt() / 20.).detach()) if active.any() else 0.,
        output_parameter_update_L2=float(torch.stack([(p - before[n]).square().sum() for n, p in self.quiet_parameters.items()]).sum().sqrt().detach()),
        output_gradient_L2=float(torch.stack(gradients).sum().sqrt()) if gradients else 0.,
        native_Adam_step_min=min(steps.values()), native_Adam_step_max=max(steps.values()))
    row = dict(completed_updates=completed, learning_clock=1501 + completed, metrics=metrics,
        checks={k: bool(v) for k, v in checks.items()}, native_update_completed=True, created_unix=time.time())
    self.quiet_rows.append(row)
    train.write_json(output / 'quiet_output_updates.json', dict(schema='native-quiet-output-updates-v1',
        updates=self.quiet_rows, source_checkpoint=str(source), source_sha256=train.digest(source),
        wrapper_sha256=train.digest(__file__),
        interpretation='Encoder displacement is the existing actor observation estimate, not privileged physical XY; these training logs do not prove standing.'))
    if logger is not None and logger.writer is not None:
        for name, value in metrics.items(): logger.writer.add_scalar('QuietOutputTraining/' + name, value, row['learning_clock'] - 1)
        logger.writer.add_scalar('QuietOutputTraining/passed', int(all(checks.values())), row['learning_clock'] - 1)
        logger.writer.flush()
    if not all(checks.values()): os.kill(os.getpid(), 15)
    return losses


exploration_init.install_axis_std_logging = logging
RecoveryRolloutMask.__init__ = initialize
RecoveryRolloutMask.compute_returns = returns
RecoveryRolloutMask.update = update
train.main()

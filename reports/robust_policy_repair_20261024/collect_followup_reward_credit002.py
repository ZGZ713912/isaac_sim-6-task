"""Four real fixed-policy rollouts through the original moving-push window.

Native sampling, step processing and GAE are used, with no optimizer attempt.
The first window recomputes the original-reward GAE against frozen044. Further
windows use the same act/step/process/GAE sequence without a policy update.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
if '--diagnostic-rollout-only' not in sys.argv:
    raise ValueError('Only zero-optimizer diagnostics are authorized here')
sys.argv.remove('--diagnostic-rollout-only')
output = Path(sys.argv[sys.argv.index('--output') + 1])
import numpy as np
import torch
import train_robust_getup as train
from wheeled_algo.recovery_rollout import RecoveryRolloutMask
from wheeled_algo import exploration_init
from wheeled_tasks.chassis import followup_continuity_reward as reward_module

native_install = reward_module.install
native_logging = exploration_init.install_axis_std_logging
native_process = RecoveryRolloutMask.process_env_step
native_returns = RecoveryRolloutMask.compute_returns
env = logger = None


def install(environment, weight):
    global env
    env = environment
    if weight != 5.:
        raise ValueError('This diagnostic tests only the declared density5 hypothesis')
    return native_install(environment, weight)


def logging(instance):
    global logger
    logger = instance
    return native_logging(instance)


def array(value):
    return value.detach().cpu().contiguous().numpy()


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return value


def receipt(value):
    a = array(value)
    return dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest())


def process(self, obs, rewards, dones, extras):
    if not hasattr(self, 'continuity_rows'):
        self.continuity_rows = []
    last = env.training_reward_last
    base = last['native_reward'].clone()
    expected = base + last['bonus']
    if 'time_outs' in extras:
        bootstrap = self.algorithm.gamma * torch.squeeze(
            self.algorithm.transition.values * extras['time_outs'].unsqueeze(1).to(self.algorithm.device), 1)
        base += bootstrap
        expected += bootstrap
    index = self.storage.step
    native_process(self, obs, rewards, dones, extras)
    if not torch.equal(self.storage.rewards[index, :, 0], expected):
        raise ValueError('Native storage reward differs from declared bonus and timeout bootstrap')
    self.continuity_rows.append(dict(base_reward=base, bonus=last['bonus'].clone(),
        native_hold=last['native_followup_hold'].clone(), moving_fall=last['in_motion_fall'].clone()))


def tensors(self, last_observations):
    st = self.storage
    values = {'observations__' + str(k): receipt(v) for k, v in st.observations.items()}
    for name in ('actions', 'dones', 'rewards', 'values', 'actions_log_prob', 'returns', 'advantages'):
        values[name] = receipt(getattr(st, name))
    values['learning_mask'] = receipt(self.learning)
    values['bootstrap_mask'] = receipt(self.bootstrap)
    for i, value in enumerate(st.distribution_params):
        values['distribution_param_' + str(i)] = receipt(value)
    for key, value in last_observations.items():
        values['last_observations__' + str(key)] = receipt(value)
    values['last_critic_values'] = receipt(self.algorithm.critic(last_observations))
    return values


def collect(self, last_observations):
    native_returns(self, last_observations)
    st, alg = self.storage, self.algorithm
    source = Path(sys.argv[sys.argv.index('--resume') + 1])
    source_state = torch.load(source, map_location='cpu', weights_only=False)
    reference_path = ROOT / 'runs/robust_policy_repair_20261024/diagnosis044_motion_output_native_zero/rollout_diagnosis.json'
    reference = json.loads(reference_path.read_text())
    first_actual = tensors(self, last_observations)
    actual_rewards = st.rewards.clone()
    actual_returns, actual_advantages = st.returns.clone(), st.advantages.clone()
    baseline_rewards = torch.stack([row['base_reward'] for row in self.continuity_rows]).unsqueeze(-1)
    st.rewards.copy_(baseline_rewards)
    native_returns(self, last_observations)
    first_baseline = tensors(self, last_observations)
    baseline_checks = {k: first_baseline.get(k) == v for k, v in reference['tensors'].items()}
    baseline_checks['tensor_namespace_exact'] = first_baseline.keys() == reference['tensors'].keys()
    train.write_json(output / 'first_window_original_reward_comparison.json', dict(
        checks=baseline_checks, all_exact=all(baseline_checks.values()),
        reference=str(reference_path), reference_sha256=train.digest(reference_path),
        actual_reward_tensors=first_actual, original_reward_tensors=first_baseline,
        interpretation='Native reward substitution in the same storage; all sampling and transitions are original.', new_PPO=0))
    if not all(baseline_checks.values()):
        raise ValueError('First native window differs from044 before any optimizer attempt')
    st.rewards.copy_(actual_rewards)
    st.returns.copy_(actual_returns)
    st.advantages.copy_(actual_advantages)
    windows = []
    for window in range(4):
        rows = self.continuity_rows
        if len(rows) != st.num_transitions_per_env:
            raise ValueError('Missing native reward timer rows')
        bonus = torch.stack([row['bonus'] for row in rows])
        hold = torch.stack([row['native_hold'] for row in rows])
        moving = torch.stack([row['moving_fall'] for row in rows])
        actual_rewards, actual_returns, actual_advantages = st.rewards.clone(), st.returns.clone(), st.advantages.clone()
        st.rewards.copy_(torch.stack([row['base_reward'] for row in rows]).unsqueeze(-1))
        native_returns(self, last_observations)
        baseline_returns, baseline_advantages = st.returns.clone(), st.advantages.clone()
        st.rewards.copy_(actual_rewards)
        st.returns.copy_(actual_returns)
        st.advantages.copy_(actual_advantages)
        values = {name: array(value) for name, value in (
            ('bonus', bonus), ('native_hold', hold), ('moving_fall', moving),
            ('learning_mask', self.learning), ('observations', st.observations['policy']),
            ('rewards', st.rewards), ('returns', st.returns), ('advantages', st.advantages),
            ('values', st.values), ('dones', st.dones), ('bootstrap_mask', self.bootstrap),
            ('original_returns', baseline_returns), ('original_advantages', baseline_advantages),
            ('last_critic_values', alg.critic(last_observations)),
            ('original_rewards', torch.stack([row['base_reward'] for row in rows])))}
        target = output / f'window{window:02d}_native_reward_credit.npz'
        np.savez_compressed(target, **values)
        row = dict(window=window, steps=st.num_transitions_per_env, environments=st.num_envs,
            valid_frames=int(self.learning.sum()), bonus_sum=float(bonus.sum()),
            moving_fall_history_frames=int((moving & self.learning).sum()),
            observed_current_fallen_frames=int(((st.observations['policy'][..., 9] > -.7) & self.learning).sum()),
            moving_fall_bonus_frames=int(((bonus > 0.) & moving & self.learning).sum()),
            moving_fall_bonus_sum=float((bonus * moving).sum()),
            two_second_followup_frames=int(((hold >= 2.) & self.learning).sum()),
            moving_fall_return_credit_sum=float(((actual_returns - baseline_returns)[..., 0] * moving * self.learning).sum()),
            moving_fall_advantage_delta_sum=float(((actual_advantages - baseline_advantages)[..., 0] * moving * self.learning).sum()),
            reward_delta_peak=float(bonus.max()), data=str(target), data_sha256=train.digest(target),
            native_GAE=True, created_unix=time.time())
        windows.append(row)
        if logger is not None and logger.writer is not None:
            for key in ('valid_frames', 'bonus_sum', 'moving_fall_history_frames', 'observed_current_fallen_frames',
                        'moving_fall_bonus_frames', 'moving_fall_bonus_sum', 'two_second_followup_frames', 'reward_delta_peak',
                        'moving_fall_return_credit_sum', 'moving_fall_advantage_delta_sum'):
                logger.writer.add_scalar('FollowupCredit/' + key, row[key], window)
            logger.writer.flush()
        train.write_json(output / 'credit_windows_progress.json', dict(windows=windows, new_PPO=0,
            units='TensorBoard step is zero-based fixed-policy rollout window,512 policy steps per window.'))
        if window == 3:
            break
        st.clear()
        self.continuity_rows = []
        # Exact native runner rollout pipeline; logger book keeping has no
        # role in sampling, transitions, masks or GAE and is not replayed.
        for _ in range(st.num_transitions_per_env):
            actions = alg.act(last_observations)
            obs, rewards, dones, extras = env.step(actions.to(env.device))
            if not (torch.isfinite(rewards).all() and all(torch.isfinite(v).all() for v in obs.values())):
                raise ValueError('Nonfinite fixed-policy native transition')
            last_observations, rewards, dones = obs.to(alg.device), rewards.to(alg.device), dones.to(alg.device)
            alg.process_env_step(last_observations, rewards, dones, extras)
        native_returns(self, last_observations)
    actor_exact = all(torch.equal(value.detach().cpu(), source_state['actor_state_dict'][key])
        for key, value in alg.actor.state_dict().items())
    critic_exact = all(torch.equal(value.detach().cpu(), source_state['critic_state_dict'][key])
        for key, value in alg.critic.state_dict().items())
    from wheeled_algo.periodic_entry_actor import exact_state_equal
    optimizer = alg.optimizer.state_dict()
    expected_optimizer = source_state['optimizer_state_dict']
    expected_optimizer['param_groups'] = optimizer['param_groups']
    adam_exact = exact_state_equal(cpu_state(optimizer), cpu_state(expected_optimizer))
    readback = output / "native_state_readback.pt"
    torch.save(dict(actor_state_dict=cpu_state(alg.actor.state_dict()),
        critic_state_dict=cpu_state(alg.critic.state_dict()), optimizer_state_dict=cpu_state(optimizer),
        scope="Diagnostic readback only; no physical/RNG training-resume claim"), readback)
    checks = dict(first_window_original044_exact=all(baseline_checks.values()),
        all_four_native_windows=len(windows) == 4, actor_critic_Adam_exact=actor_exact and critic_exact and adam_exact,
        original_push_window_reached=sum(x['moving_fall_history_frames'] for x in windows) > 0,
        actual_moving_fall_bonus_exposed=sum(x['moving_fall_bonus_frames'] for x in windows) > 0,
        all_step_bonuses_bounded=max(x['reward_delta_peak'] for x in windows) <= .025000001)
    shutil.copyfile(output / 'model_initial.pt', output / 'model_stopped.pt')
    train.write_json(output / 'rollout_diagnosis.json', dict(schema='native-followup-credit-zero-PPO-v1',
        checks=checks, windows=windows, source_checkpoint=str(source), source_sha256=train.digest(source),
        source_clock=1501, native_sampling_and_GAE=True, optimizer_attempts=0, optimizer_steps=0, new_PPO=0,
        actor_exact=actor_exact, critic_exact=critic_exact, Adam_exact=adam_exact,
        native_state_readback=str(readback), native_state_readback_sha256=train.digest(readback),
        checkpoint_note='Exact native initial state copied; four-window physical/RNG state is not a training-resume claim.',
        reward_contract=env.training_reward_contract, wrapper_sha256=train.digest(Path(__file__)),
        qualification=None, created_unix=time.time()))
    raise train.TrainingStopped()


def forbid_update(self):
    raise RuntimeError('Zero-PPO credit diagnosis may never enter the optimizer')


reward_module.install = install
exploration_init.install_axis_std_logging = logging
RecoveryRolloutMask.process_env_step = process
RecoveryRolloutMask.compute_returns = collect
RecoveryRolloutMask.update = forbid_update
train.main()

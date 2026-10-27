"""True physical leg-speed maxima written once per complete RSL rollout.

Same calibrated extrema calculation as scratch_extrema_candidate001. Attach
after the physical observer; every1ms sample is retained. No action, reward,
mask, optimizer or simulator state is changed by this logger.
"""
import torch


class PhysicalPeakWitness:
    names = ('speed_rad_s', 'signed_speed_rad_s', 'applied_torque_nm',
        'motor_power_w', 'lane', 'leg_policy_index', 'physical_substep_one_based',
        'policy_interval_zero_based', 'phase_code', 'training_role_index',
        'episode_physical_time_s')

    def __init__(self, num_envs, rollout_steps, device):
        if num_envs <= 0 or rollout_steps <= 0:
            raise ValueError('Need positive native population and complete rollout length')
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.rows = torch.zeros(rollout_steps, len(self.names), device=device)
        self.calls = 0

    def observe(self, dq, torque, *, passive, recovering, finished, role, episode_step):
        if dq.shape != (5, self.num_envs, 4) or torque.shape != dq.shape:
            raise ValueError('Require matching true1ms leg speed and actual torque samples')
        if any(value.shape != (self.num_envs,) for value in
                (passive, recovering, finished, role, episode_step)):
            raise ValueError('Need the owning pre-interval phase for every native lane')
        if self.calls >= self.rollout_steps:
            raise ValueError('Flush the previous complete rollout before observing another')
        index = dq.detach().abs().reshape(-1).argmax()
        physical = index // (self.num_envs*4)
        lane = (index // 4) % self.num_envs
        leg = index % 4
        speed, tau = dq[physical, lane, leg], torque[physical, lane, leg]
        phase = torch.where(finished[lane], 3, torch.where(passive[lane], 0,
            torch.where(recovering[lane], 1, 2)))
        value = torch.stack((speed.abs(), speed, tau, speed*tau, lane, leg,
            physical+1, index.new_tensor(self.calls), phase, role[lane],
            episode_step[lane]*.005+(physical+1)*.001))
        self.rows[self.calls].copy_(value.detach())
        self.calls += 1

    def snapshot(self):
        if self.calls != self.rollout_steps:
            raise ValueError('Need every native interval of the completed rollout')
        if not torch.isfinite(self.rows).all():
            raise ValueError('A native peak witness is nonfinite')
        row = self.rows[self.rows[:, 0].argmax()].tolist()
        return dict(zip(self.names, row, strict=True))

    def write(self, writer, completed_update):
        if writer is None or completed_update <= 0:
            raise ValueError('Need the original RSL writer and completed positive update')
        result = self.snapshot()
        for name, value in result.items():
            writer.add_scalar('PhysicalPeakWitness/'+name, value, completed_update-1)
        self.rows.zero_()
        self.calls = 0
        return result


class StandingRolloutExtrema:
    thresholds_rad_s = (3., 10., 50.)

    def __init__(self, num_envs, device, rollout_steps=None):
        if num_envs <= 0:
            raise ValueError('Need a positive native environment count')
        self.num_envs = num_envs
        self.thresholds = torch.tensor(self.thresholds_rad_s, device=device)
        self.peak_per_leg = torch.zeros(4, device=device)
        self.counts = torch.zeros(3, dtype=torch.int64, device=device)
        self.policy_calls = 0
        self.witness = (PhysicalPeakWitness(num_envs, rollout_steps, device)
            if rollout_steps is not None else None)

    def observe(self, dq):
        if dq.shape != (5, self.num_envs, 4) or not dq.is_floating_point():
            raise ValueError('Require all five1ms samples, every native environment and four leg motors')
        magnitude = dq.detach().abs()
        # RSL samples under inference_mode, then logs after leaving that mode.
        # Retain this ordinary persistent buffer across both phases.
        self.peak_per_leg.copy_(torch.maximum(self.peak_per_leg, magnitude.amax(dim=(0, 1))))
        self.counts += (magnitude[..., None] >= self.thresholds).sum(dim=(0, 1, 2))
        self.policy_calls += 1

    def snapshot(self, expected_calls):
        if expected_calls <= 0 or self.policy_calls != expected_calls:
            raise ValueError('Require the entire completed native rollout')
        if not torch.isfinite(self.peak_per_leg).all():
            raise ValueError('Nonfinite native physical leg-speed extrema')
        result = dict(peak_leg_speed_rad_s=float(self.peak_per_leg.max()),
            policy_decisions=self.policy_calls,
            observed_leg_motor_samples=self.policy_calls*5*self.num_envs*4)
        for axis, value in enumerate(self.peak_per_leg.tolist()):
            result[f'peak_leg{axis}_speed_rad_s'] = value
        for threshold, count in zip(self.thresholds_rad_s, self.counts.tolist()):
            result[f'leg_motor_samples_at_or_above_{int(threshold)}_rad_s'] = count
        if self.witness is not None:
            result.update({"witness_"+name: value for name, value in self.witness.snapshot().items()})
        return result

    def write(self, writer, completed_update, expected_calls):
        if writer is None or completed_update <= 0:
            raise ValueError('Need the actual RSL writer and a committed positive PPO update')
        values = self.snapshot(expected_calls)
        for name, value in values.items():
            if not name.startswith('witness_'):
                writer.add_scalar('ScratchStandingExtrema/'+name, value, completed_update-1)
        if self.witness is not None:
            self.witness.write(writer, completed_update)
        self.peak_per_leg.zero_()
        self.counts.zero_()
        self.policy_calls = 0
        return values


def install(env, *, rollout_steps=128):
    if not hasattr(env, 'physical_standing_reward') or hasattr(env, 'standing_rollout_extrema'):
        raise ValueError('Attach once after the physical standing observer')
    observer = StandingRolloutExtrema(env.num_envs, env.device, rollout_steps)
    native_step = env.step

    def step(actions):
        phase = dict(passive=env.passive.clone(), recovering=env.recovery_active.clone(),
            finished=env.finished.clone(), role=env.training_group.clone(),
            episode_step=env.episode_length_buf.clone())
        result = native_step(actions)
        samples = env.physical_standing_last
        observer.observe(samples['dq'])
        observer.witness.observe(samples['dq'], samples['torque'], **phase)
        return result

    env.step = step
    env.standing_rollout_extrema = observer
    return observer

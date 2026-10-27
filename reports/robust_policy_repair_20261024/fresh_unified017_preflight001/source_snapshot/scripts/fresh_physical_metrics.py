"""True physical leg-speed maxima written once per complete RSL rollout.

Same calibrated extrema calculation as scratch_extrema_candidate001. Attach
after the physical observer; every1ms sample is retained. No action, reward,
mask, optimizer or simulator state is changed by this logger.
"""
import torch


class StandingRolloutExtrema:
    thresholds_rad_s = (3., 10., 50.)

    def __init__(self, num_envs, device):
        if num_envs <= 0:
            raise ValueError('Need a positive native environment count')
        self.num_envs = num_envs
        self.thresholds = torch.tensor(self.thresholds_rad_s, device=device)
        self.peak_per_leg = torch.zeros(4, device=device)
        self.counts = torch.zeros(3, dtype=torch.int64, device=device)
        self.policy_calls = 0

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
        return result

    def write(self, writer, completed_update, expected_calls):
        if writer is None or completed_update <= 0:
            raise ValueError('Need the actual RSL writer and a committed positive PPO update')
        values = self.snapshot(expected_calls)
        for name, value in values.items():
            writer.add_scalar('ScratchStandingExtrema/'+name, value, completed_update-1)
        self.peak_per_leg.zero_()
        self.counts.zero_()
        self.policy_calls = 0
        return values


def install(env):
    if not hasattr(env, 'physical_standing_reward') or hasattr(env, 'standing_rollout_extrema'):
        raise ValueError('Attach once after the physical standing observer')
    observer = StandingRolloutExtrema(env.num_envs, env.device)
    native_step = env.step

    def step(actions):
        result = native_step(actions)
        observer.observe(env.physical_standing_last['dq'])
        return result

    env.step = step
    env.standing_rollout_extrema = observer
    return observer

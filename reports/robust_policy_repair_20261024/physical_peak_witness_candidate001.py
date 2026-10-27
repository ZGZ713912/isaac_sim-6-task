"""Unadopted scalar witness for the real highest leg-speed sample of a rollout.

Every interval contributes all four legs at all five physical boundaries. The
selected torque and control phase share that exact sample/owning interval.
Persistent buffers are allocated outside RSL inference_mode for safe logging.
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

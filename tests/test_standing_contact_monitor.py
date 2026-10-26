import math
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.standing_contact_monitor import StandingContactMonitor


class ContactEnv:
    def __init__(self, missing_substep=None):
        self.dt, self.decimation, self.wheel_ids = .001, 5, [0, 1]
        self.origins = torch.zeros(1, 3)
        pose = torch.tensor([[[0., 0., .06, math.sqrt(.5), 0., 0., math.sqrt(.5)]]]).expand(1, 2, 7).clone()
        self.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)))
        self.contact_force = torch.zeros(1, 2, 3)
        self.missing_substep = missing_substep
        self.calls = 0
        self.reward = torch.tensor([.3])

    def wheel_centers(self):
        return self.robot.data.body_link_pose_w.torch[..., :3]

    def closure_gap(self):
        self.calls += 1
        return torch.zeros(1)

    def step(self, actions):
        for i in range(self.decimation):
            self.contact_force[..., 2] = 50.
            if i == self.missing_substep:
                self.contact_force[:, 1, 2] = 0.
            self.closure_gap()
        return {'policy': actions}, self.reward, torch.zeros(1, dtype=torch.bool), {}


def test_monitor_retains_reward_actions_rng_and_all_five_physics_samples():
    env = ContactEnv()
    monitor = StandingContactMonitor(env, record=True)
    actions = torch.zeros(1, 6)
    rng = torch.get_rng_state().clone()
    result = env.step(actions)
    assert result[0]['policy'] is actions and result[1] is env.reward
    assert torch.equal(rng, torch.get_rng_state()) and env.calls == 5
    assert monitor.last['contact_every_ms'].tolist() == [True]
    assert monitor.arrays()['force'].shape == (1, 5, 1, 2, 3)


def test_missing_earlier_substep_fails_even_when_last_substep_has_two_contacts():
    env = ContactEnv(missing_substep=1)
    monitor = StandingContactMonitor(env)
    env.step(torch.zeros(1, 6))
    assert (env.contact_force[..., 2] == 50.).all()
    assert monitor.last['contact_every_ms'].tolist() == [False]
    assert monitor.last['missing_ms'].tolist() == [1]
    assert monitor.last['normal_load_min_n'].tolist() == [[50., 0.]]


def test_double_installation_is_rejected():
    env = ContactEnv()
    StandingContactMonitor(env)
    with pytest.raises(ValueError, match='once'):
        StandingContactMonitor(env)

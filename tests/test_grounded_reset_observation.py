"""A reset must publish its final pose before the first actor call.

The fake native views implement only rigid transforms. These tests exercise
the real HistoryStack and reset wrapper, without starting Isaac or physics.
SCRATCH_RESET_SOURCE selects a saved candidate while live training pins src.
"""
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.v40.core import HistoryStack


def rotate(quaternion, vector):
    xyz = quaternion[..., :3]
    cross = torch.linalg.cross(xyz, vector, dim=-1)
    return vector + 2 * (quaternion[..., 3:] * cross + torch.linalg.cross(xyz, cross, dim=-1))


class FrameBuilder:
    @staticmethod
    def get_observations(env):
        return {"policy": env.history.update(env.native_frame(), env.tick)}


class ResetEnv:
    def __init__(self):
        self.num_envs, self.device, self.tick, self.dt, self.policy_dt = 3, "cpu", 17, .001, .005
        self.origins = torch.zeros(3, 3)
        self.recovery_lane = torch.zeros(3, dtype=torch.bool)
        self.pending_command = torch.tensor([[0., 0., .305]]).expand(3, -1).clone()
        self.previous_potential = torch.zeros(3)
        self.standing_observer = self.standing_sensor_winding = None
        self.pitch = torch.tensor([0., -.5, 1.]) * math.pi / 180.
        self.wheel_ids, self.ids = [0, 1], list(range(6))
        self.pose = torch.tensor([[0., 0., .305, 0., 0., 0., 1.]]).expand(3, -1).clone()
        self.history = HistoryStack(3, "cpu", length=1, dim=35)
        self.domain = SimpleNamespace(imu_mount=None, bias=torch.zeros(3, 18),
            sensor_history=torch.zeros(3, 3, 18), observed=torch.zeros(3, 18), last_sensor_tick=17)
        self.data = SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=self.pose),
            joint_pos=SimpleNamespace(torch=torch.zeros(3, 6)))
        self.robot = SimpleNamespace(data=self.data, write_root_link_pose_to_sim_index=self.write_pose)
        self.update_views()
        self.prime(torch.arange(3))

    def update_views(self):
        q = self.pose[:, None, 3:].expand(-1, 2, -1)
        relative = self.pose.new_tensor([[0., .2, -.235], [0., -.2, -.237]]).expand(3, -1, -1)
        self.centers = self.pose[:, None, :3] + rotate(q, relative)
        x, y, z, w = q.unbind(-1)
        s = math.sqrt(.5)
        wheel_q = torch.stack(((w+x)*s, (y+z)*s, (z-y)*s, (w-x)*s), -1)
        self.data.body_link_pose_w = SimpleNamespace(torch=torch.cat((self.centers, wheel_q), -1))

    def native_frame(self):
        frame = torch.zeros(3, 35)
        frame[:, 3] = 5 * self.pending_command[:, 2]
        x, y, z, w = self.pose[:, 3:].unbind(-1)
        frame[:, 7:10] = torch.stack((2*(w*y-x*z), -2*(w*x+y*z), -1+2*(x*x+y*y)), -1)
        frame[:, 28] = 1.
        return frame

    def prime(self, ids):
        self.history.reset(ids)
        frame = FrameBuilder.get_observations(self)["policy"]
        self.domain.sensor_history[:, ids] = frame[ids, 4:22][None]
        self.domain.observed[ids] = frame[ids, 4:22]

    def write_pose(self, *, root_pose, env_ids):
        self.pose[env_ids] = root_pose
        self.update_views()

    def wheel_centers(self):
        return self.centers

    def potential(self):
        return torch.zeros(3)

    def reset(self, ids):
        self.pose[ids] = self.pose.new_tensor([0., 0., .305, 0., 0., 0., 1.])
        self.update_views()
        self.prime(ids)  # Parent reset reads the unperturbed birth first.
        self.pose[ids, 4] = (self.pitch[ids] / 2).sin()
        self.pose[ids, 6] = (self.pitch[ids] / 2).cos()
        self.update_views()  # Scratch reset applies pitch after that read.


@pytest.fixture
def reset_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "wheeled_tasks.chassis.env", SimpleNamespace(ChassisEnv=FrameBuilder))
    monkeypatch.setitem(sys.modules, "wheeled_tasks.chassis.root_kinematics",
                        SimpleNamespace(update_robot_data=lambda env, dt: env.update_views()))
    path = Path(os.environ.get("SCRATCH_RESET_SOURCE",
        Path(__file__).resolve().parents[1] / "src/wheeled_tasks/chassis/grounded_standing_reset.py"))
    spec = importlib.util.spec_from_file_location("wheeled_tasks.chassis.grounded_reset_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initial_placement_publishes_roll_before_first_actor(reset_module):
    env = ResetEnv()
    reset_module.install(env)
    torch.testing.assert_close(env.domain.observed[:, 3:6], env.native_frame()[:, 7:10], rtol=0., atol=1e-7)


def test_final_pitches_reach_first_actor_without_a_physics_step(reset_module):
    env = ResetEnv()
    reset_module.install(env)
    env.reset(torch.arange(3))
    frame = FrameBuilder.get_observations(env)["policy"]
    torch.testing.assert_close(frame[:, 7:10], env.native_frame()[:, 7:10], rtol=0., atol=1e-7)
    torch.testing.assert_close(env.domain.observed[:, 3:6], frame[:, 7:10], rtol=0., atol=1e-7)
    assert env.tick == 17


def test_partial_reset_preserves_other_lanes_tick_and_rng(reset_module):
    env = ResetEnv()
    reset_module.install(env)
    env.reset(torch.arange(3))
    before = env.history.buffer.clone()
    sensors_before = env.domain.sensor_history.clone()
    rng = torch.random.get_rng_state().clone()
    env.pitch[1] = math.radians(.25)
    env.reset(torch.tensor([1]))
    frame = FrameBuilder.get_observations(env)["policy"]
    torch.testing.assert_close(frame[1, 7:10], env.native_frame()[1, 7:10], rtol=0., atol=1e-7)
    assert torch.equal(env.history.buffer[[0, 2]], before[[0, 2]])
    assert torch.equal(env.domain.sensor_history[:, [0, 2]], sensors_before[:, [0, 2]])
    assert torch.equal(torch.random.get_rng_state(), rng) and env.tick == 17
    assert torch.equal(FrameBuilder.get_observations(env)["policy"], frame)

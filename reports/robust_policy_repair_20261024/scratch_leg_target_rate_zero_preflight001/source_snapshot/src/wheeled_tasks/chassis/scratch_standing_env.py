"""Reset-only standing curriculum for a completely new six-action policy."""
from __future__ import annotations

import math
import torch

from .robust_getup_env import RobustGetupEnv
from .root_kinematics import update_robot_data


class ScratchStandingEnv(RobustGetupEnv):
    def __init__(self, *args, all_heights_after=512, **kwargs):
        self.scratch_ready = False
        self.all_heights_after = all_heights_after
        self.evaluation_cases = None
        super().__init__(*args, **kwargs)
        mapping = self.bank["diagnostic_height_starts"]["mapping"]
        self.height_pose = {height: mapping[name] for height, name in
                            ((.23, "stand_low"), (.305, "stand"), (.42, "stand_high"))}
        self.scratch_ready = True
        self.recovery_lane.zero_()
        from .grounded_standing_reset import install
        self.reset(torch.arange(self.num_envs, device=self.device))
        install(self)

    def reset(self, ids):
        if not self.scratch_ready:
            return super().reset(ids)
        cases = [dict(pose_index=self.height_pose[.305], recovery=False, command=[0., 0., .305])
                 for _ in range(self.num_envs)]
        if self.evaluation_cases is None:
            heights = (.305,) if self.actor_updates < self.all_heights_after else (.23, .305, .42)
            selected = torch.randint(len(heights), (len(ids),), generator=self.generator, device=self.device).tolist()
            for lane, index in zip(ids.tolist(), selected):
                height = heights[index]
                cases[lane] = dict(pose_index=self.height_pose[height], recovery=False, command=[0., 0., height])
        else:
            cases = self.evaluation_cases
        self.fixed_cases = cases
        try:
            super().reset(ids)
        finally:
            self.fixed_cases = None
        # Training gets small independently sampled pitch disturbances. These
        # are initial conditions only, followed by the native six-motor policy.
        if self.evaluation_cases is None:
            angle = (self.random(len(ids)) * 2. - 1.) * math.radians(.5)
        else:
            angle = torch.tensor([cases[lane]["pitch_deg"] for lane in ids.tolist()], device=self.device) * math.pi / 180.
        root = self.robot.data.root_link_pose_w.torch[ids].clone()
        x, y, z, w = root[:, 3:].unbind(-1)
        s, c = (angle / 2.).sin(), (angle / 2.).cos()
        root[:, 3:] = torch.stack((x*c-z*s, y*c+w*s, z*c+x*s, w*c-y*s), -1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=ids)
        update_robot_data(self, self.dt)
        self.previous_potential[ids] = self.potential()[ids]

    def sample_commands(self, ids):
        # No mid-episode command changes in this standing foundation stage.
        if not self.scratch_ready:
            return super().sample_commands(ids)


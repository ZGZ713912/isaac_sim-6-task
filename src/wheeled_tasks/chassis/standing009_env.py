"""Standing births and command-only curriculum; no scripted motor trajectory."""
import math

import torch

from .scratch_standing_env import ScratchStandingEnv
from .standing009_curriculum import case, commands_at


class Standing009TrackingEnv(ScratchStandingEnv):
    def __init__(self, *args, **kwargs):
        self.course_ready = False
        super().__init__(*args, **kwargs)
        self.course = self.getup_cfg["standing009_curriculum"]
        self.course_updates = 0
        self.course_cases = [case(self.course, "stand") for _ in range(self.num_envs)]
        self.course_ready = True
        self.reset(torch.arange(self.num_envs, device=self.device))

    def reset(self, ids):
        if not self.course_ready:
            return super().reset(ids)
        if self.evaluation_cases is not None:
            for i in ids.tolist():
                self.course_cases[i] = self.evaluation_cases[i]
            return super().reset(ids)
        progress = min(1., self.course_updates/self.course["tracking"]["ramp_updates"])
        heights = (.305,) if progress < 1. else (.23, .305, .42)
        pick = torch.randint(len(heights), (len(ids),), generator=self.generator, device=self.device).tolist()
        pitch = ((self.random(len(ids))*2.-1.)*math.radians(.5)).tolist()
        for lane, index, angle in zip(ids.tolist(), pick, pitch):
            role = self.course["tracking"]["roles_per20"][lane % 20]
            height = heights[index] if role in ("stand", "forward", "reverse", "left", "right") else .305
            row = case(self.course, role, height, math.degrees(angle), progress)
            row["pose_index"] = self.height_pose[height]
            self.course_cases[lane] = row
        self.evaluation_cases = self.course_cases
        try:
            super().reset(ids)
        finally:
            self.evaluation_cases = None

    @torch.inference_mode()
    def step(self, actions):
        # This advances upper-level requests only. All six motor actions are
        # still produced by the live policy with native feedback and PD.
        elapsed = self.episode_length_buf*self.policy_dt
        self.pending_command.copy_(commands_at(self.course_cases, elapsed, self.course))
        self.goal_command.copy_(self.pending_command)
        self.commands.copy_(self.pending_command)
        self.raw_height.copy_(self.commands[:, 2])
        return super().step(actions)

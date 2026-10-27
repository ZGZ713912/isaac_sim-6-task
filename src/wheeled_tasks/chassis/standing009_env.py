"""Standing births and command-only curriculum; no scripted motor trajectory."""
import math

import torch

from .scratch_standing_env import ScratchStandingEnv
from .standing009_curriculum import case, commands_at
from .standing009_retention import enabled, critic_anchor


class Standing009TrackingEnv(ScratchStandingEnv):
    def __init__(self, *args, **kwargs):
        self.course_ready = False
        super().__init__(*args, **kwargs)
        self.course = self.getup_cfg["standing009_curriculum"]
        self.course_updates = 0
        self.course_cases = [case(self.course, "stand") for _ in range(self.num_envs)]
        self.course_anchor = torch.zeros(self.num_envs, 2, device=self.device)
        self.course_anchor_for_step = self.course_anchor.clone()
        self.course_quiet = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.course_height_learning = torch.zeros_like(self.course_quiet)
        self.course_role_masks = {}
        self.course_role_signature = None
        self.course_ready = True
        self.reset(torch.arange(self.num_envs, device=self.device))

    def finish_course_reset(self, ids, evaluation):
        self.course_anchor[ids] = self.robot.data.root_link_pose_w.torch[ids, :2]-self.origins[ids, :2]
        self.course_quiet[ids] = True
        roles = tuple(c["role"] for c in self.course_cases)
        if roles != self.course_role_signature:
            self.course_role_masks = {role: torch.tensor([v == role for v in roles], device=self.device)
                                      for role in sorted(set(roles))}
            learning = self.course.get("repair", {}).get("height_learning_roles", [])
            self.course_height_learning = torch.tensor([v in learning for v in roles], device=self.device)
            self.course_role_signature = roles
        if enabled(self.course) and not evaluation:
            cfg = self.course["repair"]
            seconds = [cfg["standing_episode_seconds"] if roles[i] == "stand" else cfg["motion_episode_seconds"]
                       for i in ids.tolist()]
            self.episode_limits[ids] = torch.tensor([round(v/self.policy_dt) for v in seconds], device=self.device)

    def reset(self, ids):
        if not self.course_ready:
            return super().reset(ids)
        if self.evaluation_cases is not None:
            for i in ids.tolist():
                self.course_cases[i] = self.evaluation_cases[i]
            result = super().reset(ids)
            self.finish_course_reset(ids, True)
            return result
        progress = min(1., self.course_updates/self.course["tracking"]["ramp_updates"])
        heights = (.305,) if progress < 1. or enabled(self.course) else (.23, .305, .42)
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
        self.finish_course_reset(ids, False)

    def get_observations(self):
        observations = super().get_observations()
        if self.course_ready and enabled(self.course):
            if not self.cfg.get("zero_critic_wheel_positions"):
                raise ValueError("Retention critic needs the declared unused wheel-position entries")
            offset = self.cfg["actor_frame_dim"]+3+1+2
            columns = (offset+self.ids[2], offset+self.ids[5])
            error = self.robot.data.root_link_pose_w.torch[:, :2]-self.origins[:, :2]-self.course_anchor
            observations["critic"] = critic_anchor(observations["critic"], error, columns)
        return observations

    @torch.inference_mode()
    def step(self, actions):
        # This advances upper-level requests only. All six motor actions are
        # still produced by the live policy with native feedback and PD.
        elapsed = self.episode_length_buf*self.policy_dt
        next_commands = commands_at(self.course_cases, elapsed, self.course)
        quiet = (next_commands[:, :2].abs() < .01).all(-1)
        entered = quiet & ~self.course_quiet
        self.course_anchor[entered] = (self.robot.data.root_link_pose_w.torch[:, :2]-self.origins[:, :2])[entered]
        self.course_quiet.copy_(quiet)
        self.course_anchor_for_step = self.course_anchor.clone()
        self.pending_command.copy_(next_commands)
        self.goal_command.copy_(self.pending_command)
        self.commands.copy_(self.pending_command)
        self.raw_height.copy_(self.commands[:, 2])
        return super().step(actions)

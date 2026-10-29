"""Standard grounded 305mm births, with an independent balance/motion task."""
import math

import torch

from .robust_getup_env import RobustGetupEnv
from .root_kinematics import update_robot_data
from .upright_locomotion import training_case


class UprightLocomotionEnv(RobustGetupEnv):
    def __init__(self, *args, all_heights_after=None, **kwargs):
        self.upright_ready = False
        self.evaluation_cases = None
        super().__init__(*args, **kwargs)
        self.upright_task = self.getup_cfg["upright_locomotion"]
        mapping = self.bank["diagnostic_height_starts"]["mapping"]
        self.height_pose = {.305: mapping["stand"]}
        self.upright_ready = True
        self.recovery_lane.zero_()
        self.reset(torch.arange(self.num_envs, device=self.device))
        from .grounded_standing_reset import install
        install(self, include_motion=True)
        self.startup_report["upright_locomotion"] = dict(
            task=self.upright_task, passive_start=False, recovery_enabled=False,
            six_trainable_actions=True, fixed_leg_targets=False,
            nonwheel_contact="terminal from any native 1ms sample",
            state_writes="episode reset only")
        self.startup_report["unified_getup"].update(
            task_active=False, contact_support="nonwheel ground contact is terminal",
            motor_effort_during_settle="no passive settling phase")

    def reset(self, ids):
        if not self.upright_ready:
            return super().reset(ids)
        cases = (self.evaluation_cases if self.evaluation_cases is not None else
                 [training_case(self.upright_task, i, self.height_pose[.305])
                  for i in range(self.num_envs)])
        if any(cases[i]["recovery"] or cases[i]["command"][2] != .305 for i in ids.tolist()):
            raise ValueError("Upright task accepts only standard 305mm non-recovery births")
        self.fixed_cases = cases
        try:
            super().reset(ids)
        finally:
            self.fixed_cases = None
        if self.evaluation_cases is None:
            lo, hi = self.upright_task["training_pitch_range_deg"]
            angle = (lo + (hi-lo) * self.random(len(ids))) * math.pi / 180.
        else:
            angle = torch.tensor([cases[i]["pitch_deg"] for i in ids.tolist()], device=self.device) * math.pi / 180.
        root = self.robot.data.root_link_pose_w.torch[ids].clone()
        x, y, z, w = root[:, 3:].unbind(-1)
        s, c = (angle / 2.).sin(), (angle / 2.).cos()
        root[:, 3:] = torch.stack((x*c-z*s, y*c+w*s, z*c+x*s, w*c-y*s), -1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=ids)
        update_robot_data(self, self.dt)
        self.previous_potential[ids] = self.potential()[ids]

    def sample_commands(self, ids):
        if not self.upright_ready:
            return super().sample_commands(ids)

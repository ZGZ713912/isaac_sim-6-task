"""One validated prone family; six-motor RL owns every active action."""
import torch

from .robust_getup_env import RobustGetupEnv
from .root_kinematics import update_robot_data
from .standard_prone import physical_knee_degrees, route_potential, support_error


class StandardProneEnv(RobustGetupEnv):
    def __init__(self, *args, all_heights_after=None, **kwargs):
        self.prone_ready = False
        self.evaluation_cases = None
        super().__init__(*args, **kwargs)
        self.prone_task = self.getup_cfg["standard_prone"]
        matches = [i for i, p in enumerate(self.poses) if p["name"] == self.prone_task["reset_pose_name"]]
        if len(matches) != 1:
            raise ValueError("Need one authenticated standard prone reset geometry")
        self.initialization_pose_index = matches[0]
        self.height_pose = {.305: self.bank["diagnostic_height_starts"]["mapping"]["stand"]}
        self.prone_start_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.prone_ready = True
        self.reset(torch.arange(self.num_envs, device=self.device))
        self.startup_report["standard_prone"] = dict(task=self.prone_task,
            knee_coordinate="physical passive knee angle; lower stop40deg, upper107deg",
            support_alignment="Horizontal COM alignment plus penalty when the wheel midpoint is not below the mass-weighted whole-robot COM",
            initial_output_bias="prone reset joint geometry; all six outputs trainable",
            scripted_target_or_motor_takeover=False, state_writes="episode reset only")

    def reset(self, ids):
        if not self.prone_ready:
            return super().reset(ids)
        cases = (self.evaluation_cases if self.evaluation_cases is not None else
                 [dict(pose_index=self.initialization_pose_index, recovery=True, command=[0., 0., .305])
                  for _ in range(self.num_envs)])
        if any(not cases[i]["recovery"] or cases[i]["pose_index"] != self.initialization_pose_index for i in ids.tolist()):
            raise ValueError("Standard-prone campaign must start from its declared prone pose")
        self.fixed_cases = cases
        try:
            super().reset(ids)
        finally:
            self.fixed_cases = None
        root = self.robot.data.root_link_pose_w.torch[ids].clone()
        clearance = self.poses[self.initialization_pose_index]["min_collision_clearance_m"]
        root[:, 2] -= clearance - self.prone_task["reset_collision_clearance_m"]
        self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=ids)
        update_robot_data(self, self.dt)
        self.prone_start_xy[ids] = root[:, :2]
        self.previous_potential[ids] = self.potential()[ids]
        # Only a vertical translation was made. IMU gravity, encoder histories
        # and actuator-domain RNG remain exactly those of the native reset.

    def route_state(self):
        knees = physical_knee_degrees(self.robot.data.joint_pos.torch[:, self.knee_ids],
                                     self.knee_bounds["L_joint2"][0])
        mass = self.body_mass
        com = (self.robot.data.body_com_pose_w.torch[:, :, :3] * mass[:, :, None]).sum(1) / mass.sum(-1, keepdim=True)
        wheel_midpoint = self.wheel_centers().mean(1)
        alignment = support_error(wheel_midpoint, com, self.prone_task["support_min_below_com_m"])
        excursion = (self.robot.data.root_link_pose_w.torch[:, :2] - self.prone_start_xy).norm(dim=-1)
        return dict(knee_deg=knees, support_error_m=alignment, com_world_m=com,
                    wheel_midpoint_world_m=wheel_midpoint, xy_excursion_m=excursion,
                    wheel_below_com_m=com[:, 2] - wheel_midpoint[:, 2])

    def potential(self):
        if not self.prone_ready:
            return super().potential()
        _, _, gravity, height, _, _, _ = self.state()
        route = self.route_state()
        return route_potential(gravity, route["knee_deg"], route["support_error_m"], height, self.prone_task)[0]

    def sample_commands(self, ids):
        if not self.prone_ready:
            return super().sample_commands(ids)

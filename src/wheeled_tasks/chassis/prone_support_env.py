"""A six-axis learned rise from normally-up belly-grounded leg configurations."""
from copy import deepcopy
import json

import torch

from . import prone_support as task
from .prone_motion_env import ProneMotionEnv
from .recovery_impact import RecoveryImpact
from .scut_observation import POLICY_FROM_CONTROL


class ProneSupportEnv(ProneMotionEnv):
    def __init__(self, *args, **kwargs):
        self.support_ready = self.support_hook = False
        self.support_native_probe = False
        from .collision_geometry import selected_asset, native_audit, CollisionGeometryError
        collision = kwargs["settings"]["prone_motion_task"]["chassis_collision"]
        with selected_asset(args[3], collision):
            super().__init__(*args, **kwargs)
        audit = native_audit(self, collision)
        self.startup_report["chassis_collision"] = audit
        if not audit["all_checks_pass"]:
            raise CollisionGeometryError(audit)
        if self.cfg.get("prone_motion_performance") != "eager":
            raise ValueError("Prone support capture requires completed native physics callbacks")
        self.whole_impact = RecoveryImpact(self.num_envs, self.device, self.robot.body_names,
            self.plan["recovery_impact"], self.dt, self.body_mass.sum(-1))
        self.support_contact = torch.zeros_like(self.contact_force)
        native_updated = self.control_audit.physics_updated
        def updated(dt):
            result = native_updated(dt)
            self._observe_support_physics(dt)
            return result
        self.control_audit.physics_updated = updated
        self.support_hook = True
        self.startup_report["prone_support"] = dict(config=self.support_config,
            training_pose_indices=self.support_train, holdout_pose_indices=self.support_holdout,
            native_reset_admitted=False, reset_admission="separate zero-PPO probe required",
            capture_clock="completed_native_steps", actor_inputs_added=False,
            goal_command="fixed_at_reset_from_external_height_curriculum; no per-episode contact switch",
            scripted_actions=False, legacy_standing_curriculum_disabled=True)
        self.startup_report["prone_motion"].update(
            reset="normally-up belly-grounded; independent left/right knees and wheel orbits",
            motor_owner="six trainable actor outputs throughout every enabled interval",
            route_bias="weak folding potential fades before wheel capture and extension",
            fallback_script=False)
        self.reset(torch.arange(self.num_envs, device=self.device))

    def _initialize_support(self):
        if self.support_ready:
            return
        from .full_robot_v202 import digest
        self.support_config = task.validate(self.plan["prone_support"])
        descriptor = self.support_config["reset_parameters"]
        path = self.asset_root / descriptor["file"]
        if digest(path) != descriptor["sha256"]:
            raise ValueError("Prone reset parameters changed")
        reset = json.loads(path.read_text())
        if (reset["schema"] != "wheel-leg-prone-support-resets-v1"
                or reset["collision_geometry_sha256"] != self.plan["chassis_collision"]["geometry"]["sha256"]):
            raise ValueError("Prone resets do not bind the corrected collision model")
        offset = len(self.poses)
        poses = deepcopy(reset["poses"])
        self.poses.extend(poses)
        self.bank_joints = torch.cat((self.bank_joints, self.bank_joints.new_tensor(
            [[p["full_joint_positions"][n] for n in self.robot.joint_names] for p in poses])))
        self.bank_roots = torch.cat((self.bank_roots, self.bank_roots.new_tensor(
            [p["root_position_m"] + p["root_quaternion_xyzw"] for p in poses])))
        self.bank_velocities = torch.cat((self.bank_velocities, torch.zeros_like(self.bank_joints[offset:])))
        self.bank_family = torch.cat((self.bank_family, self.bank_family.new_full((len(poses),),
            int(self.bank_family[self.height_pose[.305]]))))
        self.bank_tilt = torch.cat((self.bank_tilt, self.bank_tilt.new_zeros(len(poses))))
        self.support_train = [offset+i for i,p in enumerate(poses) if p["recovery"]["split"] == "train"]
        self.support_holdout = [offset+i for i,p in enumerate(poses) if p["recovery"]["split"] == "holdout"]
        for split in (self.support_train, self.support_holdout):
            if {self.poses[i]["recovery"]["leg_class"] for i in split} != set(task.LEG_CLASSES):
                raise ValueError("Every split must contain four independent leg classes")
        self.height_course = task.HeightCurriculum(self.plan["seed"], self.support_config)
        self.goal_height = torch.full((self.num_envs,), .23, device=self.device)
        self.support_enabled_at = torch.full_like(self.goal_height, float("nan"))
        self.native_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.support_gap = torch.zeros_like(self.native_streak)
        self.birth_phase = torch.zeros_like(self.native_streak)
        self.ended_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.native_capture_steps = self.support_total_completed = self.support_standing_completed = 0
        self.support_ready = True

    def course_reset_cases(self, cases, ids):
        self._initialize_support()
        cases = list(cases)
        for lane in ids:
            if self.evaluation_cases is None:
                cls = task.LEG_CLASSES[(lane+self.prone_reset_generation) % 4]
                pool = [i for i in self.support_train if self.poses[i]["recovery"]["leg_class"] == cls]
                index = self.height_course.rng.choice(pool)
                phase = self.height_course.phase
                goal = self.support_config["goal_heights_m"][phase]
                cases[lane] = dict(role="prone", recovery=True, pose_index=index,
                    pitch_deg=0., birth_height=self.poses[index]["root_position_m"][2],
                    target=[0.,0.,goal], command=[0.,0.,goal], support_birth_phase=phase)
            else:
                admitted = self.support_holdout + (self.support_train if self.support_native_probe else [])
                if cases[lane]["pose_index"] not in admitted:
                    raise ValueError("Cold prone support evaluation must use the held-out reset split")
                goal = cases[lane]["target"][2]
                if goal != .305:
                    raise ValueError("Independent completion always requires default standing height")
            self.goal_height[lane] = goal
            self.birth_phase[lane] = cases[lane].get("support_birth_phase", 2)
        return cases

    @torch.inference_mode()
    def reset(self, ids):
        if not getattr(self, "course_ready", False):
            return super().reset(ids)
        self._initialize_support()
        super().reset(ids)
        self.native_streak[ids] = self.support_gap[ids] = 0
        self.ended_recorded[ids] = False
        self.support_enabled_at[ids] = float("nan")
        seconds = self.support_config["episode_active_s"]+.2
        if self.evaluation_cases is not None:
            seconds = max(self.course_cases[i].get("evaluation_seconds", seconds) for i in ids.tolist())
        self.episode_limits[ids] = round(seconds/self.policy_dt)
        if self.support_hook:
            self.whole_impact.reset(ids)
            self.support_contact[ids] = 0.
        self.previous_potential[ids] = self.potential()[ids]

    def task_commands(self, elapsed):
        command = self.goal_height.new_zeros((self.num_envs, 3))
        command[:, 2] = self.goal_height
        return command

    def _geometry(self):
        route = self.route_state()
        root = self.robot.data.root_link_pose_w.torch
        x,y,z,w = root[:, 3:].unbind(-1)
        forward = torch.stack((1.-2.*(y*y+z*z), 2.*(x*y+w*z)), -1)
        forward /= forward.norm(dim=-1, keepdim=True).clamp_min(.1)
        wheels = self.wheel_centers()
        delta = wheels-route["com_world_m"][:, None]
        return route, (delta[..., :2]*forward[:, None]).sum(-1), -delta[..., 2]

    def _scores(self):
        _, omega, gravity, height, _, _, _ = self.state()
        route, error, below = self._geometry()
        velocity = self.robot.data.root_com_vel_w.torch[:, :3]
        result = task.scores(gravity, height, velocity, omega, route["knee_deg"], error, below,
                             self.goal_height, self.support_config)
        return result, route, error

    def potential(self):
        if not self.support_ready:
            return super().potential()
        return task.potential(self._scores()[0])

    @torch.inference_mode()
    def step(self, actions):
        self.whole_impact.begin_interval()
        return super().step(actions)

    def _observe_support_physics(self, dt):
        if dt != self.dt:
            raise RuntimeError("Capture clock differs from completed native physics")
        self.native_capture_steps += 1
        import warp as wp
        from .standing_support import wheel_bottom_clearance
        for ids, view in self.contact_views:
            force = wp.to_torch(view.get_contact_force_matrix(dt=dt))
            self.support_contact[ids] = force.reshape(len(ids), self.body_count, -1, 3).sum(2)
        self.whole_impact.observe(self.support_contact, ~self.finished)
        data = self.robot.data
        root = data.root_link_pose_w.torch
        x,y,z,w = root[:, 3:].unbind(-1)
        forward = torch.stack((1.-2.*(y*y+z*z), 2.*(x*y+w*z)), -1)
        forward /= forward.norm(dim=-1, keepdim=True).clamp_min(.1)
        com = (data.body_com_pose_w.torch[..., :3]*self.body_mass[..., None]).sum(1)/self.body_mass.sum(-1, keepdim=True)
        error = ((self.wheel_centers()[..., :2]-com[:, None, :2])*forward[:, None]).sum(-1)
        excursion = (root[:, :2]-self.prone_start_xy).norm(dim=-1)
        wheel_pose = data.body_link_pose_w.torch[:, self.wheel_ids]
        clearance = wheel_bottom_clearance(self.wheel_centers(), wheel_pose[..., 3:], self.origins[:, 2:3])
        weight = self.body_mass.sum(-1)*9.81
        force = self.support_contact[..., 2].clamp_min(0.)/weight[:, None]
        nonwheel = force[:, self.nonwheel_ids].sum(-1)
        posture, support = task.capture_conditions(data.projected_gravity_b.torch,
            data.root_link_pose_w.torch[:, 2]-self.origins[:, 2], data.root_com_vel_w.torch[:, :3],
            data.root_com_ang_vel_b.torch, data.joint_vel.torch[:, self.ids][:, POLICY_FROM_CONTROL],
            error, clearance, force[:, self.wheel_ids], nonwheel, excursion,
            self.goal_height, self.support_config)
        self.native_streak[:], self.support_gap[:] = task.advance_capture(self.native_streak,
            self.support_gap, posture, support, ~(self.passive | self.finished), dt,
            self.support_config["ignored_support_loss_s"])

    def task_stable(self, *unused):
        return self.native_streak >= round(self.support_config["capture_hold_s"]/self.dt)

    def task_transition(self, state, terminated):
        self.support_enabled_at[state["enable"]] = state["elapsed"][state["enable"]]
        age = torch.nan_to_num(state["elapsed"]-self.support_enabled_at, nan=0.)
        scores, route, error = self._scores()
        bad = dict(part_impact=self.recovery_impact.violation.clone(),
            excursion=route["xy_excursion_m"] > self.support_config["excursion_abort_m"],
            rise_timeout=(~self.passive & (age >= self.support_config["episode_active_s"]) & ~self.ever_recovered))
        terminated |= bad["part_impact"] | bad["excursion"]
        rejected = state["first"] & terminated
        state["first"] &= ~terminated
        self.ever_recovered[rejected] = False
        self.recovered_at[rejected] = float("nan")
        first = state["first"].clone()
        weight = self.body_mass.sum(-1)*9.81
        force = self.support_contact[..., 2].clamp_min(0.)/weight[:, None]
        terms = task.densities(scores, self.robot.data.root_com_vel_w.torch[:, :3], state["omega"],
            state["torque"], state["dq"][:, POLICY_FROM_CONTROL], state["action_delta"],
            route["xy_excursion_m"], force[:, self.wheel_ids], force[:, self.nonwheel_ids].sum(-1),
            self.whole_impact.impulse_peak, weight, self.ever_recovered)
        now = task.potential(scores)
        shaping = self.getup_cfg["reward_gamma"]*torch.where(terminated, 0., now)-self.previous_potential
        self.previous_potential.copy_(now)
        shaping += 7.*first  # Base environment pays the remaining3 once.
        if self.recovery_training_run and self.evaluation_cases is None:
            ending = terminated | bad["rise_timeout"] | first
            for lane in (ending & ~self.ended_recorded).nonzero().flatten().tolist():
                cls = self.poses[self.course_cases[lane]["pose_index"]]["recovery"]["leg_class"]
                self.height_course.observe(int(self.birth_phase[lane]), cls, bool(first[lane]))
            self.ended_recorded |= ending
            self.episode_limits.copy_(torch.where(bad["rise_timeout"] | first,
                                                   self.episode_length_buf, self.episode_limits))
        self.support_total_completed += int(first.sum())
        standing_first = first & (self.goal_height == .305)
        self.support_standing_completed += int(standing_first.sum())
        return sum(terms.values()), shaping, terminated, dict(task_reward_terms=terms, task_failures=bad,
            prone_route=route, support_completed_event=first, standing_completed_event=standing_first,
            support_native_hold_s=self.native_streak.float()*self.dt, support_goal_height_m=self.goal_height.clone(),
            support_active_seconds=age, support_wheel_x_error_m=error,
            support_whole_impulse_peak_ns=self.whole_impact.episode_impulse_peak.clone(),
            recovery_episode_force_peak_n=self.recovery_impact.episode_force_peak.clone(),
            recovery_episode_impulse_peak_ns=self.recovery_impact.episode_impulse_peak.clone(),
            recovery_part_impact_bad=bad["part_impact"])

    def task_metrics(self, state, extra, reward, done, terminated):
        metrics = {"ProneSupport/goal_completed_events": extra["support_completed_event"].float().sum(),
            "ProneSupport/standing_completed_events": extra["standing_completed_event"].float().sum(),
            "ProneSupport/height_phase": reward.new_tensor(self.height_course.phase),
            "ProneSupport/native_hold_peak_s": extra["support_native_hold_s"].max(),
            "ProneSupport/wheel_x_error_mean_m": extra["support_wheel_x_error_m"].abs().mean(),
            "ProneSupport/episode_events": done.float().sum()}
        for name, value in extra["task_reward_terms"].items():
            metrics["ProneSupportReward/"+name+"_per_second"] = value.mean()
        for index, group in enumerate(("chassis", "gimbal", "legs", "wheels")):
            metrics["ProneSupport/whole_impulse/"+group+"_Ns"] = extra["support_whole_impulse_peak_ns"][:, index].max()
        metrics.update(self.encoder_increment_actions.metrics())
        return metrics

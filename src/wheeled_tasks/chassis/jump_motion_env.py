"""A mixed v4 course with reset-only aerial births and fully learned takeoff."""
from __future__ import annotations

import torch

from .prone_motion_env import ProneMotionEnv
from .jump_motion import JumpMotion, BASE, AIRBORNE, JUMP, PHASE_NAMES, sample_range
from .root_kinematics import update_robot_data, root_link_linear_velocity
from . import prone_motion


class JumpMotionEnv(ProneMotionEnv):
    def __init__(self, *args, **kwargs):
        self.jump_training = None
        self.jump_sampling_enabled = False
        super().__init__(*args, **kwargs)
        self.jump_training = JumpMotion(self.num_envs, self.device, self.dt,
            self.plan["jump_training"], self.body_mass.sum(-1))
        self.jump_sampling_enabled = True
        grounded_reset = self.reset

        def reset(ids):
            result = grounded_reset(ids)
            self._finish_jump_reset(ids)
            return result

        self.reset = reset
        self.reset(torch.arange(self.num_envs, device=self.device))
        self.startup_report["jump_training"] = dict(self.plan["jump_training"],
            state_writes="airborne_episode_reset_only", motor_owner="six_axis_rl",
            external_forces=False, policy_dim=35, odometry_validity_slot=31,
            context_slots=[28, 29, 30, 32, 33, 34],
            context_source="command_mode_reference_velocity_requested_apex_command_clock",
            privileged_actor_inputs=False, physics_event_dt_s=self.dt,
            takeoff_and_landing_behavior_qualified=False)

    def course_reset_cases(self, cases, ids):
        if not self.jump_sampling_enabled or self.evaluation_cases is not None:
            return cases
        cfg = self.plan["jump_training"]
        for i in ids:
            role = cfg["roles_per10"][i % 10]
            if role == "base":
                continue
            h = .23 if role == "airborne" and (i//10) % 2 else .305
            cases[i] = dict(role="stand", recovery=False, pose_index=self.height_pose[h],
                birth_height=h, target=[0., 0., cfg["standing_height_m"]], command=[0., 0., h],
                pitch_deg=0., jump_mode=AIRBORNE if role == "airborne" else JUMP)
        return cases

    @torch.inference_mode()
    def _finish_jump_reset(self, ids):
        from .standing_support import wheel_bottom_clearance
        from .grounded_standing_reset import refresh_reset_sensors
        cfg = self.jump_training.cfg
        modes = torch.tensor([self.course_cases[i].get("jump_mode", BASE) for i in ids.tolist()], device=self.device)
        difficulty = min(1., self.actor_updates/cfg["difficulty_full_update"])
        delay = cfg["request_delay_s"][0]+self.random(len(ids))*(cfg["request_delay_s"][1]-cfg["request_delay_s"][0])
        apex = cfg["release_height_m"]+sample_range(cfg, "apex_above_release", difficulty, self.random(len(ids)))
        self.jump_training.reset(ids, modes, delay, apex)
        aerial = ids[modes == AIRBORNE]
        if len(aerial):
            root = self.robot.data.root_link_pose_w.torch[aerial].clone()
            # Small local roll/pitch disturbances preserve valid closed-chain
            # joint configurations from the same authenticated reset bank.
            roll = (self.random(len(aerial))*2.-1.)*(.04+.06*difficulty)
            pitch = (self.random(len(aerial))*2.-1.)*(.06+.10*difficulty)
            sr, cr = (roll/2.).sin(), (roll/2.).cos()
            sp, cp = (pitch/2.).sin(), (pitch/2.).cos()
            local = torch.stack((sr*cp, cr*sp, -sr*sp, cr*cp), -1)
            xyz, w = root[:, 3:6].clone(), root[:, 6:7].clone()
            root[:, 3:6] = w*local[:, :3]+local[:, 3:]*xyz+torch.linalg.cross(xyz, local[:, :3])
            root[:, 6] = (w*local[:, 3:]-(xyz*local[:, :3]).sum(-1, keepdim=True))[:, 0]
            self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=aerial)
            update_robot_data(self, self.dt)
            pose = self.robot.data.body_link_pose_w.torch[aerial][:, self.wheel_ids]
            clearance = wheel_bottom_clearance(self.wheel_centers()[aerial], pose[:, :, 3:], self.origins[aerial, 2:3])
            desired = sample_range(cfg, "air_clearance", difficulty, self.random(len(aerial)))
            root[:, 2] += desired-clearance.amin(-1)
            self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=aerial)
            velocity = root.new_zeros(len(aerial), 6)
            velocity[:, :2] = (self.random(len(aerial)*2).reshape(-1, 2)*2.-1.)*(.1+.3*difficulty)
            velocity[:, 2] = -self.random(len(aerial))*(.1+.2*difficulty)
            velocity[:, 3:] = (self.random(len(aerial)*3).reshape(-1, 3)*2.-1.)*(.1+.2*difficulty)
            self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity, env_ids=aerial)
            update_robot_data(self, self.dt)
            self.contact_force[aerial] = 0.
            refresh_reset_sensors(self, aerial)
        active = ids[modes != BASE]
        self.episode_limits[active] = round(cfg["episode_seconds"]/self.policy_dt)
        self.local_standing_odometry.reset(active)
        self.pending_command[ids] = self.task_commands(self.episode_length_buf*self.policy_dt)[ids]
        self.goal_command[ids] = self.pending_command[ids]
        self.commands[ids] = self.pending_command[ids]
        self.raw_height[ids] = self.commands[ids, 2]

    def get_observations(self):
        result = super().get_observations()
        if self.jump_training is not None and self.jump_training.has_active:
            selected = self.jump_training.active
            # Preserve every base-lane input, including the independent slot31
            # validity bit. Flight must not turn wheel spin into standing XY.
            policy = result["policy"].clone()
            critic = result["critic"].clone()
            context = self.jump_training.context()
            slots = [28, 29, 30, 32, 33, 34]
            for col, slot in enumerate(slots):
                policy[:, slot] = torch.where(selected, context[:, col], policy[:, slot])
                critic[:, slot] = torch.where(selected, context[:, col], critic[:, slot])
            for slot in (14, 15, 31):
                policy[:, slot] = torch.where(selected, 0., policy[:, slot])
                critic[:, slot] = torch.where(selected, 0., critic[:, slot])
            result["policy"], result["critic"] = policy, critic
        return result

    def task_commands(self, elapsed):
        value = super().task_commands(elapsed)
        if self.jump_training is not None and self.jump_training.has_active:
            selected = self.jump_training.active
            value[selected, :2] = 0.
            value[:, 2] = torch.where(selected, self.jump_training.reference()[0], value[:, 2])
        return value

    def task_reward_terms(self, state, route, error, rising):
        terms = super().task_reward_terms(state, route, error, rising)
        if self.jump_training is None or not self.jump_training.has_active:
            return terms
        active = self.jump_training.active
        own = self.jump_training.rewards(state, route["encoder_wheel_body_m"],
            root_link_linear_velocity(self, frame="world")[:, 2],
            prone_motion.normalized_action_delta(state["action_delta"], self.plan))
        terms = {k: torch.where(active, 0., v) for k, v in terms.items()}
        terms.update({k: torch.where(active, v, 0.) for k, v in own.items()})
        return terms

    def task_failures(self, state, route, rising):
        bad = super().task_failures(state, route, rising)
        if self.jump_training is not None and self.jump_training.has_active:
            active = self.jump_training.active
            bad = {k: v & ~active for k, v in bad.items()}
            bad["jump_failure"] = self.jump_training.failures(state["height"], state["gravity"], state["nonwheel_force"])
        return bad

    def task_transition(self, state, terminated):
        density, bonus, terminated, extra = super().task_transition(state, terminated)
        if self.jump_training is not None and self.jump_training.has_active:
            jump = self.jump_training
            jump.reject_failed_successes(terminated)
            bonus = bonus+jump.event_bonus()*(~terminated)
            extra["jump_training"] = dict(mode=jump.mode.clone(), phase=jump.phase.clone(),
                liftoff_event=jump.liftoff_event.clone(), landing_event=jump.landing_event.clone(),
                success_event=(jump.success_event & ~terminated).clone(), request_event=jump.request_event.clone(),
                target_apex_m=jump.target_apex.clone(), peak_height_m=jump.peak_height.clone(),
                release_vz_m_s=jump.release_vz.clone(), flight_time_s=jump.flight_time.clone(),
                stable_hold_s=jump.stable_hold.clone(), wheel_force_peak_n=jump.wheel_force_peak.clone())
        return density, bonus, terminated, extra

    def task_metrics(self, state, extra, reward, done, terminated):
        result = super().task_metrics(state, extra, reward, done, terminated)
        if "jump_training" not in extra:
            return result
        jump = extra["jump_training"]
        for mode, name in ((AIRBORNE, "Airborne"), (JUMP, "Jump")):
            mask = jump["mode"] == mode
            denominator = mask.sum().clamp_min(1)
            result[f"{name}/sampled_lanes_per_step"] = mask.float().sum()
            result[f"{name}/failures_per_step"] = (terminated & mask).float().sum()
            for event in ("request", "liftoff", "landing", "success"):
                result[f"{name}/{event}_events_per_step"] = (jump[event+"_event"] & mask).float().sum()
            for field in ("target_apex_m", "peak_height_m", "release_vz_m_s", "flight_time_s", "stable_hold_s", "wheel_force_peak_n"):
                result[f"{name}/{field}"] = torch.where(mask, jump[field], 0.).sum()/denominator
            for phase, label in enumerate(PHASE_NAMES):
                result[f"{name}/phase_{label}_fraction"] = ((jump["phase"] == phase) & mask).float().sum()/denominator
        for key, value in extra["task_reward_terms"].items():
            if key.startswith("jump_"):
                result["JumpReward/"+key+"_per_second"] = value.sum()/self.jump_training.active.sum().clamp_min(1)
        return result

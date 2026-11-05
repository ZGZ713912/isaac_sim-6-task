"""Opt-in jump curriculum on the unchanged WheelBipe v2.0.2 plant."""
from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .environment import WheelbipePortEnv
from .mixed_curriculum import JumpObserver, REWARD_NAMES, TASKS, TaskSchedule
from .training_profile import apply_phase, phase_for_update, specification


class WheelbipeMixedEnv(WheelbipePortEnv):
    def __init__(self, *args, **kwargs):
        self.task_schedule = self.jump_observer = None
        self._mixed_observed_tick = -1
        self._mixed_interval_age = None
        super().__init__(*args, **kwargs)
        spec = self.cfg.port_mixed_spec
        self.task_schedule = TaskSchedule(self.num_envs, self.device, self.cfg.seed, spec)
        mass = self.robot.root_physx_view.get_masses().to(self.device).sum(-1)
        self.jump_observer = JumpObserver(self.task_schedule, self.physics_dt, mass)
        self._mixed_sensor_wheel_ids = self._find_contact_sensor_indices(["left_wheel_link", "right_wheel_link"])
        if len(self._mixed_sensor_wheel_ids) != 2:
            raise ValueError("Jump observer requires two named wheel contact bodies")
        self._mixed_sensor_other_ids = [i for i in range(len(self.contact_sensor.body_names))
                                       if i not in self._mixed_sensor_wheel_ids]
        wheel_names = ("L_link3", "R_link3")
        bodies = {body["name"]: body for body in self.asset.spec["bodies"]}
        collisions = [bodies[name]["collisions"][0] for name in wheel_names]
        if any(c["type"] != "cylinder" for c in collisions):
            raise ValueError("Wheel clearance must use the authenticated cylinder geometry")
        self._mixed_wheel_ids = [self.robot.cad_body_names.index(name) for name in wheel_names]
        self._mixed_wheel_offset = torch.tensor([[row[3] for row in c["origin"][:3]] for c in collisions], device=self.device)
        self._mixed_wheel_axis = torch.tensor([[row[2] for row in c["origin"][:3]] for c in collisions], device=self.device)
        self._mixed_wheel_radius = torch.tensor([c["radius"] for c in collisions], device=self.device)
        self._mixed_wheel_half_length = torch.tensor([c["length"]*.5 for c in collisions], device=self.device)

    def _mixed_enabled(self):
        return self.task_schedule is not None and not hasattr(self, "eval_command")

    def learning_curriculum_state(self):
        return dict(profile_sha256=self.cfg.port_profile_sha256,
            role=self.task_schedule.role.clone(), generator=self.task_schedule.generator.get_state())

    def restore_learning_curriculum_state(self, state):
        if (state["profile_sha256"] != self.cfg.port_profile_sha256
                or not torch.equal(state["role"].to(self.device), self.task_schedule.role)):
            raise ValueError("Mixed curriculum contract or lane allocation differs")
        self.task_schedule.generator.set_state(state["generator"].cpu())

    def _mixed_age(self, physics=False):
        age = self.episode_length_buf.float()*self.step_dt
        if physics:
            age = age+(self.port_physics_calls % self.cfg.decimation)*self.physics_dt
        return age

    def _update_training_profile(self):
        if getattr(self.cfg, "port_training_profile", "") != "foundation-jump-v1":
            return super()._update_training_profile()
        if not getattr(self, "_port_ready", False) or hasattr(self, "eval_command"):
            return
        update = max(getattr(self.cfg, "port_start_completed_updates", 0), self._get_training_iteration())
        self._mixed_update = update
        phase = phase_for_update(update, specification("foundation-jump-v1"))
        if phase["start_update"] != self.port_training_phase:
            apply_phase(self.cfg, update)
            self.port_training_phase = phase["start_update"]

    def _mixed_commands(self):
        if not self._mixed_enabled():
            return
        phase = dict(phase_for_update(getattr(self, "_mixed_update", 0), self.cfg.port_mixed_spec))
        phase["completed_update"] = getattr(self, "_mixed_update", 0)
        command, height = self.task_schedule.update_commands(self._mixed_age(), phase)
        self.command.copy_(command)
        self.height_cmd.copy_(height)
        # Disable inherited random standing/heading/special-mode postprocessing
        # only in this opt-in adapter; the foundation implementation is intact.
        self.command_generator.vel_command_b.copy_(command)
        self.command_generator.is_standing_env.zero_()
        self.command_generator.is_heading_env.zero_()
        self.command_generator.special_mode_id.fill_(-1)

    def _on_command_updated(self):
        super()._on_command_updated()
        self._mixed_commands()

    def _get_ctrl_mode_obs_raw(self):
        raw = super()._get_ctrl_mode_obs_raw()
        if self._mixed_enabled():
            raw = self.task_schedule.context(raw, self._mixed_age())
        return raw

    def _custom_reset_random(self, env_ids):
        super()._custom_reset_random(env_ids)
        if not self._mixed_enabled():
            return
        schedule = self.task_schedule
        schedule.reset(env_ids, getattr(self, "_mixed_update", 0))
        self.jump_observer.reset(env_ids)
        selected = (schedule.mask("jump") | schedule.mask("airborne"))[env_ids]
        ids = env_ids[selected]
        if len(ids):
            # Only reset writes state. Active jumps receive no force, torque,
            # pose, joint-target, or velocity assistance during dynamics.
            q = self.robot.data.default_joint_pos[ids].clone()
            root = self.robot.data.default_root_state[ids].clone()
            root[:, :3] += self.terrain.env_origins[ids]
            minimum = self.collision_support.minimum_z(q, root[:, 3:7])
            root[:, 2] = self.terrain.env_origins[ids, 2]-minimum+.002
            root[:, 2] += schedule.drop[ids]*schedule.mask("airborne")[ids]
            root[:, 7:] = 0.
            self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), None, ids)
            self.robot.write_root_state_to_sim(root, ids)
            self.predefined_reset_ground_zero_torque_until_time[ids] = 0.

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._mixed_commands()

    def _pre_physics_step(self, actions):
        if self._mixed_enabled():
            self.jump_observer.begin_interval()
            self._mixed_interval_age = self._mixed_age().clone()
        super()._pre_physics_step(actions)

    def _observe_jump(self):
        if not self._mixed_enabled() or self._mixed_observed_tick == self.port_physics_calls:
            return
        self._mixed_observed_tick = self.port_physics_calls
        forces = self.contact_sensor.data.net_forces_w.norm(dim=-1)
        wheel_force = forces[:, self._mixed_sensor_wheel_ids]
        other_force = forces[:, self._mixed_sensor_other_ids].amax(-1)
        q = self.robot.data.body_quat_w[:, self._mixed_wheel_ids]
        center = self.robot.data.body_pos_w[:, self._mixed_wheel_ids]+quat_apply(q,
            self._mixed_wheel_offset.expand(self.num_envs, -1, -1))
        axis = quat_apply(q, self._mixed_wheel_axis.expand(self.num_envs, -1, -1))
        extent_z = (self._mixed_wheel_radius*(1-axis[:, :, 2].square()).clamp_min(0.).sqrt()
                    +self._mixed_wheel_half_length*axis[:, :, 2].abs())
        clearance = center[:, :, 2]-extent_z-self.terrain.env_origins[:, 2, None]
        height = self.robot.data.root_pos_w[:, 2]-self.terrain.env_origins[:, 2]
        self.jump_observer.observe(self._mixed_age(physics=True), height,
            self.robot.data.root_lin_vel_w, self.robot.data.projected_gravity_b,
            clearance, wheel_force, other_force)

    def _apply_action(self):
        self._observe_jump()
        super()._apply_action()

    def _get_dones(self):
        self._observe_jump()
        terminate, timeout = super()._get_dones()
        if self._mixed_enabled():
            schedule = self.task_schedule
            jump = schedule.mask("jump") | schedule.mask("airborne")
            timeout |= jump & (self._mixed_age() >= self.cfg.port_mixed_spec["jump"]["episode_seconds"])
            bad_success = self.jump_observer.success_event & terminate
            self.jump_observer.success_event[bad_success] = False
            self.jump_observer.succeeded[bad_success] = False
        return terminate, timeout

    def _postprocess_reward_terms(self, terms):
        terms = super()._postprocess_reward_terms(terms)
        if not self._mixed_enabled() or self._mixed_interval_age is None:
            return terms
        age = self._mixed_interval_age
        ref = self.task_schedule.reference(age)
        for name in self.cfg.port_mixed_spec["reward_policy"]["suppressed_during_jump_request"]:
            if name in terms:
                terms[name] = terms[name]*(~ref["active"])
        relative = self.robot.data.body_pos_w[:, self._wheel_link_idx]-self.robot.data.root_pos_w[:, None, :]
        inverse = self.robot.data.root_quat_w.clone()
        wheel_body = quat_apply_inverse(inverse[:, None, :].expand(-1, 2, -1), relative)
        height = self.robot.data.root_pos_w[:, 2]-self.terrain.env_origins[:, 2]
        terms.update(self.jump_observer.reward_terms(age, height,
            self.robot.data.root_lin_vel_w[:, 2], wheel_body, self.step_dt))
        return terms

    def _get_rewards(self):
        rewards = super()._get_rewards()
        if self._mixed_enabled():
            logs = self.extras.setdefault("log", {})
            for name in TASKS:
                mask = self.task_schedule.mask(name)
                valid = mask & ~self._numerical_safety_reset_buf
                logs["Mixed/"+name+"/allocated_envs"] = mask.sum().float()
                logs["Mixed/"+name+"/finite_physics_samples_per_step"] = valid.sum().float()
                logs["Mixed/"+name+"/paid_reward_mean_per_step"] = rewards[mask].mean()
                logs["Mixed/"+name+"/paid_reward_rms_per_step"] = rewards[mask].square().mean().sqrt()
            for name in REWARD_NAMES:
                logs["Mixed/reward/"+name+"_mean_per_step"] = self._last_reward_terms[name].mean()
            for name in ("liftoff_event", "landing_event", "success_event", "timeout_event"):
                logs["Mixed/jump/"+name+"_count_per_step"] = getattr(self.jump_observer, name).sum().float()
            logs["Mixed/jump/request_active_envs"] = self.task_schedule.reference(self._mixed_age())["active"].sum().float()
            logs["Mixed/jump/force_peak_bodyweights"] = (self.jump_observer.force_peak/self.jump_observer.weight).max()
        return rewards

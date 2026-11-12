"""Opt-in jump curriculum on the unchanged WheelBipe v2.0.2 plant."""
from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .environment import WheelbipePortEnv
from .mixed_curriculum import (DRIVE_OFF, FLAT, OVER, JumpObserver, PlatformObserver, TaskSchedule,
    reward_names, tasks_for)
from .training_profile import apply_phase, phase_for_update


class WheelbipeMixedEnv(WheelbipePortEnv):
    def __init__(self, *args, **kwargs):
        self.task_schedule = self.jump_observer = self.platform_observer = None
        self._early_schedule = None
        self._mixed_observed_tick = -1
        self._mixed_interval_age = None
        super().__init__(*args, **kwargs)
        spec = self.cfg.port_mixed_spec
        # Platform lanes are allocated while the scene is built; reuse that
        # exact schedule (same seeded generator) instead of drawing again.
        self.task_schedule = self._early_schedule or TaskSchedule(self.num_envs, self.device, self.cfg.seed, spec)
        body_masses = self.robot.root_physx_view.get_masses().to(self.device)
        mass = body_masses.sum(-1)
        self._mixed_mass_fractions = body_masses/mass[:, None]
        self.jump_observer = JumpObserver(self.task_schedule, self.physics_dt, mass)
        if self.task_schedule.has_platform:
            self.platform_observer = PlatformObserver(self.task_schedule, self.physics_dt, mass)
            # Signed wheel odometry. The mirrored wheel joints roll forward with
            # the sign of their CAD joint axis z (L_joint3 +z, R_joint3 -z);
            # natively checked: forward 1 m/s gives +15 / -15 rad/s.
            axes = {joint["name"]: joint["axis"] for joint in self.asset.spec["joints"]}
            self._wheel_forward_sign = torch.tensor([1. if axes[name][2] > 0 else -1.
                for name in ("L_joint3", "R_joint3")], device=self.device)
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

    def _spawn_task_terrain(self):
        """One static box per platform lane, under that lane's own env prim."""
        spec = getattr(self.cfg, "port_mixed_spec", None)
        if spec is None or "platform" not in tasks_for(spec):
            return
        import isaaclab.sim as sim_utils
        from pxr import Usd, UsdGeom
        schedule = TaskSchedule(self.scene.num_envs, self.device, self.cfg.seed, spec)
        self._early_schedule = schedule
        p = spec["platform"]
        ground = self.cfg.terrain.physics_material
        material = sim_utils.RigidBodyMaterialCfg(static_friction=ground.static_friction,
            dynamic_friction=ground.dynamic_friction, restitution=0.,
            friction_combine_mode=ground.friction_combine_mode)
        origins = self.terrain.env_origins.detach().cpu()
        lanes = schedule.mask("platform").nonzero().flatten().tolist()
        for i in lanes:
            env_path = self.scene.env_prim_paths[i]
            # Robots reset at the terrain origin; express the box relative to
            # the env prim so both agree even if the two grids differ.
            world = UsdGeom.Xformable(self.sim.stage.GetPrimAtPath(env_path)).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()).ExtractTranslation()
            rise = float(schedule.rise[i])
            local = (float(origins[i, 0])-world[0]+p["near_edge_x_m"]+.5*p["length_m"],
                     float(origins[i, 1])-world[1], float(origins[i, 2])-world[2]+.5*rise)
            cfg = sim_utils.CuboidCfg(size=(p["length_m"], p["width_m"], rise),
                collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=material)
            cfg.func(env_path+"/Platform", cfg, translation=local)
        self.platform_lane_ids = lanes

    def _wheel_geometry(self):
        """Env-local wheel-centre XY and wheel-bottom height above the floor, (N, 2)."""
        q = self.robot.data.body_quat_w[:, self._mixed_wheel_ids]
        center = self.robot.data.body_pos_w[:, self._mixed_wheel_ids]+quat_apply(q,
            self._mixed_wheel_offset.expand(self.num_envs, -1, -1))
        axis = quat_apply(q, self._mixed_wheel_axis.expand(self.num_envs, -1, -1))
        extent_z = (self._mixed_wheel_radius*(1-axis[:, :, 2].square()).clamp_min(0.).sqrt()
                    +self._mixed_wheel_half_length*axis[:, :, 2].abs())
        origin = self.terrain.env_origins
        return center[..., :2]-origin[:, None, :2], center[:, :, 2]-extent_z-origin[:, 2, None]

    def _get_height_reward_reference_height(self, relative_obs_height, wheel_height_w):
        value = super()._get_height_reward_reference_height(relative_obs_height, wheel_height_w)
        if self.platform_observer is not None and self._mixed_enabled():
            # Reward-only: height above the surface under the wheels.
            xy, _ = self._wheel_geometry()
            value = value-self.task_schedule.support_height(xy).mean(-1)
        return value

    def _tuck_compensation_eligible(self):
        if not self._mixed_enabled():
            return super()._tuck_compensation_eligible()
        # The control clock at this point follows the pre-action observer.
        # It depends only on the request and time, never actual flight/contact.
        age = self.episode_length_buf.float()*self.step_dt
        age = age+((self.port_physics_calls-1) % self.cfg.decimation)*self.physics_dt
        return self.task_schedule.reference(age)["flight"]

    def _leg_target_overtravel_mask(self, *, physics=False):
        if self.cfg.port_leg_target_overtravel_scope == "all" or not self._mixed_enabled():
            return super()._leg_target_overtravel_mask(physics=physics)
        age = self.episode_length_buf.float()*self.step_dt
        if physics:
            age = age+((self.port_physics_calls-1) % self.cfg.decimation)*self.physics_dt
        ref = self.task_schedule.reference(age)
        return ref["requested"] & ref["active"]

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
        if not getattr(self.cfg, "port_training_profile", "").startswith("foundation-jump-"):
            return super()._update_training_profile()
        if not getattr(self, "_port_ready", False) or hasattr(self, "eval_command"):
            return
        update = max(getattr(self.cfg, "port_start_completed_updates", 0), self._get_training_iteration())
        self._mixed_update = update
        motion_update = update+getattr(self.cfg, "port_motion_update_offset", 0)
        phase = phase_for_update(motion_update, self.cfg.port_mixed_spec)
        if phase["start_update"] != self.port_training_phase:
            apply_phase(self.cfg, motion_update)
            self.port_training_phase = phase["start_update"]

    def _mixed_commands(self):
        if not self._mixed_enabled():
            return
        motion_update = getattr(self, "_mixed_update", 0)+getattr(self.cfg, "port_motion_update_offset", 0)
        phase = dict(phase_for_update(motion_update, self.cfg.port_mixed_spec))
        phase["completed_update"] = motion_update
        command, height = self.task_schedule.update_commands(self._mixed_age(), phase)
        if self.cfg.port_mixed_spec["jump"].get("reward_version", 1) >= 2:
            ref = self.task_schedule.reference(self._mixed_age())
            height = torch.where(ref["requested"], ref["height"], height)
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
            # Only reset writes physical state. During dynamics the actor PD
            # and explicit v4 motor feedforward are the sole active leg effort;
            # no root velocity, pose or contact-dependent control is injected.
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
        if self.platform_observer is not None:
            self.platform_observer.reset(env_ids)
            ids = env_ids[schedule.mask("platform")[env_ids]]
            if len(ids):
                # Heading +x toward the box. Over: in front at the sampled run-up;
                # drive-off: on top; locked lanes: far away on flat ground.
                q = self.robot.data.default_joint_pos[ids].clone()
                root = self.robot.data.default_root_state[ids].clone()
                root[:, :3] += self.terrain.env_origins[ids]
                root[:, 0] += schedule.start_x[ids]
                minimum = self.collision_support.minimum_z(q, root[:, 3:7])
                lift = torch.where(schedule.mode[ids] == DRIVE_OFF, schedule.rise[ids], 0.)
                root[:, 2] = self.terrain.env_origins[ids, 2]-minimum+.002+lift
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
            if self.platform_observer is not None:
                self.platform_observer.begin_interval()
                # Deployable signed wheel odometry fires the explicit request.
                wheels = [self.motor_ids[i] for i in (2, 5)]
                speed = (self.robot.data.joint_vel[:, wheels]*self._wheel_forward_sign).mean(-1)
                self.task_schedule.advance_platform_requests(self._mixed_age(),
                    speed*self.cfg.port_mixed_spec["jump"]["wheel_radius_m"]*self.step_dt)
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
        # Keep the tuck observer on the same body-relative wheel-to-chassis
        # length proxy used by the air reward.  This is diagnostic/reward state
        # only; it never enters the actor observation or advances the request
        # clock.
        relative = self.robot.data.body_pos_w[:, self._wheel_link_idx]-self.robot.data.root_pos_w[:, None, :]
        root_quat = self.robot.data.root_quat_w
        wheel_body = quat_apply_inverse(root_quat[:, None, :].expand(-1, 2, -1), relative)
        leg_lengths = wheel_body[:, :, (0, 2)].norm(dim=-1)
        kwargs = {}
        speed_error = (self.robot.data.root_lin_vel_b[:, :2]-self.command[:, :2]).norm(dim=-1)
        if self.cfg.port_mixed_spec["jump"].get("reward_version", 1) >= 2:
            kwargs["horizontal_speed_error"] = speed_error
        if self.platform_observer is not None:
            j = self.cfg.port_mixed_spec["jump"]
            xy = center[..., :2]-self.terrain.env_origins[:, None, :2]
            support = self.task_schedule.support_height(xy)
            bottom = clearance
            clearance = clearance-support
            height = height-support.mean(-1)
            kwargs["on_top"] = ((support > 0.).all(-1) & (clearance < j["landing_clearance_m"]).all(-1)
                                & (wheel_force > j["force_threshold_n"]).all(-1))
        if self.jump_observer.sequence is not None:
            from .jump_objectives import JumpPhysics
            data = self.robot.data
            weights = self._mixed_mass_fractions
            com_height = (data.body_com_pos_w[..., 2]*weights).sum(-1)-self.terrain.env_origins[:, 2]
            com_vz = (data.body_com_lin_vel_w[..., 2]*weights).sum(-1)
            lower, upper = self.collision_support.bounds["base_link"]
            vertical = -data.projected_gravity_b
            chassis_bottom = ((vertical*(upper+lower)/2).sum(-1)
                -(vertical.abs()*(upper-lower)/2).sum(-1)
                +data.root_pos_w[:, 2]-self.terrain.env_origins[:, 2])
            # This conservative chassis AABB diagnostic is not a collision or
            # success oracle. Wheel clearance uses the actual cylinders above.
            legs = (0, 1, 3, 4)
            kwargs["physics"] = JumpPhysics(com_height, com_vz,
                self.contact_sensor.data.net_forces_w[:, self._mixed_sensor_wheel_ids, 2],
                self.motor_effort[:, legs], data.joint_vel[:, [self.motor_ids[i] for i in legs]],
                chassis_bottom)
        self.jump_observer.observe(self._mixed_age(physics=True), height,
            self.robot.data.root_lin_vel_w, self.robot.data.projected_gravity_b,
            clearance, wheel_force, other_force, leg_lengths=leg_lengths, **kwargs)
        if self.platform_observer is not None:
            force_x = self.contact_sensor.data.net_forces_w[:, self._mixed_sensor_wheel_ids, 0]
            self.platform_observer.observe(self._mixed_age(physics=True), xy, bottom, support,
                wheel_force, force_x, height, self.robot.data.root_lin_vel_w,
                self.robot.data.projected_gravity_b, speed_error, other_force)

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
            if schedule.has_platform:
                timeout |= schedule.mask("platform") & (
                    self._mixed_age() >= self.cfg.port_mixed_spec["platform"]["episode_seconds"])
            bad_success = self.jump_observer.success_event & terminate
            self.jump_observer.success_event[bad_success] = False
            self.jump_observer.succeeded[bad_success] = False
            if self.jump_observer.sequence is not None:
                self.jump_observer.sequence.record_outcomes(self.jump_observer, terminate | timeout, terminate)
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
        for name, scale in self.cfg.port_mixed_spec["reward_policy"].get("jump_request_cost_scales", {}).items():
            if name in terms:
                terms[name] = terms[name]*torch.where(ref["active"], scale, 1.)
        sequence = self.jump_observer.sequence
        if sequence is not None and "joint_torque" in terms:
            terms["joint_torque"] = terms["joint_torque"]*torch.where(sequence.push,
                sequence.spec["push_torque_cost_fraction"], 1.)
        relative = self.robot.data.body_pos_w[:, self._wheel_link_idx]-self.robot.data.root_pos_w[:, None, :]
        inverse = self.robot.data.root_quat_w.clone()
        wheel_body = quat_apply_inverse(inverse[:, None, :].expand(-1, 2, -1), relative)
        height = self.robot.data.root_pos_w[:, 2]-self.terrain.env_origins[:, 2]
        if self.platform_observer is not None:
            xy, bottom = self._wheel_geometry()
            height = height-self.task_schedule.support_height(xy).mean(-1)
        terms.update(self.jump_observer.reward_terms(age, height,
            self.robot.data.root_lin_vel_w[:, 2], wheel_body, self.step_dt))
        if self.platform_observer is not None:
            terms.update(self.platform_observer.reward_terms(age, xy, bottom, self.step_dt))
        return terms

    def _get_rewards(self):
        rewards = super()._get_rewards()
        if self._mixed_enabled():
            logs = self.extras.setdefault("log", {})
            for name in self.task_schedule.tasks:
                mask = self.task_schedule.mask(name)
                valid = mask & ~self._numerical_safety_reset_buf
                logs["Mixed/"+name+"/allocated_envs"] = mask.sum().float()
                logs["Mixed/"+name+"/finite_physics_samples_per_step"] = valid.sum().float()
                logs["Mixed/"+name+"/paid_reward_mean_per_step"] = rewards[mask].mean()
                logs["Mixed/"+name+"/paid_reward_rms_per_step"] = rewards[mask].square().mean().sqrt()
            for name in reward_names(self.cfg.port_mixed_spec):
                logs["Mixed/reward/"+name+"_mean_per_step"] = self._last_reward_terms[name].mean()
            for name in ("air_alignment", "air_tuck", "air_extension"):
                logs["Mixed/reward/jump_"+name+"_mean_per_step"] = getattr(
                    self.jump_observer, "last_"+name).mean()
            for name in ("landing_buffer_height", "landing_buffer_velocity"):
                logs["Mixed/reward/jump_"+name+"_mean_per_step"] = getattr(
                    self.jump_observer, "last_"+name).mean()
            for name in ("liftoff_event", "landing_event", "success_event", "timeout_event"):
                logs["Mixed/jump/"+name+"_count_per_step"] = getattr(self.jump_observer, name).sum().float()
                for task in ("jump", "airborne"):
                    logs["Mixed/physical/"+task+"/"+name+"_count_per_step"] = (
                        getattr(self.jump_observer, name) & self.task_schedule.mask(task)).sum().float()
            logs["Mixed/jump/request_active_envs"] = self.task_schedule.reference(self._mixed_age())["active"].sum().float()
            logs["Mixed/jump/force_peak_bodyweights"] = (self.jump_observer.force_peak/self.jump_observer.weight).max()
            logs["Mixed/jump/tuck_seen_count_per_step"] = self.jump_observer.tuck_seen.sum().float()
            logs["Mixed/jump/max_bilateral_tuck_delta_mean_m"] = self.jump_observer.max_bilateral_tuck_delta.mean()
            logs["Mixed/jump/landing_buffer_active_envs"] = self.jump_observer.landing_buffer_active.sum().float()
            logs["Mixed/jump/landing_buffer_used_envs"] = self.jump_observer.landing_buffer_used.sum().float()
            logs["Mixed/jump/apex_seen_envs"] = self.jump_observer.apex_seen.sum().float()
            logs["Mixed/jump/extension_phase_envs"] = self.jump_observer.extension_started.sum().float()
            jump = self.task_schedule.mask("jump")
            ref = self.task_schedule.reference(self._mixed_age())
            logs["Mixed/active_jump/requested_envs"] = ref["requested"].sum().float()
            logs["Mixed/active_jump/push_envs"] = ref["push"].sum().float()
            logs["Mixed/active_jump/height_command_mean_m"] = self.height_cmd[jump].mean()
            logs["Mixed/active_jump/root_height_mean_m"] = (
                self.robot.data.root_pos_w[jump, 2]-self.terrain.env_origins[jump, 2]).mean()
            logs["Mixed/active_jump/upward_velocity_mean_m_s"] = self.robot.data.root_lin_vel_w[jump, 2].mean()
            logs["Mixed/active_jump/leg_motor_abs_mean_Nm"] = self.motor_effort[jump][:, (0,1,3,4)].abs().mean()
            bad_gap = self.constraint_gaps().amax(-1) > .003
            logs["Port/constraint_gap_over_3mm_env_fraction"] = bad_gap.float().mean()
            for task in self.task_schedule.tasks:
                logs["Mixed/"+task+"/constraint_gap_over_3mm_env_fraction"] = (
                    bad_gap[self.task_schedule.mask(task)].float().mean())
            self._log_arcs_and_platforms(logs)
            if self.jump_observer.sequence is not None:
                sequence = self.jump_observer.sequence
                # PPO writes these snapshots at its update boundary. Sending
                # cumulative integer counts through ep_infos would average
                # them over rollout steps and recreate fractional "totals".
                if not getattr(self, "port_sequence_metrics_in_runner", False):
                    logs.update(sequence.metrics())
                # Explicitly distinguish paid contributions from legacy raw
                # shape logs; all of these include weights and policy_dt.
                for name in ("jump_push_velocity", "jump_release_energy", "jump_clearance", "jump_impact"):
                    for task in sequence.totals:
                        mask = self.task_schedule.mask(task)
                        logs["JumpSequence/"+task+"/paid_"+name+"_mean_per_step"] = self._last_reward_terms[name][mask].mean()
                for stage in ("tuck", "extension"):
                    j = self.cfg.port_mixed_spec["jump"]
                    paid = getattr(self.jump_observer, "last_air_"+stage)*j["dense_scale"]*2*j["air_"+stage+"_scale"]*self.step_dt
                    for task in sequence.totals:
                        logs["JumpSequence/"+task+"/paid_air_"+stage+"_mean_per_step"] = paid[self.task_schedule.mask(task)].mean()
        return rewards

    def _log_arcs_and_platforms(self, logs):
        s = self.task_schedule
        if self.cfg.port_mixed_spec.get("arc_commands"):
            arc = ((s.mask("forward") | s.mask("reverse") | s.mask("dash"))
                   & (self.command[:, 0].abs() > .2) & (self.command[:, 2].abs() > .1))
            logs["Mixed/arc/envs"] = arc.sum().float()
            if arc.any():
                velocity = self.robot.data.root_lin_vel_b
                logs["Mixed/arc/yaw_error_abs_mean_rad_s"] = (
                    self.robot.data.root_ang_vel_b[arc, 2]-self.command[arc, 2]).abs().mean()
                logs["Mixed/arc/vx_error_abs_mean_m_s"] = (velocity[arc, 0]-self.command[arc, 0]).abs().mean()
        o = self.platform_observer
        if o is None:
            return
        lane = s.mask("platform")
        over = lane & (s.mode == OVER)
        logs["Mixed/platform/height_limit_m"] = torch.tensor(float(s.height_limit), device=self.device)
        for mode, name in ((FLAT, "flat"), (OVER, "over"), (DRIVE_OFF, "drive_off")):
            logs["Mixed/platform/"+name+"_envs"] = (lane & (s.mode == mode)).sum().float()
        for name in o.EVENTS:
            logs["Mixed/platform/"+name.removesuffix("_event")+"_count_per_step"] = getattr(o, name).sum().float()
        logs["Mixed/platform/top_success_count_per_step"] = (self.jump_observer.success_event & over).sum().float()
        logs["Mixed/platform/requested_envs"] = (over & (s.delay < 999.)).sum().float()
        logs["Mixed/platform/on_top_envs"] = (o.seen_top & lane).sum().float()
        done = over & o.top_landed
        logs["Mixed/platform/edge_clearance_mean_m"] = (o.min_clearance[done].mean() if done.any()
            else torch.zeros((), device=self.device))

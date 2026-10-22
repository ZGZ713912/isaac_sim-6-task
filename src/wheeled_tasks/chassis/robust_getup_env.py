"""Native PhysX task: passive fall -> six-motor RL -> standing -> locomotion.

ChassisEnv owns the authenticated mechanism, contact material and PD setup.
This explicit task owns its reset distribution, reward and termination rules;
the old recovery FSM, scripted torques and ordinary fall termination are never
called. No physical state is written between episode resets.
"""
from __future__ import annotations

import json
import math

import torch
import warp as wp

from .env import ChassisEnv
from .root_kinematics import root_link_linear_velocity, update_robot_data
from .scut_observation import CONTROL_FROM_POLICY
from .unified_getup import FAMILIES, classify_gravity, joint_tracking_score, project_commands, stable_standing, push_window
from .robust_getup import RobustDomain
from .getup_feedback import GetupFeedback
from .getup_reward import potential_reward
from .quiet_standing import QuietStanding, standing_request_mask, standing_support_cost, standing_position_cost


class RobustGetupEnv(ChassisEnv):
    def __init__(self, config, manifest, control, root, *, bank, settings, fixed_cases=None, **kwargs):
        self.unified_ready = False
        self.robust_ready = False
        super().__init__(config, manifest, control, root, **kwargs)
        if self.recovery is not None or self.nonstandard_start is not None:
            raise ValueError("Unified RL cannot stack with scripted recovery/capture")
        self.getup_cfg, self.fixed_cases = settings, fixed_cases
        self.bank = json.loads(bank.read_text())
        if self.bank["asset_manifest_sha256"] != config["asset_manifest_sha256"]:
            raise ValueError("Recovery bank belongs to another mechanism")
        self.poses = self.bank["poses"]
        self.bank_joints = torch.tensor([[p["full_joint_positions"][name] for name in self.robot.joint_names]
                                         for p in self.poses], device=self.device)
        self.bank_roots = torch.tensor([p["root_position_m"] + p["root_quaternion_xyzw"] for p in self.poses], device=self.device)
        self.bank_family = torch.tensor([FAMILIES.index(p["family"]) for p in self.poses], device=self.device)
        self.bank_tilt = torch.tensor([max(abs(p["roll_deg"]), abs(p["pitch_deg"])) for p in self.poses], device=self.device)
        self.family_pools = [(self.bank_family == i).nonzero().flatten() for i in range(len(FAMILIES))]
        n, device = self.num_envs, self.device
        self.actor_updates = 0
        self.pose_index = torch.zeros(n, dtype=torch.long, device=device)
        self.spawn_family = torch.zeros_like(self.pose_index)
        self.settled_family = torch.full_like(self.pose_index, -1)
        self.recovery_lane = torch.arange(n, device=device) % 10 >= round(settings["motion_fraction"] * 10)
        self.passive = torch.zeros(n, dtype=torch.bool, device=device)
        self.recovery_active = self.passive.clone()
        self.released = self.passive.clone()
        self.ever_recovered = self.passive.clone()
        self.followup_success = self.passive.clone()
        self.finished = self.passive.clone()
        self.stable_hold = torch.zeros(n, device=device)
        self.passive_hold = torch.zeros(n, device=device)
        self.followup_hold = torch.zeros(n, device=device)
        self.recovered_at = torch.full((n,), float("nan"), device=device)
        self.previous_potential = torch.zeros(n, device=device)
        self.pending_command = torch.zeros(n, 3, device=device)
        self.goal_command = self.pending_command.clone()
        self.last_applied_command = self.pending_command.clone()
        self.push_selected = torch.zeros(n, dtype=torch.bool, device=device)
        self.push_sign = torch.ones(n, device=device)
        self.push_force = torch.zeros(n, 1, 3, device=device)
        self.push_torque = torch.zeros_like(self.push_force)
        self.push_body_ids = torch.tensor([self.robot.body_names.index("base_link")], device=device)
        self.sequence_time = torch.full((n, 3), float("inf"), device=device)
        self.sequence_command = torch.zeros(n, 3, 3, device=device)
        self.in_motion_fall = torch.zeros(n, dtype=torch.bool, device=device)
        self.recovered_after_push = self.in_motion_fall.clone()
        self.contact_impulse_history = torch.zeros(round(settings["impact_window_seconds"] / self.dt), n, device=device)
        self.impulse_cursor = 0
        self.max_episode_length = round(settings["episode_seconds"] / self.policy_dt)
        self.episode_limits.fill_(self.max_episode_length)
        self.v5.action_bounds = self.v5.action_bounds.new_tensor([settings["leg_action_clip"], settings["leg_action_clip"],
            settings["wheel_action_clip"], settings["leg_action_clip"], settings["leg_action_clip"], settings["wheel_action_clip"]])
        self.v5.settings = dict(self.v5.settings, action_clip=settings["leg_action_clip"], wheel_action_clip=settings["wheel_action_clip"])
        self.domain = RobustDomain(n, device, settings, kwargs["seed"], self.dt)
        if self.dynamics_randomization is not None:
            self.domain.enabled.copy_(self.dynamics_randomization.enabled)
        self.getup_feedback = GetupFeedback(n, device, self.domain.enabled, settings, self.domain.rng)
        self.paired_feedback_enabled = torch.full((n, 1), settings.get("paired_feedback_winding", False),
                                                  device=device, dtype=torch.bool)
        native_efforts = self.v5.motor_efforts
        native_response = self.actuator_response.apply_torque
        native_springs = self.v5.spring_efforts

        def motor_efforts(q, dq, legs, wheels):
            q, dq = self.getup_feedback.observe(q, dq)
            if settings.get("paired_feedback_winding", False):
                from .paired_feedback import paired_position_error
                indices = list(self.v5.LEGS)
                aligned = legs - paired_position_error(legs, q[:, indices])
                q = q.clone()
                q[:, indices] = torch.where(self.paired_feedback_enabled, aligned, q[:, indices])
            tau = native_efforts(q, dq, legs, wheels)
            indices = list(self.v5.LEGS)
            correction = ((self.domain.kp - 1.) * self.v5.settings["leg_kp"] * (legs - q[:, indices])
                          - (self.domain.kd - 1.) * self.v5.settings["leg_kd"] * dq[:, indices])
            tau[:, indices] = (tau[:, indices] + correction).clamp(-40., 40.)
            return tau

        self.v5.motor_efforts = motor_efforts
        self.actuator_response.apply_torque = lambda tau: native_response(self.domain.apply_torque(tau, self.passive | self.finished))
        self.v5.spring_efforts = lambda position: native_springs(position) * self.domain.spring
        self.previous_effort = torch.zeros(n, 6, device=device)
        self.quiet_standing = QuietStanding(n, device, self.policy_dt, settings["quiet_sway_m"],
                                           settings.get("quiet_position_weight", .5))
        self.wheel_air_run = torch.zeros(n, 2, device=device)
        self.dual_wheel_air_run = torch.zeros(n, device=device)
        for name in ("L_link3", "R_link3"):
            body = next(body for body in self.model_spec["bodies"] if body["name"] == name)
            collider = body["collisions"][0]
            if (len(body["collisions"]) != 1 or collider["type"] != "cylinder"
                    or collider["radius"] != .06
                    or abs(collider["length"] - .02500000037252903) > 1e-9
                    or any(collider["origin"][i][j] != float(i == j) for i in range(3) for j in range(3))):
                raise ValueError("Standing clearance requires the authenticated 107 wheel cylinders")
        self.robust_ready = True
        self.unified_ready = True
        self.reset(torch.arange(n, device=device))
        self.startup_report["unified_getup"] = dict(settings=settings, bank_poses=len(self.poses),
            full_rl=True, scripted_torque_calls=0, motor_effort_during_settle="zero_all_six",
            contact_support="temporary nonwheel support allowed; mechanical/impact protection retained",
            state_writes="episode_reset_only", policy_history="35D single frame with previous action",
            self_collision_scope=self.bank["self_collision_scope"])
        self.startup_report["robust_domains"] = self.domain.summary()
        self.startup_report["getup_ground"] = dict(tile_face_m=8., half_extent_xy_m=[48., 48.],
            termination_xy_m=settings["boundary_xy_m"], max_requested_path_12s_m=36.,
            source="separate tiled flat patches for arbitrary reset heading")

    @torch.inference_mode()
    def reset(self, ids):
        super().reset(ids)
        if not self.unified_ready:
            return
        count = len(ids)
        self.sequence_time[ids] = float("inf")
        self.push_selected[ids] = False
        chosen = torch.zeros(count, dtype=torch.long, device=self.device)
        if self.fixed_cases is not None:
            for j, lane in enumerate(ids.cpu().tolist()):
                case = self.fixed_cases[lane]
                chosen[j] = case["pose_index"]
                self.recovery_lane[lane] = case["recovery"]
                self.pending_command[lane] = self.pending_command.new_tensor(case["command"])
                self.push_selected[lane] = case.get("push", False)
                self.push_sign[lane] = case.get("push_sign", 1.)
                for stage, (start, command) in enumerate(case.get("sequence", [])):
                    self.sequence_time[lane, stage] = start
                    self.sequence_command[lane, stage] = self.pending_command.new_tensor(command)
        else:
            relative = max(0, self.actor_updates - self.getup_cfg["origin_updates"])
            progress = min(1., relative / self.getup_cfg["curriculum_updates"])
            weights = torch.tensor([0., .1, .1, .2, .2, .3, .1], device=self.device)
            family = torch.multinomial(weights, count, replacement=True, generator=self.generator)
            # Easy poses dominate initially, but hard side/inverted births have
            # a nonzero quota from the first block. Selection happens at reset.
            hard = self.random(count) < (.5 + .5 * progress)
            family = torch.where(hard, family, 1 + (self.random(count) > .5).long())
            for f, pool in enumerate(self.family_pools):
                selected = (family == f).nonzero().flatten()
                easy = pool[self.bank_tilt[pool] <= 35]
                if relative < self.getup_cfg["curriculum_updates"] // 2 and len(easy):
                    pool = easy
                elif f != 0:
                    minimum = 150 if f == FAMILIES.index("inverted") else 60
                    difficult = pool[self.bank_tilt[pool] >= minimum]
                    if len(difficult):
                        pool = difficult
                if len(selected):
                    chosen[selected] = pool[(self.random(len(selected)) * len(pool)).long().clamp_max(len(pool) - 1)]
                    if f == FAMILIES.index("inverted"):
                        exact = pool[self.bank_tilt[pool] == 180]
                        forced = selected[self.random(len(selected)) < .6]
                        if len(exact) and len(forced):
                            chosen[forced] = exact[(self.random(len(forced)) * len(exact)).long().clamp_max(len(exact) - 1)]
            upright_pool = self.family_pools[0]
            motion = ~self.recovery_lane[ids]
            chosen[motion] = upright_pool[1]  # aligned 65-degree mechanism, no root tilt
            self.sample_commands(ids)
            self.push_selected[ids] = ~self.recovery_lane[ids] & (self.random(count) < self.getup_cfg["motion_push_fraction"])
            self.push_sign[ids] = torch.where(self.random(count) < .5, -1., 1.)
        self.pose_index[ids] = chosen
        self.spawn_family[ids] = self.bank_family[chosen]
        self.settled_family[ids] = -1
        roots = self.bank_roots[chosen].clone()
        roots[:, :3] += self.origins[ids]
        # Random heading leaves gravity and exact ground clearance unchanged.
        if self.fixed_cases is None:
            angle = (self.random(count) * 2 - 1) * math.pi
            s, c = (angle / 2).sin(), (angle / 2).cos()
            x, y, z, w = roots[:, 3:].unbind(-1)
            roots[:, 3:] = torch.stack((c*x - s*y, s*x + c*y, c*z + s*w, c*w - s*z), -1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=roots, env_ids=ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(count, 6, device=self.device), env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=self.bank_joints[chosen], env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(count, self.joint_count, device=self.device), env_ids=ids)
        self.robot.reset(ids)
        update_robot_data(self, self.dt)
        self.passive[ids] = self.recovery_lane[ids]
        self.recovery_active[ids] = self.recovery_lane[ids]
        self.released[ids] = ~self.recovery_lane[ids]
        self.ever_recovered[ids] = False
        self.followup_success[ids] = False
        self.finished[ids] = False
        self.in_motion_fall[ids] = False
        self.recovered_after_push[ids] = False
        self.stable_hold[ids] = self.passive_hold[ids] = self.followup_hold[ids] = 0.
        self.recovered_at[ids] = float("nan")
        self.goal_command[ids] = self.pending_command[ids]
        if self.getup_cfg["hold_motion_during_recovery"]:
            self.goal_command[ids[self.recovery_lane[ids]], :2] = 0.
        self.commands[ids] = project_commands(self.goal_command[ids], self.getup_cfg)
        self.raw_height[ids] = self.commands[ids, 2]
        self._measured_action_history(ids)
        self.previous_actions[ids] = self.actions[ids]
        self.before_previous_actions[ids] = self.actions[ids]
        self.previous_potential[ids] = self.potential()[ids]
        self.contact_impulse_history[:, ids] = 0.
        self.history.reset(ids)
        if self.robust_ready:
            self.quiet_standing.reset(ids)
            self.wheel_air_run[ids] = 0.
            self.dual_wheel_air_run[ids] = 0.
            self.domain.reset(ids, ChassisEnv.get_observations(self)["policy"][ids, 4:22])
            self.getup_feedback.reset(ids, self.robot.data.joint_pos.torch[ids][:, self.ids],
                                      self.robot.data.joint_vel.torch[ids][:, self.ids])
            self.previous_effort[ids] = 0.

    def get_observations(self):
        observations = super().get_observations()
        if self.robust_ready:
            frame = observations["policy"].clone()
            frame[:, 4:22] = self.domain.observe(frame[:, 4:22], self.tick)
            observations["policy"] = frame
            if self.getup_cfg["quiet_critic_anchor"]:
                xy = self.robot.data.root_link_pose_w.torch[:, :2] - self.origins[:, :2]
                # All getup patches are flat. Replace the two redundant ground
                # height channels, explicitly versioned in the task contract.
                observations["critic"][:, -2:] = (20. * self.quiet_standing.displacement(xy)).clamp(-5., 5.)
        return observations

    def _measured_action_history(self, ids):
        q = self.robot.data.joint_pos.torch[ids][:, self.ids]
        delta = q[:, list(self.v5.LEGS)] - self.v5.nominal[list(self.v5.LEGS)]
        self.actions[ids, :] = 0.
        # Indexing twice would update a temporary rather than the real history.
        values = torch.zeros(len(ids), 6, device=self.device)
        if self.getup_cfg.get("cyclic_leg_targets", False):
            from wheeled_algo.paired_recovery import paired_offsets
            values[:, list(self.v5.LEGS)] = paired_offsets(delta) / .25
        else:
            values[:, list(self.v5.LEGS)] = torch.atan2(delta.sin(), delta.cos()) / .25
        self.actions[ids] = values

    def sample_commands(self, ids):
        count = len(ids)
        pick = self.random(count)
        reverse = self.random(count) < .5
        vx = (.25 + 1.25 * self.random(count)) * torch.where(reverse, -1., 1.)
        high = self.random(count) < .2
        vx = torch.where(high, (1.5 + self.random(count) * torch.where(reverse, 1., 1.5)) * torch.where(reverse, -1., 1.), vx)
        yaw = (.3 + 3.7 * self.random(count)) * torch.where(self.random(count) < .5, -1., 1.)
        # Independent axes, not alternating straight/spin commands: 30% stand,
        # 20% straight, 10% spin, 40% arcs, in every learning block.
        vx[pick < .3] = 0.
        vx[(pick >= .5) & (pick < .6)] = 0.
        yaw[pick < .5] = 0.
        self.pending_command[ids, 0] = vx
        self.pending_command[ids, 1] = yaw
        if self.getup_cfg.get("arc_tuning_fraction", 0.):
            from .arc_rehearsal import moderate_arc_commands
            # Only this opt-in pilot changes command exposure. Its remaining
            # motion lanes retain the original stand/straight/spin/fast mix.
            selected = ids[(~self.recovery_lane[ids])
                           & (self.random(count) < self.getup_cfg["arc_tuning_fraction"])]
            draws = torch.rand(len(selected), 4, device=self.device, generator=self.generator)
            self.pending_command[selected, :2] = moderate_arc_commands(draws)
        if self.getup_cfg["quiet_training"]:
            stationary = ids[~self.recovery_lane[ids]]
            self.pending_command[stationary, :2] = 0.
        height_pick = self.random(count)
        mid, low, _ = self.getup_cfg.get("quiet_height_probabilities", [.5, .25, .25])
        self.pending_command[ids, 2] = torch.where(height_pick < mid, .305,
            torch.where(height_pick < mid + low, .23, .42))

    def potential(self):
        _, _, gravity, height, _, _, _ = self.state()
        wheel_z = self.wheel_centers()[:, :, 2] - self.origins[:, 2:3] - .06
        reach = torch.exp(-wheel_z.clamp_min(0.) / .12).mean(-1)
        return (1. - gravity[:, 2]) * .5 + .5 * (height / .305).clamp(0., 1.) + .3 * reach

    @torch.inference_mode()
    def step(self, actions):
        active_interval = ~self.passive.clone()
        previous = self.actions.clone()
        self.previous_actions.copy_(self.actions)
        q = self.robot.data.joint_pos.torch[:, self.ids]
        legs, wheels, clipped = self.v5.decode(actions[:, CONTROL_FROM_POLICY], q)
        self.actions.copy_(clipped)
        gap_peak = torch.zeros(self.num_envs, device=self.device)
        impact_peak = torch.zeros_like(gap_peak)
        impulse_peak = torch.zeros_like(gap_peak)
        mechanical = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        motor_peak_passive = torch.zeros_like(gap_peak)
        wheel_clearance_peak = torch.full((self.num_envs, 2), -torch.inf, device=self.device)
        wheel_contact_fraction = torch.zeros_like(wheel_clearance_peak)
        wheel_air_run_peak = self.wheel_air_run.clone()
        dual_wheel_air_run_peak = self.dual_wheel_air_run.clone()
        wheel_air_seconds = torch.zeros_like(wheel_clearance_peak)
        pushing = push_window(self.episode_length_buf * self.policy_dt, self.push_selected,
                              self.passive, self.finished, self.getup_cfg)
        self.push_force.zero_()
        self.push_torque.zero_()
        self.push_force[:, 0, 1] = pushing * self.push_sign * self.getup_cfg["push_force_n"]
        self.push_torque[:, 0, 0] = (pushing * self.push_sign * self.getup_cfg["push_torque_nm"]
                                   * self.getup_cfg["push_torque_direction"])
        for _ in range(self.decimation):
            q = self.robot.data.joint_pos.torch[:, self.ids]
            dq = self.robot.data.joint_vel.torch[:, self.ids]
            tau = self.v5.motor_efforts(q, dq, legs, wheels)
            tau = self.actuator_response.apply_torque(tau)
            tau.clamp_(min=-self.v5.current_motor_bounds, max=self.v5.current_motor_bounds)
            tau[self.passive | self.finished] = 0.
            motor_peak_passive = torch.maximum(motor_peak_passive, tau.abs().amax(-1) * self.passive)
            effort = torch.zeros_like(self.nominal)
            effort[:, self.ids] = tau
            effort[:, self.spring_ids] = self.v5.spring_efforts(self.robot.data.joint_pos.torch[:, self.spring_ids])
            if bool(pushing.any()):
                self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
                    forces=self.push_force, torques=self.push_torque, body_ids=self.push_body_ids, is_global=False)
            self.robot.set_joint_effort_target_index(target=effort)
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            update_robot_data(self, self.dt)
            for ids, view in self.contact_views:
                matrix = wp.to_torch(view.get_contact_force_matrix(dt=self.dt))
                self.contact_force[ids] = matrix.reshape(len(ids), self.body_count, -1, 3).sum(2)
            from .standing_support import wheel_bottom_clearance
            wheel_pose = self.robot.data.body_link_pose_w.torch[:, self.wheel_ids]
            wheel_clearance = wheel_bottom_clearance(self.wheel_centers(), wheel_pose[:, :, 3:], self.origins[:, 2:3])
            wheel_clearance_peak = torch.maximum(wheel_clearance_peak, wheel_clearance)
            wheel_contact_fraction += (self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2.).float() / self.decimation
            geometric_air = wheel_clearance > .003
            self.wheel_air_run = torch.where(geometric_air, self.wheel_air_run + self.dt, 0.)
            self.dual_wheel_air_run = torch.where(geometric_air.all(-1), self.dual_wheel_air_run + self.dt, 0.)
            wheel_air_run_peak = torch.maximum(wheel_air_run_peak, self.wheel_air_run)
            dual_wheel_air_run_peak = torch.maximum(dual_wheel_air_run_peak, self.dual_wheel_air_run)
            wheel_air_seconds += geometric_air * self.dt
            gap_peak = torch.maximum(gap_peak, self.closure_gap())
            body_force = self.contact_force.norm(dim=-1).amax(-1)
            impact_peak = torch.maximum(impact_peak, body_force)
            # A single 1 ms force spike is not a sustained destructive impact.
            # Integrate the peak body force over a physical 20 ms window while
            # retaining every raw peak in the evidence and reward diagnostics.
            self.contact_impulse_history[self.impulse_cursor].copy_(body_force * self.dt)
            self.impulse_cursor = (self.impulse_cursor + 1) % len(self.contact_impulse_history)
            impulse_peak = torch.maximum(impulse_peak, self.contact_impulse_history.sum(0))
            compression, _ = self.v5.spring_state(self.robot.data.joint_pos.torch[:, self.spring_ids],
                                                  self.robot.data.joint_vel.torch[:, self.spring_ids])
            mechanical |= ((compression < -.001) | (compression > self.v5.stroke + .001)).any(-1)
            for name, index in zip(("L_joint2", "R_jonit2"), self.knee_ids):
                lo, hi = self.knee_bounds[name]
                joint = self.robot.data.joint_pos.torch[:, index]
                mechanical |= (joint < lo - .03) | (joint > hi + .03)
        self.tick += 1
        self.episode_length_buf += 1
        velocity, omega, gravity, height, q, dq, local = self.state()
        velocity = root_link_linear_velocity(self, frame="body")
        if not bool(torch.isfinite(torch.cat((velocity, omega, gravity, q, dq), -1)).all()):
            raise RuntimeError("Nonfinite unified recovery transition")
        elapsed = self.episode_length_buf * self.policy_dt
        contact = self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2.
        settled = (omega.norm(dim=-1) < .7) & (dq.abs().amax(-1) < 3.)
        self.passive_hold = torch.where(self.passive & settled, self.passive_hold + self.policy_dt, 0.)
        enable = self.passive & (elapsed >= self.getup_cfg["passive_min_seconds"]) & (
            (self.passive_hold >= self.getup_cfg["passive_hold_seconds"]) | (elapsed >= self.getup_cfg["passive_max_seconds"]))
        self.settled_family[enable] = classify_gravity(gravity)[enable]
        self.passive[enable] = False
        # Passive placement impacts are recorded but are not actor failures.
        # Start the active impact window at enable, with no stale passive force.
        self.contact_impulse_history[:, enable] = 0.
        self._measured_action_history((~active_interval).nonzero().flatten())
        stable = stable_standing(gravity, omega, height, contact, dq) & ~self.passive
        self.stable_hold = torch.where(stable, self.stable_hold + self.policy_dt, 0.)
        recovered = self.recovery_active & (self.stable_hold >= self.getup_cfg["stable_seconds"])
        self.recovery_active[recovered] = False
        self.released[recovered] = True
        first = recovered & ~self.ever_recovered
        self.recovered_at[first] = elapsed[first]
        self.ever_recovered |= recovered
        self.recovered_after_push |= recovered & self.in_motion_fall & self.push_selected
        # A fall while moving starts recovery in-place; never reset the body.
        fallen = ~self.passive & ((gravity[:, 2] > -.7) | (height < .13))
        self.in_motion_fall |= fallen & ~self.recovery_lane & (elapsed >= self.getup_cfg["push_start_seconds"])
        self.recovery_active |= fallen
        self.released &= ~fallen
        self.followup_success[fallen] = False
        if self.fixed_cases is not None:
            for stage in range(self.sequence_time.shape[1]):
                change = elapsed >= self.sequence_time[:, stage]
                self.pending_command[change] = self.sequence_command[change, stage]
        self.goal_command.copy_(self.pending_command)
        if self.getup_cfg["hold_motion_during_recovery"]:
            self.goal_command[self.recovery_active | self.passive, :2] = 0.
        error_vx = velocity[:, 0] - self.commands[:, 0]
        error_yaw = omega[:, 2] - self.commands[:, 1]
        tracking = joint_tracking_score(error_vx, error_yaw)
        upright = ((-gravity[:, 2] - .5) / .5).clamp(0., 1.)
        followup = (self.ever_recovered & self.released & stable
                    & (error_vx.abs() < .25) & (error_yaw.abs() < .4))
        self.followup_hold = torch.where(followup, self.followup_hold + self.policy_dt, 0.)
        self.followup_success |= self.followup_hold >= self.getup_cfg["followup_seconds"]
        potential = self.potential()
        gap_bad = gap_peak > self.getup_cfg["closure_gap_m"]
        impact_bad = (impulse_peak > self.getup_cfg["impact_impulse_limit_ns"]) & active_interval
        boundary_bad = (local[:, :2].abs() > local.new_tensor(self.getup_cfg["boundary_xy_m"])).any(-1)
        terminated = mechanical | gap_bad | impact_bad | boundary_bad
        progress = potential_reward(self.previous_potential, potential, terminated,
                                    self.getup_cfg["reward_gamma"], self.getup_cfg["reward_potential_gain"])
        self.previous_potential.copy_(potential)
        torque_cost = .0003 * tau.square().mean(-1)
        height_score = torch.exp(-((height - self.commands[:, 2]) / .08).square())
        quiet = ((self.commands[:, :2].abs() < self.commands.new_tensor([.1, .2])).all(-1)
                 & (gravity[:, 2] < -.94) & ~self.passive & ~self.recovery_active)
        effort_delta = (tau - self.previous_effort).square().mean(-1) / 100.
        difference = self.actions - previous
        if self.getup_cfg.get("cyclic_leg_targets", False):
            from wheeled_algo.paired_recovery import wrap
            difference[:, list(self.v5.LEGS)] = wrap(difference[:, list(self.v5.LEGS)] * .25) / .25
        action_delta = difference.square().mean(-1)
        leg_speed = dq[:, list(self.v5.LEGS)].square().mean(-1)
        quiet_cost = quiet * (self.getup_cfg["quiet_action_rate_cost"] * action_delta
            + self.getup_cfg["quiet_motor_speed_cost"] * leg_speed
            + self.getup_cfg["quiet_effort_delta_cost"] * effort_delta)
        quiet_request = (standing_request_mask(self.commands, self.passive, self.recovery_lane, self.ever_recovered)
                         if self.getup_cfg.get("standing_support_training", False) else quiet)
        spectral_cost, target_power, velocity_power, quiet_radius = self.quiet_standing.update(
            quiet_request, local[:, :2],
            self.actions * self.actions.new_tensor([.25, .25, 10., .25, .25, 10.]), dq)
        if self.getup_cfg["quiet_spectral_cost"]:
            quiet_cost += spectral_cost * (elapsed > 2.)
        support_cost = standing_support_cost(wheel_clearance_peak, wheel_contact_fraction)
        if self.getup_cfg.get("standing_support_training", False):
            quiet_cost += quiet_request * support_cost * (elapsed >= .2)
        position_cost = standing_position_cost(quiet_radius, self.getup_cfg["quiet_sway_m"])
        quiet_cost += (quiet_request * (elapsed >= .2) * position_cost
                       * self.getup_cfg.get("standing_early_position_weight", 0.))
        self.previous_effort.copy_(tau)
        density = (2. * (1. - gravity[:, 2]) * .5 + upright * (2. * height_score + 4. * tracking)
                   + 2. * contact.float().mean(-1)
                   - .02 * omega.square().sum(-1) - torque_cost
                   - .002 * action_delta
                   - self.getup_cfg["recovery_leg_speed_cost"] * self.recovery_active * leg_speed
                   - .0002 * (impact_peak - 400.).clamp_min(0.) - .5 * self.recovery_active - quiet_cost)
        reward = (density * self.policy_dt + progress + 3. * first) * active_interval
        timeout = self.episode_length_buf >= self.max_episode_length
        done = terminated | timeout
        reward -= 3. * terminated
        info = dict(time_outs=timeout & ~terminated, policy_learning_mask=active_interval,
                    policy_bootstrap_mask=~self.passive & ~terminated,
                    log={"UnifiedGetup/recovery_active_fraction": self.recovery_active.float().mean(),
                         "UnifiedGetup/upright_fraction": stable.float().mean(),
                         "UnifiedGetup/recovered_events": first.float().sum(),
                         "UnifiedGetup/episode_success_events": (done & self.followup_success).float().sum(),
                         "UnifiedGetup/terminated_events": terminated.float().sum(),
                         "UnifiedGetup/episode_events": done.float().sum(),
                         "UnifiedGetup/passive_motor_peak_nm": motor_peak_passive.max(),
                         "UnifiedGetup/closure_gap_max_m": gap_peak.max(),
                         "UnifiedGetup/impact_peak_n": impact_peak.max(),
                         "UnifiedGetup/contact_impulse_peak_ns": impulse_peak.max(),
                         "UnifiedGetup/external_push_fraction": pushing.float().mean(),
                         "UnifiedGetup/in_motion_fall_fraction": self.in_motion_fall.float().mean(),
                         "UnifiedGetup/recovered_after_push_fraction": self.recovered_after_push.float().mean(),
                         "UnifiedGetup/mixed_command_fraction": ((self.commands[:, 0].abs() > .05) & (self.commands[:, 1].abs() > .05)).float().mean(),
                         "UnifiedGetup/moderate_arc_active_fraction": ((self.commands[:, 0].abs() >= .4)
                             & (self.commands[:, 0].abs() <= 1.25) & (self.commands[:, 1].abs() >= .4)
                             & (self.commands[:, 1].abs() <= 1.4) & ~self.passive & ~self.recovery_active).float().mean(),
                         "UnifiedGetup/vx_mae_m_s": error_vx.abs().mean(),
                         "UnifiedGetup/yaw_mae_rad_s": error_yaw.abs().mean()})
        standing_request = (self.commands[:, :2].abs() < self.commands.new_tensor([.1, .2])).all(-1) & ~self.passive
        info["log"].update({
            "Standing/wheel_clearance_peak_m": wheel_clearance_peak[standing_request].max() if standing_request.any() else gap_peak.new_zeros(()),
            "Standing/dual_support_fraction": (wheel_contact_fraction[standing_request] == 1.).all(-1).float().mean() if standing_request.any() else gap_peak.new_zeros(()),
            "Standing/geometric_air_fraction": (wheel_clearance_peak[standing_request] > .003).any(-1).float().mean() if standing_request.any() else gap_peak.new_zeros(())})
        info["log"].update({"Quiet/active_fraction": quiet.float().mean(),
            "Quiet/action_delta_rms": action_delta[quiet].mean().sqrt() if quiet.any() else action_delta.new_zeros(()),
            "Quiet/leg_speed_rms_rad_s": leg_speed[quiet].mean().sqrt() if quiet.any() else leg_speed.new_zeros(()),
            "Quiet/effort_delta_rms_nm": (100. * effort_delta[quiet].mean()).sqrt() if quiet.any() else effort_delta.new_zeros(()),
            "Quiet/fast_target_normalized_power": target_power[quiet].mean() if quiet.any() else target_power.new_zeros(()),
            "Quiet/fast_velocity_normalized_power": velocity_power[quiet].mean() if quiet.any() else velocity_power.new_zeros(()),
            "Quiet/position_radius_m": quiet_radius[quiet].mean() if quiet.any() else quiet_radius.new_zeros(()),
            "Standing/request_position_radius_m": quiet_radius[quiet_request].mean() if quiet_request.any() else quiet_radius.new_zeros(()),
            "Standing/support_cost_density": support_cost[quiet_request].mean() if quiet_request.any() else support_cost.new_zeros(()),
            "Standing/position_deadband_cost_density": position_cost[quiet_request].mean() if quiet_request.any() else position_cost.new_zeros(()),
            "Robust/domain_fraction": self.domain.enabled.float().mean(),
            "Robust/motor_strength_mean": self.domain.strength.mean(),
            "Robust/sensor_delay_mean_ms": self.domain.sensor_delay.float().mean() * 5.})
        for f, name in enumerate(FAMILIES):
            mask = self.settled_family == f
            info["log"][f"UnifiedGetup/post_settle/{name}_episodes"] = (enable & mask).float().sum()
            info["log"][f"UnifiedGetup/post_settle/{name}_recoveries"] = (first & mask).float().sum()
        if self.cfg.get("record_diagnostics"):
            info["diagnostics"] = dict(velocity=velocity.clone(), omega=omega.clone(), gravity=gravity.clone(),
                height=height.clone(), commands=self.commands.clone(), requested_command=self.pending_command.clone(),
                actions=self.actions.clone(), motor_effort=tau.clone(), motor_velocity=dq.clone(),
                gap=gap_peak.clone(), impact=impact_peak.clone(), impulse=impulse_peak.clone(), passive=~active_interval,
                passive_motor_peak=motor_peak_passive.clone(), enable_event=enable.clone(),
                spawn_family=self.spawn_family.clone(), settled_family=self.settled_family.clone(),
                recovered=self.ever_recovered.clone(), recovered_at=self.recovered_at.clone(),
                followup_success=self.followup_success.clone(), stable=stable.clone(),
                done=done.clone(), terminated=terminated.clone(), mechanical=mechanical.clone(),
                closure_gap=gap_bad.clone(), impact_bad=impact_bad.clone(), boundary_bad=boundary_bad.clone(), time=elapsed.clone(),
                local_position=local.clone(), motor_position=q.clone(), wheel_contact=contact.clone())
            info["diagnostics"].update(wheel_clearance=wheel_clearance.clone(),
                wheel_clearance_peak=wheel_clearance_peak.clone(), wheel_contact_fraction=wheel_contact_fraction.clone(),
                wheel_air_run_peak=wheel_air_run_peak.clone(), dual_wheel_air_run_peak=dual_wheel_air_run_peak.clone(),
                wheel_air_seconds=wheel_air_seconds.clone())
            info["diagnostics"].update(pushing=pushing.clone(), in_motion_fall=self.in_motion_fall.clone(),
                                       recovered_after_push=self.recovered_after_push.clone())
        if self.fixed_cases is None:
            change = (self.episode_length_buf % round(4. / self.policy_dt) == 0) & ~done
            if self.getup_cfg["quiet_training"]:
                change &= self.recovery_lane
            if bool(change.any()):
                self.sample_commands(change.nonzero().flatten())
            if bool(done.any()):
                self.reset(done.nonzero().flatten())
                info["policy_bootstrap_mask"][done] = ~self.passive[done] & ~terminated[done]
        else:
            self.finished |= done
        # Limit once before the next actor observation, using the RMCS priority.
        self.commands.copy_(project_commands(self.goal_command, self.getup_cfg))
        self.raw_height.copy_(self.commands[:, 2])
        self.history.last_tick = None
        return self.get_observations(), reward, done, info

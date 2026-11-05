"""Native flip-only environment and an explicitly masked script-to-actor course."""
from copy import deepcopy
import math

import torch

from .prone_motion_env import ProneMotionEnv
from . import rollover as task
from .recovery_impact import RecoveryImpact
from .scut_observation import POLICY_FROM_CONTROL


class RolloverEnv(ProneMotionEnv):
    def __init__(self, *args, **kwargs):
        self.rollover_ready = False
        self._rollover_hook = False
        super().__init__(*args, **kwargs)
        if self.cfg.get("prone_motion_performance") != "eager":
            raise ValueError("The rollover native capture hook is admitted on the eager backend only")
        self.whole_impact = RecoveryImpact(self.num_envs, self.device, self.robot.body_names,
            self.plan["recovery_impact"], self.dt, self.body_mass.sum(-1))
        self.rollover_contact = torch.zeros_like(self.contact_force)
        # Retain the original task's contact read/guard. This separate read is
        # after an actual completed physical update and cannot overwrite it.
        native_updated = self.control_audit.physics_updated
        def updated(dt):
            result = native_updated(dt)
            self._observe_rollover_physics(dt)
            return result
        self.control_audit.physics_updated = updated
        self._rollover_hook = True
        self.startup_report["rollover_training"] = dict(self.rollover_config,
            actual_training_pose_indices=self.roof_train, holdout_pose_indices=self.roof_holdout,
            capture_clock="consecutive_completed_native_1ms_steps",
            prefix_release="only_actor_previous_action_coordinates_zeroed; no_physical_or_controller_reset",
            legacy_standing_curriculum_disabled=True, standing_completion_claim=False)
        self.startup_report["prone_motion"].update(
            motor_owner="script prefix explicitly excluded from PPO, then six-axis trainable actor",
            reset="inverted authenticated bank, independent left/right orbits and mechanical knees",
            route_bias="soft110deg and alignment guidance concurrent with actual flip progress",
            fallback_script=False)
        self.reset(torch.arange(self.num_envs, device=self.device))

    def _initialize_rollover(self):
        if self.rollover_ready:
            return
        self.rollover_config = task.validate(self.plan["rollover_training"])
        self.prefix_course = task.PrefixCurriculum(self.seed_value if hasattr(self, "seed_value") else
                                                  self.plan["seed"], self.rollover_config)
        self.roof_train = [i for i,p in enumerate(self.poses) if p.get("recovery", {}).get("family") == "roof_down"
                           and p["recovery"]["split"] == "train"]
        self.roof_holdout = [i for i,p in enumerate(self.poses) if p.get("recovery", {}).get("family") == "roof_down"
                             and p["recovery"]["split"] == "holdout"]
        if not self.roof_train or set(self.roof_train) & set(self.roof_holdout):
            raise ValueError("Rollover must retain disjoint inverted training and holdout pools")
        for split in (self.roof_train, self.roof_holdout):
            classes = {self.poses[i]["recovery"]["leg_class"] for i in split}
            if classes != set(self.rollover_config["leg_classes"]):
                raise ValueError("Both rollover splits must cover all four independent leg classes")
        self.prefix_remaining = torch.zeros(self.num_envs, device=self.device)
        self.prefix_initial = torch.zeros_like(self.prefix_remaining)
        self.prefix_interval = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prefix_release_event = torch.zeros_like(self.prefix_interval)
        self.rollover_enabled_at = torch.full_like(self.prefix_remaining, float("nan"))
        self.native_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.native_condition = torch.zeros_like(self.prefix_interval)
        self.ended_recorded = torch.zeros_like(self.prefix_interval)
        self.birth_phase = [-1]*self.num_envs
        self.rollover_total_completed = self.rollover_zero_prefix_completed = 0
        self.native_capture_steps = 0
        self.prefix_release_count = 0
        self.prefix_release_integrity = True
        self.prefix_release_receipts = []
        geometry = self.prone_encoder_geometry
        relative = []
        for side, sign in ((0, 1.), (1, -1.)):
            beta = getattr(geometry, f"side{side}_passive_inner_deg")
            index = (beta-110.).abs().argmin()
            relative.append(sign*getattr(geometry, f"side{side}_delta_rad")[index])
        self.relative_110 = torch.stack(relative)
        self.rollover_ready = True

    def course_reset_cases(self, cases, ids):
        self._initialize_rollover()
        if self.evaluation_cases is not None:
            if any(self.poses[cases[i]["pose_index"]].get("recovery", {}).get("family") != "roof_down" for i in ids):
                raise ValueError("Rollover evaluation only accepts inverted births")
            return cases
        cases = list(cases)
        # Stratified class selection, independent bank angles within each cell.
        classes = self.rollover_config["leg_classes"]
        for lane in ids:
            cls = classes[(lane+self.prone_reset_generation) % 4]
            pool = [i for i in self.roof_train if self.poses[i]["recovery"]["leg_class"] == cls]
            pose = self.prefix_course.rng.choice(pool)
            duration, phase = self.prefix_course.sample(lane)
            cases[lane] = dict(role="prone", recovery=True, pose_index=pose, pitch_deg=0.,
                birth_height=self.poses[pose]["root_position_m"][2], target=[0.,0.,.2], command=[0.,0.,.2],
                rollover_prefix_s=duration, rollover_birth_phase=phase)
        return cases

    @torch.inference_mode()
    def reset(self, ids):
        if not getattr(self, "course_ready", False):
            return super().reset(ids)
        # Initialize before parent reset calls the task potential.
        self._initialize_rollover()
        super().reset(ids)
        rows = [self.course_cases[i] for i in ids.tolist()]
        duration = [r.get("rollover_prefix_s", 0.) if self.evaluation_cases is None else 0. for r in rows]
        self.prefix_remaining[ids] = self.prefix_initial[ids] = self.prefix_remaining.new_tensor(duration)
        self.prefix_interval[ids] = self.prefix_release_event[ids] = False
        self.rollover_enabled_at[ids] = float("nan")
        self.native_streak[ids] = 0
        self.native_condition[ids] = self.ended_recorded[ids] = False
        for lane, row in zip(ids.tolist(), rows):
            self.birth_phase[lane] = row.get("rollover_birth_phase", -1)
        seconds = self.rollover_config["episode_active_s"]+max(duration, default=0.)+.2
        if self.evaluation_cases is not None:
            seconds = max(r.get("evaluation_seconds", 12.2) for r in rows)
        self.episode_limits[ids] = round(seconds/self.policy_dt)
        if self._rollover_hook:
            self.whole_impact.reset(ids)
            self.rollover_contact[ids] = 0.
        self.previous_potential[ids] = self.potential()[ids]

    def get_observations(self):
        observation = super().get_observations()
        if self.rollover_ready:
            self.rollover_sensor_gravity = observation["policy"][:, 7:10].clone()
        return observation

    @torch.inference_mode()
    def step(self, actions):
        self.prefix_interval.copy_((self.prefix_remaining > self.policy_dt*.1) & ~self.passive & ~self.finished)
        self.prefix_release_event.zero_()
        applied = actions.clone()
        if bool(self.prefix_interval.any()):
            measured = self.getup_feedback.q[:, list(self.v5.LEGS)]
            geometry = self.prone_encoder_geometry(measured, self.getup_feedback.dq[:, list(self.v5.LEGS)])
            vector = geometry["wheel_body_m"]-self.recovery_hips[None]
            orbits = torch.atan2(vector[...,0], -vector[...,2])
            raw = task.prefix_action(measured, self.rollover_sensor_gravity, self.relative_110,
                                     self.rollover_config, policy_dt=self.policy_dt, wheel_orbits=orbits)
            applied[self.prefix_interval, :4] = raw[self.prefix_interval]
            applied[self.prefix_interval, 4:] = 0.
        prefix = self.prefix_interval.clone()
        if self._rollover_hook:
            self.whole_impact.begin_interval()
        observation, reward, done, info = super().step(applied)
        # The complete interval must be actor-controlled. The native adapter
        # also segments GAE and excludes prefixes from normalizers/losses.
        info["policy_learning_mask"] &= ~prefix
        info["policy_bootstrap_mask"] &= self.prefix_remaining <= self.policy_dt*.1
        info["diagnostics"]["rollover_prefix_interval"] = prefix
        info["diagnostics"]["rollover_learning_mask"] = info["policy_learning_mask"].clone()
        return observation, reward, done, info

    def _observe_rollover_physics(self, dt):
        if dt != self.dt:
            raise RuntimeError("Rollover hold clock differs from completed native physics")
        self.native_capture_steps += 1
        import warp as wp
        for ids, view in self.contact_views:
            force = wp.to_torch(view.get_contact_force_matrix(dt=dt))
            self.rollover_contact[ids] = force.reshape(len(ids), self.body_count, -1, 3).sum(2)
        weight = self.body_mass.sum(-1)*9.81
        data = self.robot.data
        motor_dq = data.joint_vel.torch[:, self.ids][:, POLICY_FROM_CONTROL]
        good = task.prone_condition(data.projected_gravity_b.torch,
            data.root_link_pose_w.torch[:, 2]-self.origins[:, 2], data.root_com_vel_w.torch[:, :3],
            data.root_com_ang_vel_b.torch, motor_dq,
            self.rollover_contact[..., 2].clamp_min(0.).sum(-1), weight, self.rollover_config)
        good &= ~(self.passive | self.finished | self.prefix_interval)
        self.native_condition.copy_(good)
        self.native_streak.copy_(torch.where(good, self.native_streak+1, 0))
        self.whole_impact.observe(self.rollover_contact, ~self.finished)

    def task_stable(self, *unused):
        return self.native_streak >= round(self.rollover_config["capture_hold_s"]/self.dt)

    def task_commands(self, elapsed):
        self._refresh_role_tensors()
        return self.course_targets.new_tensor([0., 0., .2]).expand(self.num_envs, -1).clone()

    def _scores(self, state=None, route=None):
        if state is None:
            velocity, omega, gravity, height, q, dq, local = self.state()
        else:
            gravity, height, omega, dq = (state[k] for k in ("gravity", "height", "omega", "dq"))
        route = self.route_state() if route is None else route
        q = self.robot.data.joint_pos.torch[:, self.ids][:, list(self.v5.LEGS)]
        common, relative, rates = task.leg_coordinates(q, dq[:, list(self.v5.LEGS)])
        vector = route["encoder_wheel_body_m"]-self.recovery_hips[None]
        orbits = torch.atan2(vector[...,0], -vector[...,2])
        return task.scores(gravity, height, self.robot.data.root_com_vel_w.torch[:, :3], omega,
                           route["knee_deg"], orbits, rates, route["encoder_valid"].all(-1))

    def potential(self):
        if not self.rollover_ready:
            return super().potential()
        return task.potential(self._scores())

    def task_transition(self, state, terminated):
        route = self.route_state()
        self.rollover_enabled_at[state["enable"]] = state["elapsed"][state["enable"]]
        active_age = torch.nan_to_num(state["elapsed"]-self.rollover_enabled_at, nan=0.)
        bad = dict(part_impact=self.recovery_impact.violation.clone(),
                   rollover_timeout=(~self.passive & (active_age >=
                       self.rollover_config["episode_active_s"]+self.prefix_initial) & ~self.ever_recovered))
        # Deadline is an episode timeout, not a mechanical failure. Original
        # mechanical/impact/boundary guards still settle hard faults unchanged.
        terminated |= bad["part_impact"]
        rejected = state["first"] & terminated
        state["first"] &= ~terminated
        self.ever_recovered[rejected] = False
        self.recovered_at[rejected] = float("nan")
        first = state["first"].clone()
        s = self._scores(state, route)
        dq = state["dq"][:, POLICY_FROM_CONTROL]
        terms = task.densities(s, state["omega"], state["torque"], dq, state["action_delta"],
            self.whole_impact.impulse_peak, self.body_mass.sum(-1)*9.81, self.rollover_config)
        now = task.potential(s)
        shaping = self.getup_cfg["reward_gamma"]*torch.where(terminated, 0., now)-self.previous_potential
        self.previous_potential.copy_(now)
        # The base path pays3 once; the separate skill declares a10-point bonus.
        shaping += (self.rollover_config["reward"]["completion_bonus"]-3.)*first
        training = self.recovery_training_run and self.evaluation_cases is None
        deadline = bad["rollover_timeout"]
        if training:
            ending = terminated | deadline | first
            for lane in (ending & ~self.ended_recorded).nonzero().flatten().tolist():
                self.prefix_course.observe(self.birth_phase[lane], bool(first[lane]))
            self.ended_recorded |= ending
            self.episode_limits.copy_(torch.where(deadline | first, self.episode_length_buf, self.episode_limits))
        self.rollover_total_completed += int(first.sum())
        self.rollover_zero_prefix_completed += int((first & (self.prefix_initial == 0.)).sum())
        self.prefix_remaining.copy_((self.prefix_remaining-self.prefix_interval*self.policy_dt).clamp_min(0.))
        release = self.prefix_interval & (self.prefix_remaining < self.policy_dt*.1)
        self.prefix_release_event.copy_(release)
        # Only policy action coordinates reset. Never touch q/dq, torques,
        # command delay, feedback, gimbal or actuator-filter state here.
        if bool(release.any()):
            self._release_prefix(release)
        return sum(terms.values()), shaping, terminated, dict(task_reward_terms=terms, task_failures=bad,
            prone_route=route, rollover_capture_condition=self.native_condition.clone(),
            rollover_native_hold_s=self.native_streak.float()*self.dt,
            rollover_completed_event=first, rollover_enabled_at=self.rollover_enabled_at.clone(),
            rollover_active_seconds=active_age, rollover_prefix_initial_s=self.prefix_initial.clone(),
            rollover_prefix_release_event=release.clone(),
            rollover_whole_impulse_peak_ns=self.whole_impact.episode_impulse_peak.clone(),
            rollover_whole_force_peak_n=self.whole_impact.episode_force_peak.clone(),
            recovery_episode_force_peak_n=self.recovery_impact.episode_force_peak.clone(),
            recovery_episode_impulse_peak_ns=self.recovery_impact.episode_impulse_peak.clone(),
            recovery_part_impact_bad=bad["part_impact"],
            rollover_completion_is_not_standing=torch.ones_like(first))

    def _release_prefix(self, release):
        """Audit the exact transition, with independent actor action history."""
        objects = (self.robot, self.sim, self.v5, self.actuator_response, self.identified_wheels,
                   self.getup_feedback, self.domain, self.gimbal)
        histories = dict(q=self.robot.data.joint_pos.torch, dq=self.robot.data.joint_vel.torch,
            root=self.robot.data.root_link_pose_w.torch, velocity=self.robot.data.root_com_vel_w.torch,
            motor_filter=self.actuator_response.state, wheel_delay=self.identified_wheels.history,
            encoder_q=self.getup_feedback.q_history, encoder_dq=self.getup_feedback.dq_history,
            domain_delay=self.domain.torque_history, domain_filter=self.domain.torque_state,
            sensor_history=self.domain.sensor_history, gimbal_target=self.gimbal.target,
            gimbal_phase=self.gimbal.phase, quiet_target=self.quiet_standing.target_slow)
        before = {k:v.clone() for k,v in histories.items()}
        clocks = (self.tick, self.getup_feedback.tick, self.getup_feedback.cursor,
                  self.identified_wheels.head, self.domain.torque_cursor)
        for history in (self.actions, self.previous_actions, self.before_previous_actions):
            history[release] = 0.
        equality = {k:bool(torch.equal(before[k], v)) for k,v in histories.items()}
        same = all(equality.values()) and clocks == (self.tick, self.getup_feedback.tick,
            self.getup_feedback.cursor, self.identified_wheels.head, self.domain.torque_cursor)
        same &= objects == (self.robot, self.sim, self.v5, self.actuator_response, self.identified_wheels,
                           self.getup_feedback, self.domain, self.gimbal)
        self.prefix_release_integrity &= same
        self.prefix_release_count += int(release.sum())
        if len(self.prefix_release_receipts) < 16:
            self.prefix_release_receipts.append(dict(policy_tick=self.tick, lanes=release.nonzero().flatten().tolist(),
                equal_state_and_history=equality, same_clocks_and_objects=same,
                nonzero_wheel_delay=bool(before["wheel_delay"].abs().any()),
                nonzero_motor_filter=bool(before["motor_filter"].abs().any())))
        if not same:
            raise RuntimeError("Script-to-actor release mutated physical state or controller history")

    def task_metrics(self, state, extra, reward, done, terminated):
        values = {"Rollover/completed_events": extra["rollover_completed_event"].float().sum(),
            "Rollover/zero_prefix_completed_events": (extra["rollover_completed_event"] &
                (extra["rollover_prefix_initial_s"] == 0.)).float().sum(),
            "Rollover/episode_events": done.float().sum(),
            "Rollover/prefix_release_events": extra["rollover_prefix_release_event"].float().sum(),
            "Rollover/prefix_phase": reward.new_tensor(self.prefix_course.phase),
            "Rollover/current_phase_window_success_rate": reward.new_tensor(
                sum(self.prefix_course.window)/max(1,len(self.prefix_course.window))),
            "Rollover/native_hold_peak_s": extra["rollover_native_hold_s"].max(),
            "Rollover/body_tilt_mean_deg": torch.rad2deg(torch.acos((-state["gravity"][:,2]).clamp(-1.,1.))).mean(),
            "Rollover/COM_speed_mean_m_s": self.robot.data.root_com_vel_w.torch[:, :3].norm(dim=-1).mean()}
        for name, value in extra["task_reward_terms"].items():
            values["RolloverReward/"+name+"_per_second"] = value.mean()
        for i, name in enumerate(("chassis", "gimbal", "legs", "wheels")):
            values["Rollover/whole_impulse/"+name+"_Ns"] = extra["rollover_whole_impulse_peak_ns"][:, i].max()
        values.update(self.encoder_increment_actions.metrics())
        return values

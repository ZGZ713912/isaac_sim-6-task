"""Mixed reset distribution and event-driven requests, without motor takeover."""
from __future__ import annotations

import json
import math
import torch

from .robust_getup_env import RobustGetupEnv
from .standard_prone_env import StandardProneEnv
from .root_kinematics import update_robot_data
from .standard_prone import route_potential, valid_disabled_start
from . import prone_motion as task


class ProneMotionEnv(RobustGetupEnv):
    route_state = StandardProneEnv.route_state

    def __init__(self, *args, all_heights_after=None, **kwargs):
        self.course_ready = False
        self.evaluation_cases = None
        if kwargs["settings"]["prone_motion_task"].get("full_vehicle") and kwargs.get("fixed_cases") is None:
            # The generic recovery constructor performs a reset before this
            # course assigns its explicit roles. Use a valid local seed pose;
            # it must not sample unsupported front/back families from the old
            # chassis bank. No physics/action occurs in this bootstrap reset.
            bank = json.loads(kwargs["bank"].read_text())
            middle = bank["diagnostic_height_starts"]["mapping"]["stand"]
            kwargs["fixed_cases"] = [dict(pose_index=middle, recovery=False,
                command=[0., 0., .305]) for _ in range(kwargs["num_envs"])]
        super().__init__(*args, **kwargs)
        self.plan = self.getup_cfg["prone_motion_task"]
        self.prone_task = self.getup_cfg["prone_geometry"]
        self.multi_pose_cfg = self.plan.get("multi_pose_recovery")
        self.recovery_stages = self.recovery_impact = None
        from .training_wheel_response import install as install_wheels
        install_wheels(self)
        if self.plan.get("gas_spring_backend") == "dev_compensation_v2":
            if self.v5.gas_compensation is None:
                raise ValueError("Unified task is missing the dev gas-spring model")
            self.v5.gas_compensation.install(self)
        else:
            if self.v5.gas_compensation is not None:
                raise ValueError("Two gas-spring compensation implementations would be active")
            from .training_gas_spring import TrainingGasSpring
            self.identified_spring = TrainingGasSpring(self.cfg["v5_control"]["identified_gas_spring"], self.device)
            self.identified_spring.install(self)
            from .encoder_spring_feedforward import EncoderGasSpringFeedforward
            EncoderGasSpringFeedforward(self.cfg["v5_control"]["encoder_spring_feedforward"]).install(self)
        if self.cfg.get("full_vehicle"):
            from .full_vehicle_training import encoder_geometry
            self.prone_encoder_geometry = encoder_geometry(self.cfg, self.asset_root).to(self.device)
        else:
            from wheeled_algo.encoder_leg_geometry import EncoderLegGeometry
            from .standing_geometry107 import table_path
            self.prone_encoder_geometry = EncoderLegGeometry(json.loads(table_path().read_text()),
                relative_margin_rad=.001).to(self.device)
        self.prone_index = next(i for i, p in enumerate(self.poses) if p["name"] == self.prone_task["reset_pose_name"])
        mapping = self.bank["diagnostic_height_starts"]["mapping"]
        self.height_pose = {h: mapping[n] for h, n in ((.23, "stand_low"), (.305, "stand"), (.42, "stand_high"))}
        self.prone_start_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.course_anchor = self.prone_start_xy.clone()
        self.quiet_hold = torch.zeros(self.num_envs, device=self.device)
        self.quiet_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.course_targets = torch.zeros(self.num_envs, 3, device=self.device)
        self.course_roles = [self.plan["roles_per20"][i % 20] for i in range(self.num_envs)]
        self._role_signature = None
        self.prone_reset_generation = 0
        if self.multi_pose_cfg is not None:
            from .multi_pose_recovery import ResetSampler, RecoveryStages, validate_bank
            from .recovery_impact import RecoveryImpact
            # The authenticated task owns asset_root/cfg; no external geometry is admitted.
            spec = json.loads((self.asset_root/self.cfg["asset_directory"]/"model_spec.json").read_text())
            from .full_robot_v202 import digest
            validate_bank(self.bank, self.multi_pose_cfg, self.cfg["asset_manifest_sha256"],
                          digest(self.asset_root/self.cfg["asset_directory"]/"model_spec.json"))
            self.recovery_sampler = ResetSampler(self.bank, self.multi_pose_cfg, kwargs["seed"])
            self.recovery_stages = RecoveryStages(self.num_envs, self.device, self.multi_pose_cfg, self.policy_dt)
            self.recovery_impact = RecoveryImpact(self.num_envs, self.device, self.robot.body_names,
                self.plan["recovery_impact"], self.dt, self.body_mass.sum(-1))
            joints = {j["name"]: j for j in spec["joints"]}
            self.recovery_hips = self.prone_start_xy.new_tensor(
                [joints[s+"_joint1"]["origin"][i][3] for s in ("L", "R") for i in range(3)]).reshape(2,3)
        self.course_ready = True
        self.reset(torch.arange(self.num_envs, device=self.device))
        from .grounded_standing_reset import install
        install(self, include_motion=True)
        backend = self.cfg.get("prone_motion_performance", "eager")
        if backend != "eager":
            from .prone_motion_performance import install as install_performance
            install_performance(self,backend)
        self.startup_report["prone_motion"] = dict(
            motor_owner="Trainable six-axis RL for every active physical interval",
            state_writes="episode reset only", fallback_script=False,
            actor_dim=35, critic_dim=self.cfg["critic_dim"], wheel_target_units="rad/s, output shaft",
            reset="Normally-up belly-grounded103deg pose, zero torque until settled; initialized standing rehearsal",
            route_bias="No folded target initialization; weak encoder/COM potential only",
            full_commands=dict(forward_m_s=self.plan["forward_speed_m_s"],
                reverse_m_s=-self.plan["forward_speed_m_s"], spin_rpm=[-100., 100.],
                arc_forward_m_s=[-self.plan["forward_speed_m_s"], self.plan["forward_speed_m_s"]],
                arc_yaw_rad_s=[-.6, .6]))
        if self.multi_pose_cfg is not None:
            self.startup_report["multi_pose_recovery"] = dict(self.multi_pose_cfg,
                initial_state_count=sum("recovery" in p for p in self.poses),
                suffix="same actor; measured dynamic state preserved at flip-to-rise entry",
                actor_inputs_added=False, static_equilibrium_qualified=False,
                reset_admission=self.bank["multi_pose_recovery"], impact=self.plan["recovery_impact"])
            self.startup_report["prone_motion"]["reset"] = "CAD-screened full-vehicle body-family x four independent fore/aft leg classes"
            self.startup_report["prone_motion"]["route_bias"] = "No fixed fold angle; shared upright-recoverable suffix; bounded progress potential"

    @torch.inference_mode()
    def reset(self, ids):
        if not self.course_ready:
            return super().reset(ids)
        cases = []
        reset_ids = set(ids.tolist())
        for i, role in enumerate(self.course_roles):
            # Standing births eventually cover all heights. Learning motion and
            # height transitions still start at305mm, so height error cannot
            # terminate the experiment before it receives an action.
            h = (.23, .305, .42)[i % 3] if role in ("height", "stand") else .305
            birth = h if role == "stand" and self.actor_updates >= self.plan["height_full_update"] else .305
            cases.append(dict(pose_index=self.prone_index if role == "prone" else self.height_pose[birth],
                recovery=role == "prone", role=role, target=task.target(role, self.actor_updates, self.plan, height=h),
                command=[0., 0., birth], pitch_deg=0., birth_height=birth))
            if self.multi_pose_cfg is not None and role == "prone" and self.evaluation_cases is None:
                selected = (self.recovery_sampler.sample(self.actor_updates) if i in reset_ids
                            else self.course_cases[i]["pose_index"])
                cases[-1].update(pose_index=selected, birth_height=self.poses[selected]["root_position_m"][2])
        if self.evaluation_cases is not None:
            cases = self.evaluation_cases
        self.course_cases = cases
        self.fixed_cases = cases
        try:
            super().reset(ids)
        finally:
            self.fixed_cases = cases if self.evaluation_cases is not None else None
        for i in ids.tolist():
            case = cases[i]
            self.course_roles[i] = case["role"]
            self.course_targets[i] = self.course_targets.new_tensor(case["target"])
            duration = (self.plan["evaluation"]["seconds"] if self.evaluation_cases is not None else
                self.plan["training_prone_seconds"] if case["recovery"] else
                self.plan["standing_seconds"] if case["role"] == "stand" else self.plan["motion_seconds"])
            self.episode_limits[i] = round(duration/self.policy_dt)
        root = self.robot.data.root_link_pose_w.torch[ids].clone()
        prone = self.recovery_lane[ids]
        if self.multi_pose_cfg is None:
            root[prone, 2] -= (self.poses[self.prone_index]["min_collision_clearance_m"]
                              - self.prone_task["reset_collision_clearance_m"])
        angle = root.new_tensor([cases[i].get("pitch_deg", 0.) for i in ids.tolist()])*math.pi/180.
        x, y, z, w = root[:, 3:].unbind(-1)
        s, c = (angle/2.).sin(), (angle/2.).cos()
        root[:, 3:] = torch.stack((x*c-z*s, y*c+w*s, z*c+x*s, w*c-y*s), -1)
        if self.multi_pose_cfg is not None:
            heading = (self.random(len(ids))*2.-1.)*math.pi if self.evaluation_cases is None else root.new_tensor(
                [cases[i].get("heading_rad", 0.) for i in ids.tolist()])
            x,y,z,w = root[:, 3:].unbind(-1)
            s,c = (heading/2.).sin(), (heading/2.).cos()
            root[:, 3:] = torch.stack((c*x-s*y,s*x+c*y,c*z+s*w,c*w-s*z), -1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=ids)
        if self.multi_pose_cfg is not None:
            velocity = root.new_tensor([[*self.poses[cases[i]["pose_index"]].get("root_linear_velocity_body_m_s", [0.,0.,0.]),
                                         *self.poses[cases[i]["pose_index"]].get("root_angular_velocity_body_rad_s", [0.,0.,0.])]
                                         for i in ids.tolist()]).reshape(len(ids),2,3)
            xyz, w = root[:,None,3:6].expand_as(velocity), root[:,None,6:7]
            cross = torch.linalg.cross(xyz,velocity)
            velocity = velocity+2.*(w*cross+torch.linalg.cross(xyz,cross))
            self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity.reshape(len(ids),6), env_ids=ids)
            from .multi_pose_recovery import FAMILIES, LEG_CLASSES
            info = [self.poses[cases[i]["pose_index"]].get("recovery", {}) for i in ids.tolist()]
            family = torch.tensor([FAMILIES.index(r.get("family", "belly_down")) for r in info], device=self.device)
            leg = torch.tensor([LEG_CLASSES.index(r.get("leg_class", "rear_rear")) for r in info], device=self.device)
            self.recovery_stages.reset(ids, family, leg)
            self.recovery_impact.reset(ids)
        update_robot_data(self, self.dt)
        self.prone_start_xy[ids] = root[:, :2]
        self.prone_reset_generation += 1
        self.course_anchor[ids] = root[:, :2]
        self.quiet_hold[ids] = 0.
        self.quiet_completed[ids] = False
        self.previous_potential[ids] = self.potential()[ids]
        self.pending_command[ids] = self.task_commands(self.episode_length_buf*self.policy_dt)[ids]
        self.goal_command[ids] = self.pending_command[ids]
        self.commands[ids] = self.pending_command[ids]
        self.raw_height[ids] = self.commands[ids, 2]

    def sample_commands(self, ids):
        if not self.course_ready:
            return super().sample_commands(ids)

    def get_observations(self):
        result = super().get_observations()
        if self.course_ready:
            xy = self.robot.data.root_link_pose_w.torch[:, :2]
            result["critic"][:, -2:] = (20.*(xy-self.course_anchor)).clamp(-5., 5.)
        return result

    def potential(self):
        if not self.course_ready:
            return super().potential()
        _, _, gravity, height, _, _, _ = self.state()
        route = self.route_state()
        if self.multi_pose_cfg is not None:
            from .multi_pose_recovery import potential
            value = potential(gravity,height,route["support_error_m"])
            return torch.where(self.recovery_lane & ~self.ever_recovered, value, 8.*torch.ones_like(value))
        # Frozen after completion: no incentive to refold a standing leg.
        value, _ = route_potential(gravity, route["knee_deg"], route["support_error_m"], height,
            self.prone_task, geometry_valid=route["encoder_valid"].all(-1))
        return torch.where(self.recovery_lane & ~self.ever_recovered, value, 15.*torch.ones_like(value))

    def task_stable(self, *args):
        return task.stable(*args)

    def task_commands(self, elapsed):
        value = task.commands(self.course_targets, elapsed, self.recovered_at, self.recovery_lane, self.plan)
        self._refresh_role_tensors()
        stand = self._stand_roles
        value[stand] = self.course_targets[stand]
        return value

    def _refresh_role_tensors(self):
        signature = tuple(self.course_roles)
        if signature != self._role_signature:
            self._role_index = torch.tensor([task.ROLES.index(r) for r in signature],device=self.device)
            self._stand_roles = self._role_index == task.ROLES.index("stand")
            self._role_signature = signature

    def task_transition(self, state, terminated):
        route = self.route_state()
        rising = self.recovery_lane & ~self.ever_recovered
        # Re-anchor only on an actual completion, or a request returning to zero.
        # No physical pose/velocity or motor target is written here.
        next_quiet = (self.pending_command[:, :2].abs() < .01).all(-1)
        was_quiet = (self.commands[:, :2].abs() < .01).all(-1)
        anchor = state["first"] | (next_quiet & ~was_quiet & (self.commands[:, 0].abs() >= .01))
        xy = self.robot.data.root_link_pose_w.torch[:, :2]
        self.course_anchor[anchor] = xy[anchor]
        error = (xy-self.course_anchor).norm(dim=-1)
        terms = task.reward_terms(state["gravity"], state["height"], state["velocity"], state["omega"],
            self.commands, state["torque"], state["dq"], task.normalized_action_delta(state["action_delta"], self.plan),
            route["xy_excursion_m"], error, rising, state["dual_support"],
            position_tail=self.plan.get("quiet_position_tail"))
        terms["leg_speed"] = -.002*state["leg_speed_square"]
        stage_extra = {}
        if self.multi_pose_cfg is not None:
            from .multi_pose_recovery import recovery_terms
            new_terms = recovery_terms(state,self.multi_pose_cfg,self.recovery_impact.cost_integral/self.policy_dt)
            terms = {key:torch.where(rising, new_terms.get(key,torch.zeros_like(value)),value)
                     for key,value in terms.items()}
            for key,value in new_terms.items():
                if key not in terms:
                    terms[key] = torch.where(rising,value,torch.zeros_like(value))
        bad = task.failures(state["gravity"], state["height"], state["nonwheel_force"],
            route["xy_excursion_m"], state["elapsed"], rising, ~self.recovery_lane, self.plan)
        if self.multi_pose_cfg is None:
            bad["invalid_prone_start"] = state["enable"] & self.recovery_lane & ~valid_disabled_start(
                state["gravity"], state["height"], state["belly_force"], self.prone_task)
        else:
            bad["part_impact"] = self.recovery_impact.violation.clone()
            # Bank admission owns initial feasibility, not belly-only support.
            bad["invalid_prone_start"] = torch.zeros_like(terminated)
            newly_failed = terminated | bad["part_impact"] | bad["refall"] | bad["rise_excursion"]
            q = self.robot.data.joint_pos.torch[:,self.ids][:,list(self.v5.LEGS)]
            dq = state["dq"][:,list(self.v5.LEGS)]
            vector = route["encoder_wheel_body_m"]-self.recovery_hips[None]
            orbits = torch.atan2(vector[...,0],-vector[...,2])
            entry, active_seconds = self.recovery_stages.update(
                dict(state,passive=self.passive | state["enable"]), eligible=self.recovery_lane,
                gravity=state["gravity"],omega=state["omega"],q=q,dq=dq,orbits=orbits,
                geometry_valid=route["encoder_valid"],
                encoder_error=route["encoder_vs_native_wheel_error_m"].amax(-1), failed=newly_failed)
            bad["rise_timeout"] = rising & (active_seconds >= self.plan["rise_deadline_seconds"])
            flip = entry & (self.recovery_stages.initial_family != 0)
            stage_extra = dict(recovery_suffix_entry_event=entry.clone(), recovery_flip_event=flip.clone(),
                recovery_enabled_at=self.recovery_stages.enabled_at.clone(),
                recovery_upright_at=self.recovery_stages.upright_at.clone(), recovery_active_seconds=active_seconds,
                recovery_initial_family=self.recovery_stages.initial_family.clone(),
                recovery_initial_leg_class=self.recovery_stages.initial_leg_class.clone(),
                recovery_entry_leg_class=self.recovery_stages.entry_leg_class.clone(),
                recovery_entry_q=self.recovery_stages.entry_q.clone(),
                recovery_entry_dq=self.recovery_stages.entry_dq.clone(),
                recovery_entry_omega=self.recovery_stages.entry_omega.clone(),
                recovery_force_peak_n=self.recovery_impact.force_peak.clone(),
                recovery_impulse_peak_ns=self.recovery_impact.impulse_peak.clone(),
                recovery_episode_force_peak_n=self.recovery_impact.episode_force_peak.clone(),
                recovery_episode_impulse_peak_ns=self.recovery_impact.episode_impulse_peak.clone(),
                recovery_part_impact_bad=bad["part_impact"].clone())
        for value in bad.values():
            terminated |= value
        if self.multi_pose_cfg is not None:
            rejected_completion = state["first"] & terminated
            state["first"] &= ~terminated
            self.ever_recovered[rejected_completion] = False
            self.recovered_at[rejected_completion] = float("nan")
        holding = self.ever_recovered & task.stable(state["gravity"], state["height"], state["velocity"],
            state["omega"], state["dq"], self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2., state["nonwheel_force"])
        self.quiet_hold = torch.where(holding & ~state["first"], self.quiet_hold+self.policy_dt, 0.)
        quiet_event = (self.quiet_hold >= 30.) & ~self.quiet_completed
        self.quiet_completed |= quiet_event
        now = self.potential()
        gain = self.plan["route_potential_gain"] if self.multi_pose_cfg is None else self.multi_pose_cfg["potential_gain"]
        shaping = gain*(self.getup_cfg["reward_gamma"]*
            torch.where(terminated, 0., now)-self.previous_potential)
        # Potential guidance exists only in original prone episodes, never in
        # initialized motion/height roles. Time and dense standing dominate.
        shaping *= self.recovery_lane
        if self.multi_pose_cfg is not None:
            shaping += self.multi_pose_cfg["flip_event_bonus"]*stage_extra["recovery_flip_event"]*(~terminated)
        self.previous_potential.copy_(now)
        return sum(terms.values()), shaping, terminated, dict(
            task_reward_terms=terms, task_failures=bad, prone_route=route,
            course_anchor_error_m=error.clone(), quiet_hold_seconds=self.quiet_hold.clone(),
            quiet30_completed_event=quiet_event.clone(), role_index=self._role_index.clone(),
            root_quaternion_xyzw=self.robot.data.root_link_pose_w.torch[:, 3:].clone(),
            task_recovery_rising=rising.clone(), **stage_extra)

    def task_metrics(self, state, extra, reward, done, terminated):
        result = {"ProneMotion/updates": reward.new_tensor(self.actor_updates),
            "ProneMotion/recovery_completed_events": state["first"].float().sum(),
            "ProneMotion/quiet30_completed_events": extra["quiet30_completed_event"].float().sum(),
            "ProneMotion/terminated_events": terminated.float().sum(),
            "ProneMotion/timeout_events": (done & ~terminated).float().sum(),
            "ProneMotion/wheel_speed_peak_rad_s": state["dq"][:, [2, 5]].abs().max()}
        if self.multi_pose_cfg is not None:
            from .multi_pose_recovery import FAMILIES, LEG_CLASSES
            from .recovery_impact import GROUPS
            result.update({"Recovery/flip_events_per_step":extra["recovery_flip_event"].float().sum(),
                "Recovery/suffix_entry_events_per_step":extra["recovery_suffix_entry_event"].float().sum(),
                "Recovery/part_impact_failures_per_step":extra["recovery_part_impact_bad"].float().sum()})
            for g,group in enumerate(GROUPS):
                result[f"Recovery/impact/{group}/force_peak_n"] = extra["recovery_force_peak_n"][:,g].max()
                result[f"Recovery/impact/{group}/window_impulse_peak_ns"] = extra["recovery_impulse_peak_ns"][:,g].max()
            for f,family in enumerate(FAMILIES):
                for l,cls in enumerate(LEG_CLASSES):
                    mask = (self.recovery_lane & (extra["recovery_initial_family"] == f)
                            & (extra["recovery_initial_leg_class"] == l))
                    prefix = f"Recovery/{family}/{cls}/"
                    first = state["first"] & mask
                    result[prefix+"enabled_events_per_step"] = (mask & state["enable"]).float().sum()
                    result[prefix+"completed_events_per_step"] = first.float().sum()
                    result[prefix+"failure_events_per_step"] = (mask & terminated).float().sum()
                    result[prefix+"active_completion_time_sum_s"] = torch.where(first,
                        extra["recovery_active_seconds"],0.).sum()
        physical = self.prone_motion_physical
        if self.leg_friction.enabled:
            loss, speed = physical["leg_friction_passive_effort"], physical["leg_friction_velocity"]
            result["LegFriction/dissipated_power_mean_w"] = -(loss*speed).sum(-1).mean()
            for index, name in enumerate(self.leg_friction.model["motor_names"]):
                result["LegFriction/"+name+"_passive_abs_mean_nm"] = loss[..., index].abs().mean()
        result.update({
            "ProneMotion/gas_spring/outside_identified_fraction": physical["spring_outside_identified"].float().mean(),
            "ProneMotion/gas_spring/compression_min_m": physical["spring_compression"].min(),
            "ProneMotion/gas_spring/compression_max_m": physical["spring_compression"].max(),
            "ProneMotion/gas_spring/force_left_mean_n": physical["spring_force"][:, :, 0].mean(),
            "ProneMotion/gas_spring/force_right_mean_n": physical["spring_force"][:, :, 1].mean()})
        prefix = "ProneMotion/gas_spring_feedforward/"
        result.update({
            prefix+"invalid_geometry_fraction": (~physical["spring_ff_geometry_valid"]).float().mean(),
            prefix+"outside_identified_fraction": physical["spring_ff_outside_identified"].float().mean(),
            prefix+"compression_error_abs_mean_m": (physical["spring_ff_compression_m"]-physical["spring_compression"]).abs().mean(),
            prefix+"total_request_saturated_fraction": (physical["spring_ff_requested_effort_nm"].abs() > 40.).float().mean()})
        for i, name in enumerate(("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")):
            result[prefix+name+"_holding_abs_mean_nm"] = physical["spring_ff_holding_effort_nm"][:, :, i].abs().mean()
            result[prefix+name+"_holding_peak_nm"] = physical["spring_ff_holding_effort_nm"][:, :, i].abs().max()
        for index, side in enumerate(("left", "right")):
            prefix = "ProneMotion/wheel_model/"+side+"/"
            result[prefix+"motor_effort_abs_mean_nm"] = physical["motor_effort"][:, :, (2, 5)[index]].abs().mean()
            result[prefix+"passive_residual_abs_mean_nm"] = physical["wheel_passive_effort"][:, :, index].abs().mean()
            result[prefix+"drive_bound_min_nm"] = physical["wheel_drive_bound"][:, :, index].min()
            result[prefix+"motoring_derated_fraction"] = (physical["wheel_drive_bound"][:, :, index] < 4.5-1e-6).float().mean()
        for name, value in extra["task_reward_terms"].items():
            result["ProneMotion/reward_density_"+name] = value.mean()
        masks = [(extra["role_index"] == index) & ~self.passive for index in range(len(task.ROLES))]
        # Preserve which tags exist and the exact selected-value reductions.
        # One transfer replaces eleven data-dependent CUDA synchronizations.
        present = torch.stack([mask.any() for mask in masks]).tolist()
        for role, mask, has_samples in zip(task.ROLES,masks,present):
            if has_samples:
                fields = dict(height_m=state["height"], vx_m_s=state["velocity"][:, 0],
                    yaw_rad_s=state["omega"][:, 2], commanded_vx_m_s=self.commands[:, 0],
                    commanded_yaw_rad_s=self.commands[:, 1], commanded_height_m=self.commands[:, 2],
                    tilt_deg=torch.rad2deg(torch.acos((-state["gravity"][:, 2]).clamp(-1., 1.))),
                    anchor_error_m=extra["course_anchor_error_m"], reward=reward)
                result.update({f"ProneMotion/{role}/{name}": value[mask].mean() for name, value in fields.items()})
                result[f"ProneMotion/{role}/spring_outside_identified_fraction"] = physical[
                    "spring_outside_identified"][:, mask].float().mean()
        return result

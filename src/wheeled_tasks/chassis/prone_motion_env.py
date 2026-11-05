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
        recovery_mastery = kwargs.pop("recovery_mastery",None)
        global_curriculum = kwargs.pop("global_curriculum",None)
        self.global_curriculum = None
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
        from .recovery_discovery import validate as validate_recovery_focus
        focus = self.plan.get("global_curriculum", {}).get("recovery_focus")
        self.recovery_training_focus = validate_recovery_focus(focus) if focus is not None else None
        self.recovery_training_run = kwargs.get("run_role") == "train"
        if "getup_translation" in self.plan:
            from .getup_translation import validate as validate_translation
            self.startup_report["getup_translation"] = validate_translation(self.plan["getup_translation"])
        self.cold_evaluation_reset_events = 0
        if self.plan.get("cold_evaluation_lifecycle") is not None:
            from .cold_evaluation_lifecycle import validate as validate_cold_lifecycle
            self.startup_report["cold_evaluation_lifecycle"] = validate_cold_lifecycle(
                self.plan["cold_evaluation_lifecycle"])
        from pxr import PhysxSchema
        solver_rows = []
        for prim in self.sim.stage.Traverse():
            if (str(prim.GetPath()).startswith("/World/envs/env_")
                    and prim.HasAPI(PhysxSchema.PhysxArticulationAPI)):
                api = PhysxSchema.PhysxArticulationAPI(prim)
                solver_rows.append([api.GetSolverPositionIterationCountAttr().Get(),
                                    api.GetSolverVelocityIterationCountAttr().Get()])
        solver_matches = len(solver_rows) == self.num_envs and all(
            row == [self.cfg["position_iterations"], self.cfg["velocity_iterations"]] for row in solver_rows)
        self.startup_report["solver_iterations"] = dict(
            source="live_stage_PhysxArticulationAPI_attributes", articulation_count=len(solver_rows),
            unique_iteration_counts=sorted({tuple(row) for row in solver_rows}),
            all_match_requested=solver_matches)
        if not solver_matches:
            raise RuntimeError("Native articulation solver attributes differ from the selected iteration counts")
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
        if self.plan.get("paired_torque_saturation") is not None:
            from .paired_torque_saturation import install as install_saturation
            install_saturation(self, self.plan["paired_torque_saturation"])
        self.encoder_increment_actions = None
        if self.plan.get("encoder_increment_actions") is not None:
            from .recovery_actions import EncoderIncrementActions
            self.encoder_increment_actions = EncoderIncrementActions(self, self.plan["encoder_increment_actions"])
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
        self.course_moving_height = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.course_abrupt_brake = self.course_moving_height.clone()
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
            self.recovery_sampler = ResetSampler(self.bank, self.multi_pose_cfg, kwargs["seed"],mastery=recovery_mastery)
            self.recovery_stages = RecoveryStages(self.num_envs, self.device, self.multi_pose_cfg, self.policy_dt)
            self.recovery_impact = RecoveryImpact(self.num_envs, self.device, self.robot.body_names,
                self.plan["recovery_impact"], self.dt, self.body_mass.sum(-1))
            joints = {j["name"]: j for j in spec["joints"]}
            self.recovery_hips = self.prone_start_xy.new_tensor(
                [joints[s+"_joint1"]["origin"][i][3] for s in ("L", "R") for i in range(3)]).reshape(2,3)
        if self.plan.get("global_curriculum") is not None:
            from .global_curriculum import GlobalCurriculum
            self.global_curriculum = global_curriculum or GlobalCurriculum(self.plan,self.recovery_sampler.mastery)
            self._append_urdf_nominal_pose()
        sampling_backend = self.cfg.get('prone_motion_sampling_backend', 'packed')
        if sampling_backend not in ('reference', 'packed'):
            raise ValueError('Unknown prone-motion sampling backend')
        from .prone_motion_sampling import ProneMotionSampling
        self.sampling = ProneMotionSampling(self) if sampling_backend == 'packed' else None
        self.startup_report['sampling_backend'] = dict(name=sampling_backend,
            physical_samples='complete, independent contiguous fields',
            control_reward_termination_rng_changed=False)
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
            if self.multi_pose_cfg.get("reset_scope") == "standard_prone_only":
                self.startup_report["prone_motion"]["reset"] = "Single standard normally-up belly-grounded103deg posture only; no side, inverted or independent-leg-class births"
        if self.global_curriculum is not None:
            self.startup_report["global_curriculum"] = dict(config=self.global_curriculum.config,
                nominal_pose_index=self.nominal_pose_index,initial_state=self.global_curriculum.state_dict(),
                admission="cold per-case/per-leg evaluation; elapsed updates never unlock commands")
        if self.multi_pose_cfg is not None and self.recovery_sampler.start_curriculum is not None:
            course = self.recovery_sampler.start_curriculum
            self.startup_report["recovery_start_curriculum"] = dict(config=course.config,
                initial_state=course.state_dict(),pools=course.pools,
                training_only=True,geometry_or_behavior_qualification=False)

    def _append_urdf_nominal_pose(self):
        """Authenticate literal nominal joints; only reset placement is grounded."""
        from copy import deepcopy
        manifest = json.loads((self.asset_root/self.cfg["asset_directory"]/"manifest.json").read_text())
        height = self.global_curriculum.config["nominal_height_m"]
        if manifest["nominal_base_height_m"] != height:
            raise ValueError("Global standing birth differs from the declared URDF nominal height")
        pose = deepcopy(self.poses[self.height_pose[.305]])
        pose.update(name="urdf_nominal",root_position_m=[0.,0.,height],
                    full_joint_positions=deepcopy(manifest["nominal_joint_pos"]))
        self.nominal_pose_index = len(self.poses)
        self.poses.append(pose)
        self.bank_joints = torch.cat((self.bank_joints,self.bank_joints.new_tensor(
            [[pose["full_joint_positions"][name] for name in self.robot.joint_names]])))
        self.bank_roots = torch.cat((self.bank_roots,self.bank_roots.new_tensor(
            [pose["root_position_m"]+pose["root_quaternion_xyzw"]])))
        self.bank_velocities = torch.cat((self.bank_velocities,torch.zeros_like(self.bank_velocities[:1])))
        self.bank_family = torch.cat((self.bank_family,self.bank_family[self.height_pose[.305]:self.height_pose[.305]+1]))
        self.bank_tilt = torch.cat((self.bank_tilt,torch.zeros_like(self.bank_tilt[:1])))

    @torch.inference_mode()
    def reset(self, ids):
        if not self.course_ready:
            return super().reset(ids)
        cases = []
        ids_cpu = ids.tolist()
        reset_ids = set(ids_cpu)
        for i, role in enumerate(self.course_roles):
            # Standing births eventually cover all heights. Learning motion and
            # height transitions still start at305mm, so height error cannot
            # terminate the experiment before it receives an action.
            h = (.23, .305, .42)[i % 3] if role in ("height", "stand") else .305
            birth = h if role == "stand" and self.actor_updates >= self.plan["height_full_update"] else .305
            cases.append(dict(pose_index=self.prone_index if role == "prone" else self.height_pose[birth],
                recovery=role == "prone", role=role, target=task.target(role, self.actor_updates, self.plan, height=h),
                command=[0., 0., birth], pitch_deg=0., birth_height=birth))
            if self.multi_pose_cfg is not None and self.global_curriculum is None and role == "prone" and self.evaluation_cases is None:
                selected = (self.recovery_sampler.sample(self.actor_updates) if i in reset_ids
                            else self.course_cases[i]["pose_index"])
                cases[-1].update(pose_index=selected, birth_height=self.poses[selected]["root_position_m"][2])
        if self.global_curriculum is not None and self.evaluation_cases is None:
            assignments = self.global_curriculum.training_descriptors(self.num_envs,self.prone_reset_generation)
            for i,row in enumerate(assignments):
                if i not in reset_ids:
                    cases[i] = self.course_cases[i]
                    continue
                birth = row["birth_height"]
                pose = (self.nominal_pose_index if row.get("nominal_birth") else self.height_pose.get(birth,self.height_pose[.305]))
                start_phase = -1
                if row["recovery"]:
                    from .recovery_start_curriculum import select_start
                    pose, start_phase = select_start(self.recovery_sampler, self.actor_updates,
                        training=self.recovery_training_run)
                cases[i] = dict(row,pose_index=pose,birth_height=self.poses[pose]["root_position_m"][2],
                    command=[0.,0.,birth], recovery_start_phase=start_phase)
        if self.evaluation_cases is not None:
            cases = self.evaluation_cases
        cases = self.course_reset_cases(cases, ids_cpu)
        self.course_cases = cases
        self.fixed_cases = cases
        try:
            if self.sampling is not None:
                super().reset(ids, fixed_case_rows=[cases[i] for i in ids_cpu])
            else:
                super().reset(ids)
        finally:
            self.fixed_cases = cases if self.evaluation_cases is not None else None
        for i in ids_cpu:
            case = cases[i]
            self.course_roles[i] = case["role"]
            if self.sampling is not None:
                continue
            self.course_targets[i] = self.course_targets.new_tensor(case["target"])
            self.course_moving_height[i] = case.get("moving_height",False)
            self.course_abrupt_brake[i] = case.get("abrupt_brake",False)
            duration = (case.get("evaluation_seconds",self.plan["evaluation"]["seconds"]) if self.evaluation_cases is not None else
                self.plan["training_prone_seconds"] if case["recovery"] else
                self.plan["standing_seconds"] if case["role"] == "stand" else self.plan["motion_seconds"])
            self.episode_limits[i] = round(duration/self.policy_dt)
        if self.sampling is not None:
            from .prone_motion_sampling import apply_course_cases
            apply_course_cases(self, ids, [cases[i] for i in ids_cpu])
        root = self.robot.data.root_link_pose_w.torch[ids].clone()
        prone = self.recovery_lane[ids]
        if self.multi_pose_cfg is None:
            root[prone, 2] -= (self.poses[self.prone_index]["min_collision_clearance_m"]
                              - self.prone_task["reset_collision_clearance_m"])
        angle = root.new_tensor([cases[i].get("pitch_deg", 0.) for i in ids_cpu])*math.pi/180.
        x, y, z, w = root[:, 3:].unbind(-1)
        s, c = (angle/2.).sin(), (angle/2.).cos()
        root[:, 3:] = torch.stack((x*c-z*s, y*c+w*s, z*c+x*s, w*c-y*s), -1)
        if self.multi_pose_cfg is not None:
            heading = (self.random(len(ids))*2.-1.)*math.pi if self.evaluation_cases is None else root.new_tensor(
                [cases[i].get("heading_rad", 0.) for i in ids_cpu])
            x,y,z,w = root[:, 3:].unbind(-1)
            s,c = (heading/2.).sin(), (heading/2.).cos()
            root[:, 3:] = torch.stack((c*x-s*y,s*x+c*y,c*z+s*w,c*w-s*z), -1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=ids)
        if self.multi_pose_cfg is not None:
            velocity = root.new_tensor([[*self.poses[cases[i]["pose_index"]].get("root_linear_velocity_body_m_s", [0.,0.,0.]),
                                         *self.poses[cases[i]["pose_index"]].get("root_angular_velocity_body_rad_s", [0.,0.,0.])]
                                         for i in ids_cpu]).reshape(len(ids),2,3)
            xyz, w = root[:,None,3:6].expand_as(velocity), root[:,None,6:7]
            cross = torch.linalg.cross(xyz,velocity)
            velocity = velocity+2.*(w*cross+torch.linalg.cross(xyz,cross))
            self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity.reshape(len(ids),6), env_ids=ids)
            from .multi_pose_recovery import FAMILIES, LEG_CLASSES
            info = [self.poses[cases[i]["pose_index"]].get("recovery", {}) for i in ids_cpu]
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
        from .standard_getup_reward import contact_guided
        if contact_guided((self.multi_pose_cfg or {}).get("dense_reward", {})):
            # Native contact forces belong to the previous physical pose until
            # the next simulation step. Never use them to shape a reset birth.
            self.contact_force[ids] = 0.
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
        _, omega, gravity, height, _, _, _ = self.state()
        route = self.route_state()
        if self.multi_pose_cfg is not None:
            from .standard_getup_reward import ordered, ordered_potential
            dense = self.multi_pose_cfg.get("dense_reward", {})
            if ordered(dense):
                state = self._recovery_reward_state(dict(gravity=gravity, height=height, omega=omega), route)
                value = ordered_potential(state, dense)
                maximum = dense["potential_capture_weight"]+dense["potential_lift_weight"]
                maximum += dense.get("potential_capture_stability_weight", 0.)
                return torch.where(self.recovery_lane & ~self.ever_recovered, value, maximum*torch.ones_like(value))
            from .multi_pose_recovery import potential
            value = potential(gravity,height,route["support_error_m"])
            return torch.where(self.recovery_lane & ~self.ever_recovered, value, 8.*torch.ones_like(value))
        # Frozen after completion: no incentive to refold a standing leg.
        value, _ = route_potential(gravity, route["knee_deg"], route["support_error_m"], height,
            self.prone_task, geometry_valid=route["encoder_valid"].all(-1))
        return torch.where(self.recovery_lane & ~self.ever_recovered, value, 15.*torch.ones_like(value))

    def _recovery_reward_state(self, state, route):
        from .standard_getup_reward import geometry_inputs, ordered, contact_guided, standing_guided
        result = dict(state, recovery_support_error_m=route["support_error_m"])
        dense = self.multi_pose_cfg.get("dense_reward", {})
        if ordered(dense):
            result.update(geometry_inputs(route, self.robot.data.root_link_pose_w.torch))
        phase_guided = dense.get("schema") == "wheel-leg-phase-recovery-reward-v1"
        if (contact_guided(dense) or phase_guided) and "recovery_wheel_load_fraction" not in result:
            from .standing_support import wheel_bottom_clearance
            poses = self.robot.data.body_link_pose_w.torch[:, self.wheel_ids]
            result["recovery_wheel_clearance_m"] = wheel_bottom_clearance(
                self.wheel_centers(), poses[:, :, 3:], self.origins[:, 2:3])
            result["recovery_wheel_load_fraction"] = self.contact_force[:, self.wheel_ids, 2].clamp_min(0.)/(
                9.81*self.body_mass.sum(-1, keepdim=True))
        if standing_guided(dense) or phase_guided:
            weight = 9.81*self.body_mass.sum(-1)
            if "nonwheel_force" in state:
                # Existing 200 Hz impact reduction; include both physical
                # samples rather than only the last contact snapshot.
                chassis = self.recovery_impact.force_peak[:, 0]
                nonwheel = state["nonwheel_force"]
                wheel_peak = self.recovery_impact.force_peak[:, 3]
            else:
                # Potential endpoints/reset use current contacts, never a
                # previous interval's peak. These loads do not gate lifting.
                chassis = self.contact_force[:, self.push_body_ids[0], 2].clamp_min(0.)
                nonwheel = self.contact_force[:, self.nonwheel_ids].norm(dim=-1).amax(-1)
                wheel_peak = self.contact_force[:, self.wheel_ids, 2].clamp_min(0.).amax(-1)
            result.update(recovery_chassis_load_fraction=chassis/weight,
                recovery_nonwheel_load_fraction=nonwheel/weight,
                recovery_peak_wheel_load_fraction=wheel_peak/weight)
        return result

    def task_stable(self, *args):
        return task.stable(*args)

    def course_reset_cases(self, cases, ids):
        return cases

    def task_reward_terms(self, state, route, error, rising):
        from .getup_translation import enabled as free_translation
        terms = task.reward_terms(state["gravity"], state["height"], state["velocity"], state["omega"],
            self.commands, state["torque"], state["dq"], task.normalized_action_delta(state["action_delta"], self.plan),
            route["xy_excursion_m"], error, rising, state["dual_support"],
            position_tail=self.plan.get("quiet_position_tail"),
            free_rise_translation=free_translation(self.plan))
        terms["leg_speed"] = -.002*state["leg_speed_square"]
        return terms

    def task_failures(self, state, route, rising):
        return task.failures(state["gravity"], state["height"], state["nonwheel_force"],
            route["xy_excursion_m"], state["elapsed"], rising, ~self.recovery_lane, self.plan)

    def course_metric_mask(self):
        return ~self.passive

    def task_commands(self, elapsed):
        value = task.commands(self.course_targets, elapsed, self.recovered_at, self.recovery_lane, self.plan,
            moving_height=self.course_moving_height,abrupt_brake=self.course_abrupt_brake)
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
        terms = self.task_reward_terms(state, route, error, rising)
        stage_extra = {}
        sequence_scores = None
        if self.multi_pose_cfg is not None:
            from .multi_pose_recovery import recovery_terms
            from .standard_getup_reward import guidance, ordered
            reward_state = self._recovery_reward_state(state, route)
            dense = self.multi_pose_cfg.get("dense_reward", {})
            if ordered(dense):
                sequence_scores = guidance(reward_state, dense)
            new_terms = recovery_terms(reward_state, self.multi_pose_cfg,
                self.recovery_impact.cost_integral/self.policy_dt, scores=sequence_scores)
            terms = {key:torch.where(rising, new_terms.get(key,torch.zeros_like(value)),value)
                     for key,value in terms.items()}
            for key,value in new_terms.items():
                if key not in terms:
                    terms[key] = torch.where(rising,value,torch.zeros_like(value))
        bad = self.task_failures(state, route, rising)
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
            if sequence_scores is not None:
                stage_extra["standard_getup_guidance"] = dict(sequence_scores,
                    physical_knee_max_deg=route["knee_deg"].amax(-1),
                    wheel_fore_aft_error_m=reward_state["recovery_fore_aft_error_m"])
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
        if self.recovery_training_focus is not None:
            from .recovery_discovery import completed_timeout
            stage_extra["recovery_training_timeout"] = completed_timeout(self.recovery_training_focus,
                self.quiet_hold, self.recovery_lane, terminated, training=self.recovery_training_run,
                evaluating=self.evaluation_cases is not None)
            # Reuse the original timeout/bootstrap path without changing the
            # passive-admitted physical loop or marking a success as failure.
            self.episode_limits.copy_(torch.where(stage_extra["recovery_training_timeout"],
                self.episode_length_buf, self.episode_limits))
            start_course = self.recovery_sampler.start_curriculum
            if start_course is not None and self.recovery_training_run and self.evaluation_cases is None:
                ended = self.recovery_lane & (terminated | (self.episode_length_buf >= self.episode_limits))
                ids = ended.nonzero().flatten().tolist()
                outcome = start_course.observe([self.course_cases[i].get("recovery_start_phase", -1) for i in ids],
                    stage_extra["recovery_training_timeout"][ended].tolist())
                stage_extra["recovery_start_outcomes"] = self.quiet_hold.new_tensor(
                    [outcome["phase"], outcome["episodes"], outcome["successes"]])
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
            rise_excursion_over_reference=(rising &
                (route["xy_excursion_m"] > self.plan["rise_training_abort_excursion_m"])),
            task_recovery_rising=rising.clone(), **stage_extra)

    def task_metrics(self, state, extra, reward, done, terminated):
        result = {"ProneMotion/updates": reward.new_tensor(self.actor_updates),
            "ProneMotion/recovery_completed_events": state["first"].float().sum(),
            "ProneMotion/quiet30_completed_events": extra["quiet30_completed_event"].float().sum(),
            "ProneMotion/terminated_events": terminated.float().sum(),
            "ProneMotion/timeout_events": (done & ~terminated).float().sum(),
            "ProneMotion/wheel_speed_peak_rad_s": state["dq"][:, [2, 5]].abs().max()}
        if getattr(self, "encoder_increment_actions", None) is not None:
            result.update(self.encoder_increment_actions.metrics())
        if "getup_translation" in self.plan:
            rising = extra["task_recovery_rising"] & ~self.passive
            count = rising.sum().clamp_min(1)
            displacement = extra["prone_route"]["xy_excursion_m"]
            result.update({
                "GetupTranslation/active_displacement_mean_m": torch.where(rising, displacement, 0.).sum()/count,
                "GetupTranslation/active_displacement_peak_m": torch.where(rising, displacement, 0.).max(),
                "GetupTranslation/over_reference_fraction": (rising &
                    extra["rise_excursion_over_reference"]).float().sum()/count})
        if self.multi_pose_cfg is not None:
            from .multi_pose_recovery import FAMILIES, LEG_CLASSES
            from .recovery_impact import GROUPS
            result.update({"Recovery/flip_events_per_step":extra["recovery_flip_event"].float().sum(),
                "Recovery/suffix_entry_events_per_step":extra["recovery_suffix_entry_event"].float().sum(),
                "Recovery/part_impact_failures_per_step":extra["recovery_part_impact_bad"].float().sum()})
            for g,group in enumerate(GROUPS):
                result[f"Recovery/impact/{group}/force_peak_n"] = extra["recovery_force_peak_n"][:,g].max()
                result[f"Recovery/impact/{group}/window_impulse_peak_ns"] = extra["recovery_impulse_peak_ns"][:,g].max()
            recovery_metrics = (self.sampling.recovery_metrics(self.recovery_lane,
                extra['recovery_initial_family'], extra['recovery_initial_leg_class'], state['first'],
                state['enable'], terminated, extra['recovery_active_seconds']) if self.sampling is not None else None)
            for f,family in enumerate(FAMILIES):
                for l,cls in enumerate(LEG_CLASSES):
                    prefix = f"Recovery/{family}/{cls}/"
                    if recovery_metrics is not None:
                        result.update({prefix+name: recovery_metrics[f*len(LEG_CLASSES)+l, j]
                            for j,name in enumerate(('enabled_events_per_step', 'completed_events_per_step',
                                'failure_events_per_step', 'active_completion_time_sum_s'))})
                        continue
                    mask = (self.recovery_lane & (extra["recovery_initial_family"] == f)
                            & (extra["recovery_initial_leg_class"] == l))
                    first = state["first"] & mask
                    result[prefix+"enabled_events_per_step"] = (mask & state["enable"]).float().sum()
                    result[prefix+"completed_events_per_step"] = first.float().sum()
                    result[prefix+"failure_events_per_step"] = (mask & terminated).float().sum()
                    result[prefix+"active_completion_time_sum_s"] = torch.where(first,
                        extra["recovery_active_seconds"],0.).sum()
        if extra.get("recovery_training_timeout") is not None:
            result.update({
                "RecoveryTraining/completion_timeout_events": extra["recovery_training_timeout"].float().sum(),
                "RecoveryTraining/recovery_lane_fraction": self.recovery_lane.float().mean()})
            start_course = self.recovery_sampler.start_curriculum
            if start_course is not None:
                result.update({
                    "RecoveryStart/phase": self.quiet_hold.new_tensor(start_course.phase),
                    "RecoveryStart/window_trials": self.quiet_hold.new_tensor(len(start_course.window)),
                    "RecoveryStart/window_success_rate": self.quiet_hold.new_tensor(
                        sum(start_course.window)/max(1,len(start_course.window)))})
                if "recovery_start_outcomes" in extra:
                    phase, episodes, successes = extra["recovery_start_outcomes"].unbind()
                    for i in range(5):
                        result[f"RecoveryStart/phase{i}/episode_events_per_step"] = torch.where(phase == i, episodes, 0.)
                        result[f"RecoveryStart/phase{i}/success_events_per_step"] = torch.where(phase == i, successes, 0.)
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
        if hasattr(self, "paired_torque_saturation"):
            request = physical["spring_ff_requested_effort_nm"].unflatten(-1, (2, 2))
            limited = physical["spring_ff_limited_effort_nm"].unflatten(-1, (2, 2))
            requested_difference = request[..., 0]-request[..., 1]
            clipped = request.clamp(-40., 40.)
            active = (~physical["wheel_disabled"]).unsqueeze(-1).expand_as(requested_difference)
            def active_mean(value):
                return torch.where(active, value, 0.).sum()/active.sum().clamp_min(1)
            result["LegAllocation/shape_priority_enabled"] = request.new_tensor(1.)
            result["LegAllocation/independent_clip_differential_error_abs_mean_nm"] = active_mean(
                (requested_difference-(clipped[..., 0]-clipped[..., 1])).abs())
            result["LegAllocation/allocated_differential_error_abs_mean_nm"] = active_mean(
                (requested_difference-(limited[..., 0]-limited[..., 1])).abs())
        for index, side in enumerate(("left", "right")):
            prefix = "ProneMotion/wheel_model/"+side+"/"
            result[prefix+"motor_effort_abs_mean_nm"] = physical["motor_effort"][:, :, (2, 5)[index]].abs().mean()
            result[prefix+"passive_residual_abs_mean_nm"] = physical["wheel_passive_effort"][:, :, index].abs().mean()
            result[prefix+"drive_bound_min_nm"] = physical["wheel_drive_bound"][:, :, index].min()
            result[prefix+"motoring_derated_fraction"] = (physical["wheel_drive_bound"][:, :, index] < 4.5-1e-6).float().mean()
        for name, value in extra["task_reward_terms"].items():
            result["ProneMotion/reward_density_"+name] = value.mean()
        if extra.get("standard_getup_guidance") is not None:
            scores = extra["standard_getup_guidance"]
            mask = extra["task_recovery_rising"] & ~self.passive
            names = ("fold_score", "support_capture", "height_gate", "support_error_m",
                     "physical_knee_max_deg", "wheel_fore_aft_error_m", "geometry_valid")
            if "dual_wheel_bearing" in scores:
                names += ("dual_wheel_bearing", "body_horizontal_rate_rad_s", "rate_readiness", "capture_stability")
            if "preparation_score" in scores:
                names += ("preparation_score", "chassis_load_fraction", "nonwheel_load_fraction",
                          "quiet_body_free_score", "fold_phase", "peak_wheel_load_fraction",
                          "fold_wheel_clearance_error", "fold_wheel_load_error")
            fields = torch.stack([scores[name].float() for name in names], -1)
            means = torch.where(mask[:, None], fields, 0.).sum(0)/mask.sum().clamp_min(1)
            result.update({"StandardGetup/"+name: means[i] for i, name in enumerate(names)})
        if self.sampling is not None:
            fields = dict(height_m=state['height'], vx_m_s=state['velocity'][:, 0],
                yaw_rad_s=state['omega'][:, 2], commanded_vx_m_s=self.commands[:, 0],
                commanded_yaw_rad_s=self.commands[:, 1], commanded_height_m=self.commands[:, 2],
                tilt_deg=torch.rad2deg(torch.acos((-state['gravity'][:, 2]).clamp(-1., 1.))),
                anchor_error_m=extra['course_anchor_error_m'], reward=reward)
            means, outside, present = self.sampling.role_metrics(torch.stack(list(fields.values()), -1),
                extra['role_index'], self.course_metric_mask(), physical['spring_outside_identified'])
            for index, (role, has_samples) in enumerate(zip(task.ROLES, present.tolist())):
                if has_samples:
                    result.update({f'ProneMotion/{role}/{name}': means[index, j]
                                   for j, name in enumerate(fields)})
                    result[f'ProneMotion/{role}/spring_outside_identified_fraction'] = outside[index]
            return result
        masks = [(extra["role_index"] == index) & self.course_metric_mask() for index in range(len(task.ROLES))]
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

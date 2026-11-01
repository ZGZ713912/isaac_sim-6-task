"""Unmodified Wheelbipe V14 task with v2.0.2 physical-model overrides."""
from __future__ import annotations

import json
import re
import torch

import isaaclab.sim as sim
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz
from agent_tasks.direct.wheelbipe.wheelbipe_V14.env import WheelbipeV14Env

from .model import ACTIVE, BODY_ALIASES, BUNDLE, JOINT_ALIASES, ROOT, CollisionSupport, PairedTargets, ResetLookup, digest


class OfflineTerrainImporter(TerrainImporter):
    """Same static PhysX plane/material, with no network grid-decoration asset."""
    def import_ground_plane(self,name,size=(2.e6,2.e6)):
        path = self.cfg.prim_path+"/"+name
        if path in self.terrain_prim_paths:
            raise ValueError("Ground plane already exists: "+path)
        self.terrain_prim_paths.append(path)
        cfg = sim.GroundPlaneCfg(usd_path=str(ROOT/"model/wheelbipe_port_v202/ground_plane.usda"),
            physics_material=self.cfg.physics_material,size=size,color=None)
        cfg.func(path,cfg)


class MappedArticulation(Articulation):
    """Expose semantic names without changing a CAD frame, axis, or USD joint."""
    @property
    def joint_names(self):
        return [JOINT_ALIASES[n] for n in super().joint_names]

    @property
    def body_names(self):
        return [BODY_ALIASES[n] for n in super().body_names]

    @property
    def cad_joint_names(self):
        return super().joint_names

    @property
    def cad_body_names(self):
        return super().body_names


class WheelbipePortEnv(WheelbipeV14Env):
    def __init__(self, *args, **kwargs):
        self._port_ready = False
        self._port_reset_q = None
        super().__init__(*args, **kwargs)
        from wheeled_tasks.chassis.full_robot_v202 import FullRobotV202
        self.asset = FullRobotV202()
        self.reset_lookup = ResetLookup(self.robot.cad_joint_names,self.device)
        self.collision_support = CollisionSupport(self.robot.cad_joint_names,self.device)
        self.reset_penetration_peak = torch.zeros((),device=self.device)
        self.control, self.wheel_response = self.asset.components(self.num_envs,self.device)
        self.paired_targets = PairedTargets(self.control.gas_compensation.model,self.device)
        # The response integrates fractional 5.5/3.5 ms delays on the actual
        # physics clock. The immutable asset's historical 1ms receipt is not
        # altered or mistaken for this experiment's 5ms sample period.
        from wheeled_tasks.chassis.identified_wheel_response import IdentifiedWheelResponse
        self.wheel_response = IdentifiedWheelResponse(self.num_envs,self.device,self.physics_dt,
            self.wheel_response.cfg,root=ROOT)
        self.leg_friction = self.asset.leg_friction(self.num_envs,self.device)
        self.motor_ids = [self.robot.cad_joint_names.index(n) for n in ACTIVE]
        self.spring_ids = [self.robot.cad_joint_names.index(n) for n in self.asset.manifest["spring_joint_names"]]
        self.leg_target_order = [self._legs_act_idx.index(self.motor_ids[i]) for i in (0,1,3,4)]
        self.wheel_target_order = [self._wheel_idx.index(self.motor_ids[i]) for i in (2,5)]
        self.motor_effort = torch.zeros(self.num_envs,6,device=self.device)
        self.applied_physical_effort = self.motor_effort.clone()
        self._initialize_domains()
        self._port_ready = True
        self.port_physics_calls = 0
        self.static_priv_obs = self._get_static_priv_obs()
        self._last_policy_tick = None
        self._policy_previous = None
        self._constraint_points0 = torch.tensor([c["local_pos0_m"] for c in self.asset.spec["constraints"]],device=self.device)
        self._constraint_points1 = torch.tensor([c["local_pos1_m"] for c in self.asset.spec["constraints"]],device=self.device)
        self._constraint_body0 = [self.robot.cad_body_names.index(c["body0"]) for c in self.asset.spec["constraints"]]
        self._constraint_body1 = [self.robot.cad_body_names.index(c["body1"]) for c in self.asset.spec["constraints"]]
        self.port_constraint_gap_peak = torch.zeros((),device=self.device)
        self.port_joint_speed_peak = torch.zeros((),device=self.device)
        self.port_body_angular_speed_peak = torch.zeros((),device=self.device)
        knee_names = ("L_joint2","R_jonit2")
        self._port_knee_ids = [self.robot.cad_joint_names.index(n) for n in knee_names]
        self._port_knee_signs = self.motor_effort.new_tensor([1.,-1.])
        endpoints = self.reset_lookup.rows[[0,-1]][:,[self.reset_lookup.names.index(n) for n in knee_names]]*self._port_knee_signs
        self._port_knee_limits = (endpoints.min(0).values,endpoints.max(0).values)

    def _setup_scene(self):
        # Same scene construction as Wheelbipe25V3Env._setup_scene; use a real
        # Articulation subclass so EventManager's type/scene checks remain valid.
        self.robot = MappedArticulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.height_scanner = RayCaster(self.cfg.height_scanner) if self._use_raycast_height() else None
        if self.height_scanner is not None:
            self.scene.sensors["height_scanner"] = self.height_scanner
        for name in ("left_wheel_height_scanner","right_wheel_height_scanner","dot_scanner"):
            cfg = getattr(self.cfg,name,None)
            enabled = cfg is not None and (name!="dot_scanner" or bool(getattr(self.cfg,"enable_scan_dot",False)))
            sensor = RayCaster(cfg) if enabled else None
            setattr(self,name,sensor)
            if sensor is not None:
                self.scene.sensors[name] = sensor
        self._set_play_height_scanner_debug_vis()
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = OfflineTerrainImporter(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim.DomeLightCfg(intensity=2000.,color=(.75,.75,.75))
        light.func("/World/Light",light)

    def _find_contact_sensor_indices(self, expressions):
        if isinstance(expressions,str):
            expressions = [expressions]
        names = [BODY_ALIASES.get(n,n) for n in self.contact_sensor.body_names]
        return sorted({i for expression in expressions for i,n in enumerate(names) if re.match(expression,n)})

    def _inverse_kinematics(self, length, angle, links_length=None, alpha_offset=None):
        if not hasattr(self,"reset_lookup"):
            self.reset_lookup = ResetLookup(self.robot.cad_joint_names,self.device)
        self._port_reset_q = self.reset_lookup.sample(length,angle)
        ids = [self.robot.joint_names.index(n) for n in self.cfg.ordered_leg_joint_names]
        return self._port_reset_q[:,ids].reshape(len(length),6,2).transpose(-2,-1)

    def _custom_reset_random(self, env_ids):
        self._port_reset_q = None
        super()._custom_reset_random(env_ids)
        # The old reset only writes twelve bar joints. Complete the two spring
        # pivots and two sliders from the same v2.0.2 closed-chain solution.
        if self._port_reset_q is not None:
            extras = [self.robot.cad_joint_names.index(n) for n in
                ("LLL_joint2","RRR_joint2","L_spring_slide","R_spring_slide")]
            value = self._port_reset_q[:,extras]
            self.robot.write_joint_state_to_sim(value,torch.zeros_like(value),extras,env_ids)
        if self._port_ready:
            # The source's fixed reset height assumed its shorter, differently
            # located collision geometry. Raise only penetrating reset states;
            # preserve orientations, velocities, leg poses and positive drops.
            root = self.robot.data.root_state_w[env_ids].clone()
            relative = self.collision_support.minimum_z(self.robot.data.joint_pos[env_ids],root[:,3:7])
            floor = self.terrain.env_origins[env_ids,2]
            penetration = floor-(root[:,2]+relative)
            self.reset_penetration_peak = torch.maximum(self.reset_penetration_peak,penetration.max())
            root[:,2] += (penetration+.002).clamp_min(0.)
            self.robot.write_root_state_to_sim(root,env_ids)
            self.wheel_response.reset(env_ids)
            self.motor_effort[env_ids] = 0.
            if self._policy_previous is not None:
                self._policy_previous[env_ids] = 0.
            self._last_policy_tick = None

    def _initialize_domains(self):
        cfg = self.cfg.port_randomization_contract
        profile = cfg["profiles"][self.cfg.port_randomization_profile]
        strength = float(self.cfg.port_randomization_strength)
        ref = cfg["reference"]
        lane = (torch.rand(self.num_envs,1,device=self.device)>=cfg["nominal_fraction"]).float()
        def amplitude(group):
            return lane*strength*profile[group]
        def uniform(shape,bound,group):
            return torch.empty(shape,device=self.device).uniform_(-1.,1.)*bound*amplitude(group)
        self.motor_strength = 1.+uniform((self.num_envs,6),ref["motor_strength_fraction"],"motor_strength")
        self.encoder_bias = uniform((self.num_envs,6),ref["encoder_bias_rad"],"encoder_bias")
        self.pd_encoder_bias = uniform((self.num_envs,4),ref["encoder_bias_rad"],"encoder_bias")
        self.gyro_bias = uniform((self.num_envs,3),ref["gyro_bias_rad_s"],"observation_noise")
        angles = uniform((self.num_envs,3),torch.tensor(ref["imu_fixed_degrees"],device=self.device)*torch.pi/180.,"imu_alignment")
        self.imu_mount_quat = quat_from_euler_xyz(*angles.unbind(-1))
        self.sensor_amplitude = amplitude("observation_noise")
        timing_enabled = lane*int(bool(profile["timing"]) and strength>0.)
        self.policy_age = (torch.randint(0,2,(self.num_envs,1),device=self.device)*timing_enabled).long()
        view = self.robot.root_physx_view
        env_ids = torch.arange(self.num_envs,dtype=torch.long)
        body_count = self.robot.num_bodies
        mass_scale = 1.+uniform((self.num_envs,body_count),ref["mass_fraction"],"mass_com")
        inertia_scale = 1.+uniform((self.num_envs,body_count),ref["independent_inertia_fraction"],"independent_inertia")
        masses, inertias, coms = view.get_masses().clone(),view.get_inertias().clone(),view.get_coms().clone()
        view.set_masses(masses*mass_scale.cpu(),env_ids)
        view.set_inertias(inertias*(mass_scale*inertia_scale).cpu().unsqueeze(-1),env_ids)
        base = self.robot.body_names.index("base_link")
        bound = torch.tensor(list(ref["base_com_offset_m"].values()),device=self.device)
        coms[:,base,:3] += uniform((self.num_envs,3),bound,"mass_com").cpu()
        view.set_coms(coms,env_ids)
        wheel_bodies = [self.robot.cad_body_names.index(n) for n in ("L_link3","R_link3")]
        scale = (mass_scale*inertia_scale)[:,wheel_bodies]
        added = torch.tensor([r["additional_armature_kg_m2"] for r in self.wheel_response.rows],device=self.device)*scale
        self.robot.write_joint_armature_to_sim(added,joint_ids=[self.motor_ids[i] for i in (2,5)])
        self.wheel_response.inertia = self.wheel_response.inertia*scale
        materials = view.get_material_properties().clone()
        wheel_shapes = self._get_material_indices([".*_wheel_link"])
        low,high = ref["ground_friction_range"]
        if profile["ground_friction"] and strength>0.:
            midpoint = .5*(low+high)
            coefficient = midpoint+torch.empty(self.num_envs,1).uniform_(low-midpoint,high-midpoint)*strength
        else:
            coefficient = torch.full((self.num_envs,1),.5*(low+high))
        materials[:,wheel_shapes,0] = coefficient
        materials[:,wheel_shapes,1] = coefficient
        materials[:,wheel_shapes,2] = 0.
        view.set_material_properties(materials,env_ids)
        self.domain_receipt = dict(profile=self.cfg.port_randomization_profile,strength=strength,
            contract_sha256=digest(ROOT/"contracts/wheel_leg_randomization_study_v2.json"),
            groups=profile,nominal_fraction=cfg["nominal_fraction"],leg_pd_scale=1.,spring_scale=1.,
            independent_inertia_fraction=ref["independent_inertia_fraction"],imu_wobble=False,
            policy_age_unit="one 20ms policy frame",wheel_aggregate_delay_ms=[5.5,3.5])

    def _update_obs(self, is_obs=False):
        super()._update_obs(is_obs)
        if not self._port_ready or not is_obs or self._last_policy_tick==self.common_step_counter:
            return
        self._last_policy_tick = self.common_step_counter
        current = torch.cat((self.obs_root_ang_vel_b,self.obs_projected_gravity_b,
                             self.obs_joint_pos,self.obs_joint_vel),-1)
        if self._policy_previous is None:
            self._policy_previous = current.clone()
        observed = torch.where(self.policy_age.bool(),self._policy_previous,current)
        self._policy_previous = current.clone()
        ref = self.cfg.port_randomization_contract["reference"]
        noise = lambda value,bound:value+torch.empty_like(value).uniform_(-bound,bound)*self.sensor_amplitude
        self.obs_root_ang_vel_b = noise(quat_apply_inverse(self.imu_mount_quat,observed[:,:3])+self.gyro_bias,ref["gyro_noise_rad_s"])
        self.obs_projected_gravity_b = noise(quat_apply_inverse(self.imu_mount_quat,observed[:,3:6]),ref["gravity_noise"])
        self.obs_joint_pos = noise(observed[:,6:12]+self.encoder_bias,ref["encoder_noise_rad"])
        self.obs_joint_vel = observed[:,12:].clone()
        self.obs_joint_vel[:,:4] = noise(self.obs_joint_vel[:,:4],ref["leg_velocity_noise_rad_s"])
        self.obs_joint_vel[:,4:] = noise(self.obs_joint_vel[:,4:],ref["wheel_velocity_noise_rad_s"])

    def _apply_action(self):
        self.port_physics_calls += 1
        q = self.robot.data.joint_pos[:,self.motor_ids]
        dq = self.robot.data.joint_vel[:,self.motor_ids]
        feedback = q.clone()
        feedback[:,(0,1,3,4)] += self.pd_encoder_bias
        legs = self.paired_targets(self.leg_actions[:,self.leg_target_order],feedback[:,(0,1,3,4)])
        wheels = self.wheel_actions[:,self.wheel_target_order].clamp(-90.,90.)
        motor = self.control.motor_efforts(feedback,dq,legs,wheels)*self.motor_strength
        motor[:,:2].clamp_(-40.,40.)
        motor[:,3:5].clamp_(-40.,40.)
        motor[:,(2,5)] = motor[:,(2,5)].clamp(-4.5,4.5)
        limits = self.control.gas_compensation.cfg["relative_soft_limits"]
        coefficients = q.new_tensor([[-1.,1.],[1.,-1.]])
        relative = (q[:,(0,1,3,4)].reshape(-1,2,2)*coefficients).sum(-1)
        effort = motor[:,(0,1,3,4)].reshape(-1,2,2)
        outward = (effort*coefficients).sum(-1)
        blocked = ((relative>=limits["maximum"]-limits["margin"])&(outward>0.))|((relative<=limits["minimum"]+limits["margin"])&(outward<0.))
        motor[:,(0,1,3,4)] = (effort-coefficients*torch.where(blocked,outward/2.,0.).unsqueeze(-1)).flatten(-2)
        enabled = self.finish_init & (self.episode_length_buf*self.step_dt >= self.predefined_reset_ground_zero_torque_until_time)
        motor[~enabled] = 0.
        motor = self.wheel_response.apply_torque(motor,dq)
        motor[~enabled] = 0.
        self.motor_effort.copy_(motor)
        physical = motor.clone()
        physical[:,(0,1,3,4)] += self.leg_friction.effort(dq[:,(0,1,3,4)])
        physical[:,(2,5)] += self.wheel_response.passive_effort
        self.applied_physical_effort.copy_(physical)
        self.robot.set_joint_effort_target(physical,joint_ids=self.motor_ids)
        self._apply_spring()
        self._update_obs()

    def _on_command_updated(self):
        super()._on_command_updated()
        if hasattr(self,"eval_command"):
            vx,wz,height = self.eval_command
            self.command[:,0],self.command[:,1],self.command[:,2] = vx,0.,wz
            self.height_cmd.fill_(height)

    def _apply_spring(self):
        compression = self.control.s0-self.robot.data.joint_pos[:,self.spring_ids]
        self.spring_force = self.control.gas_compensation.physical_force(compression)
        self.robot.set_joint_effort_target(self.spring_force,joint_ids=self.spring_ids)

    def constraint_gaps(self):
        p,q = self.robot.data.body_pos_w,self.robot.data.body_quat_w
        a = p[:,self._constraint_body0]+quat_apply(q[:,self._constraint_body0],self._constraint_points0[None].expand(self.num_envs,-1,-1))
        b = p[:,self._constraint_body1]+quat_apply(q[:,self._constraint_body1],self._constraint_points1[None].expand(self.num_envs,-1,-1))
        return (a-b).norm(dim=-1)

    def _measure_constraint_peak(self):
        gaps = self.constraint_gaps()
        peak = gaps.max()
        self.port_joint_speed_peak = torch.maximum(self.port_joint_speed_peak,
            self.robot.data.joint_vel.abs().max())
        self.port_body_angular_speed_peak = torch.maximum(self.port_body_angular_speed_peak,
            self.robot.data.root_ang_vel_w.norm(dim=-1).max())
        if getattr(self.cfg,"port_record_peaks",False) and peak>self.port_constraint_gap_peak:
            lane,pin = divmod(int(gaps.argmax()),6)
            self.port_peak_diagnostic = dict(lane=lane,pin=pin,physical_call=self.port_physics_calls,
                sample_scope="policy_observation_boundary",
                episode_time_s=float(self.episode_length_buf[lane])*self.step_dt,
                initialized=bool(self.finish_init[lane]),gaps_m=gaps[lane].cpu().tolist(),
                root_state=self.robot.data.root_state_w[lane].cpu().tolist(),
                joint_pos=self.robot.data.joint_pos[lane].cpu().tolist(),
                joint_vel=self.robot.data.joint_vel[lane].cpu().tolist(),
                previous_motor_effort_Nm=self.motor_effort[lane].cpu().tolist())
        self.port_constraint_gap_peak = torch.maximum(self.port_constraint_gap_peak,peak)

    def _get_observations(self):
        observations = super()._get_observations()
        if self._port_ready:
            self._measure_constraint_peak()
            logs = self.extras.setdefault("log",{})
            logs.update({
                "Port/constraint_gap_policy_sample_max_m":self.constraint_gaps().max(),
                "Port/constraint_gap_policy_sample_running_max_m":self.port_constraint_gap_peak.clone(),
                "Port/joint_speed_policy_sample_running_max_rad_s":self.port_joint_speed_peak.clone(),
                "Port/body_angular_speed_policy_sample_running_max_rad_s":self.port_body_angular_speed_peak.clone(),
                "Port/motor_leg_abs_mean_Nm":self.motor_effort[:,(0,1,3,4)].abs().mean(),
                "Port/motor_wheel_abs_mean_Nm":self.motor_effort[:,(2,5)].abs().mean(),
                "Port/spring_force_mean_N":self.spring_force.mean(),
                "Port/base_height_mean_m":self.robot.data.root_pos_w[:,2].mean(),
                "Port/reset_initial_penetration_running_peak_m":self.reset_penetration_peak.clone(),
                "Port/forward_error_abs_mean_m_s":(self.robot.data.root_lin_vel_b[:,0]-self.command[:,0]).abs().mean(),
                "Port/yaw_error_abs_mean_rad_s":(self.robot.data.root_ang_vel_b[:,2]-self.command[:,2]).abs().mean(),
            })
        return observations

    def _get_delay_time_lag_obs(self,delay_attr,cfg_attr,enabled_attr):
        if not self._port_ready:
            return super()._get_delay_time_lag_obs(delay_attr,cfg_attr,enabled_attr)
        if delay_attr=="obs_delay":
            # Preserve the original four privileged age slots, in physics ticks.
            return self.policy_age.float().expand(-1,4)*self.cfg.decimation
        if delay_attr=="act_delay":
            # Original slots are (legs, wheels). The wheel slot is the mean
            # calibrated delay; the receipt retains both actual side values.
            age = torch.zeros(self.num_envs,2,device=self.device)
            age[:,1] = .0045/self.physics_dt
            return age
        return super()._get_delay_time_lag_obs(delay_attr,cfg_attr,enabled_attr)

    def _get_dynamic_priv_obs(self,root_quat_inv):
        value = super()._get_dynamic_priv_obs(root_quat_inv)
        if self._port_ready:
            count = self._privileged_extra_joint_count
            stiffness = torch.zeros(self.num_envs,self.robot.num_joints,device=self.device)
            damping = torch.zeros_like(stiffness)
            stiffness[:,[self.motor_ids[i] for i in (0,1,3,4)]] = 160.
            damping[:,[self.motor_ids[i] for i in (0,1,3,4)]] = 2.5
            damping[:,[self.motor_ids[i] for i in (2,5)]] = .5
            ids = self._privileged_extra_actuate_idx
            value[:,:count] = self._clip_scale_critic_component("joint_stiffness",stiffness[:,ids])
            value[:,count:2*count] = self._clip_scale_critic_component("joint_damping",damping[:,ids])
        return value

    def _get_rear2_rear1_joint_limit_terms(self):
        # The source's two positive knee coordinates become mirrored native
        # CAD coordinates. Keep its three penalty formulas, but evaluate the
        # actual 40..110 degree branch and the correct outward velocity signs.
        ids,signs = self._port_knee_ids,self._port_knee_signs
        lower,upper = self._port_knee_limits
        ratio = min(max(float(self.cfg.rear2_rear1_joint_limit_boundary_ratio),0.),.49)
        margin = ((upper-lower)*ratio).clamp_min(1.e-6)
        position = self.joint_pos[:,ids]*signs
        below = ((lower+margin-position)/margin).clamp_min(0.)
        above = ((position-upper+margin)/margin).clamp_min(0.)
        torque = self.robot.data.applied_torque[:,ids]*signs
        velocity = self.joint_vel[:,ids]*signs
        return ((below+above).square().sum(-1),
            ((-torque).clamp_min(0.)*below+torque.clamp_min(0.)*above).square().sum(-1),
            ((-velocity).clamp_min(0.)*below+velocity.clamp_min(0.)*above).square().sum(-1))

"""V9 native terrain adapter. Sensors below are reward and diagnostic inputs only."""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_apply, quat_mul, quat_conjugate

from .terrain_curriculum import TERRAIN_TASKS
from .terrain_objectives import slope_geometry, slope_height, slope_leg_lengths


class TerrainEnvironmentMixin:
    def _spawn_v9_terrain(self, material):
        import isaaclab.sim as sim_utils
        from pxr import Gf, Usd, UsdGeom
        from .ball_field import BallField
        s = self._early_schedule
        if s.terrain is None:
            return
        t = s.terrain
        origin = self.terrain.env_origins.cpu()
        slope_cfg = s.spec["slope"]
        normals, rotations, centers = [x.cpu() for x in slope_geometry(t.slope_angle,t.slope_yaw,slope_cfg)]
        self._terrain_normal_readback_min_dot = 1.
        for task in ("slope","step_down","bump"):
            for i in s.mask(task).nonzero().flatten().tolist():
                path = self.scene.env_prim_paths[i]
                world = UsdGeom.Xformable(self.sim.stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                offset = origin[i]-torch.tensor(tuple(world))
                if task == "slope":
                    cfg = sim_utils.CuboidCfg(size=(slope_cfg["box_length_m"],slope_cfg["box_width_m"],slope_cfg["box_thickness_m"]),
                        collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=material)
                    cfg.func(path+"/Slope",cfg,translation=tuple((offset+centers[i]).tolist()),orientation=tuple(rotations[i].tolist()))
                    matrix = UsdGeom.Xformable(self.sim.stage.GetPrimAtPath(path+"/Slope")).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    normal = matrix.TransformDir(Gf.Vec3d(0.,0.,1.)).GetNormalized()
                    dot = float(torch.dot(torch.tensor(tuple(normal)),normals[i]))
                    self._terrain_normal_readback_min_dot = min(self._terrain_normal_readback_min_dot,dot)
                    if dot <= .9999:
                        raise ValueError("Slope USD normal differs from the scalar-first quaternion contract")
                elif task == "step_down":
                    c,h = s.spec[task],float(t.terrain_height[i])
                    cfg = sim_utils.CuboidCfg(size=(c["box_length_m"],c["box_width_m"],h),
                        collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=material)
                    cfg.func(path+"/StepDown",cfg,translation=tuple((offset+offset.new_tensor([c["box_length_m"]/2,0.,h/2])).tolist()))
                else:
                    c,h,yaw = s.spec[task],float(t.terrain_height[i]),float(t.bump_yaw[i])
                    for k in range(4):
                        width = float(t.strip_widths[i,k]); along = float(t.strip_starts[i,k])+width/2
                        cfg = sim_utils.CuboidCfg(size=(width,c["strip_length_m"],h),
                            collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=material)
                        cfg.func(path+f"/Bump_{k}",cfg,
                            translation=tuple((offset+offset.new_tensor([along*math.cos(yaw),along*math.sin(yaw),h/2])).tolist()),
                            orientation=(math.cos(yaw/2),0.,0.,math.sin(yaw/2)))
        self.ball_field = BallField(s,self.cfg.seed)
        self.ball_field.disabled = getattr(self.cfg,"port_skip_balls_diagnostic",False)
        if not self.ball_field.disabled:
            self.ball_field.spawn(self.sim.stage,self.scene.env_prim_paths,self.terrain.env_origins)

    def _reset_v9_terrain(self, env_ids):
        s,t = self.task_schedule,self.task_schedule.terrain
        if t is None:
            return
        self.terrain_objectives.reset(env_ids)
        ids = env_ids[t.terrain_lane[env_ids]]
        if not len(ids):
            return
        root = self.robot.data.default_root_state[ids].clone()
        root[:,:3] = self.terrain.env_origins[ids]
        root[:,0] += t.start_x[ids]; root[:,1] += t.start_y[ids]
        root[:,3:7] = 0.
        root[:,3],root[:,6] = (t.yaw[ids]/2).cos(),(t.yaw[ids]/2).sin()
        lengths = (s.height[ids,None]-.06).expand(-1,2).clone()
        slope = s.mask("slope")[ids] & t.active[ids]
        sid = ids[slope]
        lengths[slope] = slope_leg_lengths(s.height[sid],t.slope_angle[sid],t.slope_yaw[sid],t.yaw[sid])
        q = self.reset_lookup.sample(lengths,torch.zeros_like(lengths))
        minimum = self.collision_support.minimum_z(q,root[:,3:7])
        lift = torch.where(s.mask("step_down")[ids] & t.active[ids],t.terrain_height[ids],0.)
        root[:,2] += -minimum+.002+lift
        if len(sid):
            cfg = s.spec["slope"]
            local = root[slope,:2]-self.terrain.env_origins[sid,:2]
            plane = slope_height(local,t.slope_angle[sid],t.slope_yaw[sid],cfg)
            root[slope,2] = self.terrain.env_origins[sid,2]+plane+s.height[sid]+.005
            _,rotation,_ = slope_geometry(t.slope_angle[sid],t.slope_yaw[sid],cfg)
            relative = quat_mul(quat_conjugate(rotation),root[slope,3:7])
            # Select the chassis: tilted wheel AABBs conservatively extend below
            # their actual cylinders and cannot serve as a penetration oracle.
            minimum_plane = self.collision_support.minimum_z(q[slope],relative,body_names=("base_link",),exact=True)
            separation = (s.height[sid]+.005)*t.slope_angle[sid].cos()
            clearance = minimum_plane+separation
            if (clearance < .02).any():
                raise ValueError("Slope reset chassis clearance is below 2 cm")
        root[:,7:] = 0.
        self.robot.write_joint_state_to_sim(q,torch.zeros_like(q),None,ids)
        self.robot.write_root_state_to_sim(root,ids)
        self.predefined_reset_ground_zero_torque_until_time[ids] = 0.
        self.ball_field.reset(ids,self.terrain.env_origins)

    def _com_state(self):
        data,weights = self.robot.data,self._mixed_mass_fractions
        return ((data.body_com_pos_w[...,2]*weights).sum(-1)-self.terrain.env_origins[:,2],
                (data.body_com_lin_vel_w[...,2]*weights).sum(-1))

    def _observe_v9_terrain(self, age, xy, bottom, support, force_z, height, nonwheel):
        data = self.robot.data
        heading = quat_apply(data.root_quat_w,data.root_quat_w.new_tensor([1.,0.,0.]).expand(self.num_envs,-1))[:,:2]
        heading = heading/heading.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        speed = (data.root_lin_vel_w[:,:2]*heading).sum(-1)
        offset = quat_apply(data.body_quat_w[:,self._mixed_wheel_ids],self._mixed_wheel_offset.expand(self.num_envs,-1,-1))
        omega = data.body_ang_vel_w[:,self._mixed_wheel_ids]
        center_velocity = data.body_lin_vel_w[:,self._mixed_wheel_ids]+torch.cross(omega,offset,dim=-1)
        lateral = torch.stack((-heading[:,1],heading[:,0],torch.zeros_like(heading[:,0])),-1)
        rolling_speed = (omega*lateral[:,None]).sum(-1)*self._mixed_wheel_radius
        slip = (rolling_speed-(center_velocity[...,:2]*heading[:,None]).sum(-1)).abs().mean(-1)
        _,com_vz = self._com_state()
        root_xy = data.root_pos_w[:,:2]-self.terrain.env_origins[:,:2]
        self.terrain_objectives.observe(age,root_xy,xy,bottom,support,force_z,height,
            data.root_lin_vel_w[:,2],data.projected_gravity_b,speed,self.command[:,0],nonwheel,
            com_vz,slip,self.jump_observer)

    def _log_v9_terrain(self, logs):
        from .terrain_curriculum import STAND,SPIN,DRIVE,JUMP
        s,t,o = self.task_schedule,self.task_schedule.terrain,self.terrain_objectives
        for task in s.tasks:
            mask = s.mask(task)
            r,p = o.roll[mask].abs(),o.pitch[mask].abs()
            prefix = "Level/"+task+"/"
            logs[prefix+"roll_abs_mean_deg"] = r.mean()
            logs[prefix+"pitch_abs_mean_deg"] = p.mean()
            tilt = torch.maximum(r,p)
            logs[prefix+"tilt_over_3deg_fraction"] = (tilt > 3.).float().mean()
            logs[prefix+"tilt_over_6deg_fraction"] = (tilt > 6.).float().mean()
            flight = o.flight_landing_window[mask]
            valid = self._mixed_age()[mask] >= self.cfg.port_mixed_spec["level"]["reset_grace_s"]
            ordinary = ~flight & valid
            logs[prefix+"ordinary_tilt_over_3deg_fraction"] = ((tilt > 3.) & ordinary).sum()/ordinary.sum().clamp_min(1)
            logs[prefix+"ordinary_tilt_over_6deg_fraction"] = ((tilt > 6.) & ordinary).sum()/ordinary.sum().clamp_min(1)
            logs[prefix+"flight_landing_tilt_over_6deg_fraction"] = ((tilt > 6.) & flight).sum()/flight.sum().clamp_min(1)
            logs["Hold/"+task+"/radius_mean_m"] = o.hold.radius[mask].mean()
            logs["Hold/"+task+"/radius_max_m"] = o.hold.radius_max[mask].max()
            difficulty = torch.zeros_like(s.height)
            if task == "slope": difficulty = torch.rad2deg(t.slope_angle)*t.active
            elif task in ("step_down","bump"): difficulty = t.terrain_height*t.active
            elif task == "balls": difficulty = t.coverage
            elif task == "jump": difficulty = s.apex
            elif task == "platform": difficulty = s.rise
            logs["Mixed/"+task+"/difficulty_mean"] = difficulty[mask].mean()
        limits = t.limits()
        scalar = lambda x: s.height.new_tensor(float(x))
        logs["Mixed/slope/angle_limit_deg"] = scalar(limits["slope"])
        logs["Mixed/slope/jump_prob"] = scalar(limits["jump_probability"])
        for mode,name in ((SPIN,"spin"),(STAND,"stand"),(DRIVE,"drive"),(JUMP,"jump")):
            logs["Mixed/slope/"+name+"_envs"] = (s.mask("slope") & t.active & (t.mode == mode)).sum().float()
        logs["Mixed/slope/flat_envs"] = (s.mask("slope") & ~t.active).sum().float()
        spin = s.mask("slope") & (t.mode == SPIN)
        logs["Mixed/slope/spin_rpm_abs_mean"] = t.spin_rpm[spin].sum()/spin.sum().clamp_min(1)
        logs["Mixed/slope/spin_rpm_difficulty_mean"] = logs["Mixed/slope/spin_rpm_abs_mean"]
        logs["Mixed/slope/segment_success_count_per_step"] = o.slope_event.sum()
        down = o.step_down
        logs["Mixed/step_down/height_limit_m"] = scalar(limits["step_down"])
        logs["Mixed/step_down/window_envs"] = down.interval_window.sum().float()
        for name in ("success","stumble","timeout"):
            logs["Mixed/step_down/"+name+"_count_per_step"] = getattr(down,name+"_event").sum().float()
        mask = s.mask("step_down")
        logs["Mixed/step_down/landing_force_peak_bw"] = (down.force_peak[mask]/down.weight[mask]).mean()
        logs["Mixed/step_down/window_tilt_max_deg_mean"] = down.max_tilt_w[mask].mean()
        logs["Mixed/step_down/speed_ratio_min_mean"] = down.min_ratio_w[mask].mean()
        bump,mask = o.bump,s.mask("bump")
        logs["Mixed/bump/height_limit_m"] = scalar(limits["bump"])
        logs["Mixed/bump/strip_success_count_per_step"] = bump.success_count.sum()
        logs["Mixed/bump/strip_failure_count_per_step"] = bump.failure_count.sum()
        logs["Mixed/bump/window_tilt_max_deg_mean"] = bump.max_tilt[mask].amax(-1).mean()
        logs["Mixed/bump/speed_ratio_min_mean"] = bump.min_ratio[mask].amin(-1).mean()
        logs["Mixed/bump/height_dev_max_m_mean"] = bump.max_hdev[mask].amax(-1).mean()
        balls,mask = o.balls,s.mask("balls")
        logs["Mixed/balls/coverage"] = t.coverage[mask].mean()
        logs["Mixed/balls/tilt_max_deg_mean"] = balls.max_tilt[mask].mean()
        logs["Mixed/balls/radius_max_m_mean"] = balls.max_radius[mask].mean()
        logs["Mixed/balls/wheel_slip_mean_m_s"] = (balls.slip_sum[mask]/balls.samples[mask].clamp_min(1.)).mean()
        if not getattr(self,"port_sequence_metrics_in_runner",False):
            logs.update(o.metrics())

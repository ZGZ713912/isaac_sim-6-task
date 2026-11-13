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
            # minimum_z verifies reset penetration in the tilted support frame;
            # chassis clearance separately excludes the intended wheel contacts.
            minimum_plane = self.collision_support.minimum_z(q[slope],relative)
            separation = (s.height[sid]+.005)*t.slope_angle[sid].cos()
            if (minimum_plane+separation < -.003).any():
                raise ValueError("Slope reset collision envelope penetrates its plane")
            lower,upper = self.collision_support.bounds["base_link"]
            normal_b = quat_apply(quat_conjugate(relative),relative.new_tensor([0.,0.,1.]).expand(len(sid),-1))
            clearance = separation+(normal_b*(lower+upper)/2).sum(-1)-(normal_b.abs()*(upper-lower)/2).sum(-1)
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

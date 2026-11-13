"""Free rigid 17 mm balls: CPU layouts, batched USD authoring and indexed resets."""
from __future__ import annotations

import math
import re
import time

import numpy as np
import torch


def ball_count(cfg):
    return math.ceil(cfg["coverage_final"]*cfg["patch_side_m"]**2/(math.pi*(cfg["diameter_m"]/2)**2))


def make_layouts(cfg, seed):
    """Random sequential adsorption with a spatial hash; no initial overlaps."""
    count, spacing = ball_count(cfg), cfg["diameter_m"]+.0005
    half = (cfg["patch_side_m"]-cfg["diameter_m"])/2-.001
    rng = np.random.default_rng(seed)
    layouts = []
    for _ in range(cfg["layouts"]):
        points, cells = [], {}
        for _attempt in range(count*1000):
            p = rng.uniform(-half, half, 2)
            cell = tuple(np.floor(p/spacing).astype(int))
            neighbours = (cells.get((cell[0]+i, cell[1]+j), ()) for i in (-1,0,1) for j in (-1,0,1))
            if any(float(np.dot(p-q,p-q)) < spacing**2 for group in neighbours for q in group):
                continue
            points.append(p)
            cells.setdefault(cell, []).append(p)
            if len(points) == count:
                break
        if len(points) != count:
            raise ValueError("Ball layout did not reach the declared area coverage without overlap")
        layouts.append(points)
    return np.asarray(layouts, dtype=np.float32)


def parked_positions(cfg):
    count = ball_count(cfg)
    index = np.arange(count)
    width = math.ceil(math.sqrt(count))
    return np.stack((cfg["park_x_m"]+.02*(index % width), .02*(index//width)), -1).astype(np.float32)


class BallField:
    def __init__(self, schedule, seed):
        self.s, self.cfg = schedule, schedule.spec["balls"]
        self.count = ball_count(self.cfg)
        self.lanes = schedule.mask("balls").nonzero().flatten()
        self.layouts = torch.as_tensor(make_layouts(self.cfg, seed+5919), device=schedule.device)
        self.parked = torch.as_tensor(parked_positions(self.cfg), device=schedule.device)
        self.view = None
        self.disabled = False
        self.spawn_seconds = 0.

    def spawn(self, stage, env_paths, terrain_origins):
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
        started = time.monotonic()
        layer = stage.GetEditTarget().GetLayer()
        origins = terrain_origins.detach().cpu().numpy()
        layout = self.layouts[0].cpu().numpy()
        c, radius = self.cfg, self.cfg["diameter_m"]/2
        for rank, lane in enumerate(self.lanes.tolist()):
            env_path = env_paths[lane]
            world = UsdGeom.Xformable(stage.GetPrimAtPath(env_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            offset = origins[lane]-np.asarray(world)
            UsdGeom.Xform.Define(stage, env_path+"/Balls")
            mat_path = env_path+"/BallsMaterial"
            material = UsdShade.Material.Define(stage, mat_path)
            mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            friction = c["friction_variants"][rank % len(c["friction_variants"])]
            mat.CreateStaticFrictionAttr(friction); mat.CreateDynamicFrictionAttr(friction)
            mat.CreateRestitutionAttr(c["restitution"])
            pxmat = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
            pxmat.CreateFrictionCombineModeAttr("multiply")
            pxmat.CreateRestitutionCombineModeAttr("multiply")
            proto = env_path+"/Balls/b_0"
            sphere = UsdGeom.Sphere.Define(stage, proto)
            sphere.CreateRadiusAttr(radius)
            sphere.AddTranslateOp().Set(Gf.Vec3d(float(offset[0]+layout[0,0]),float(offset[1]+layout[0,1]),float(offset[2]+radius+.001)))
            prim = sphere.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(c["mass_kg"])
            UsdPhysics.CollisionAPI.Apply(prim)
            body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            body.CreateMaxDepenetrationVelocityAttr(c["max_depenetration_velocity_m_s"])
            body.CreateSolverPositionIterationCountAttr(c["solver_position_iterations"])
            body.CreateSolverVelocityIterationCountAttr(1)
            body.CreateLinearDampingAttr(c["linear_damping"])
            body.CreateAngularDampingAttr(c["angular_damping"])
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(.002); collision.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
            with Sdf.ChangeBlock():
                for i in range(1,self.count):
                    path = env_path+f"/Balls/b_{i}"
                    if not Sdf.CopySpec(layer, proto, layer, path):
                        raise RuntimeError("Failed to copy rigid ball prototype")
                    layer.GetAttributeAtPath(path+".xformOp:translate").default = Gf.Vec3d(
                        float(offset[0]+layout[i,0]),float(offset[1]+layout[i,1]),float(offset[2]+radius+.001))
            if time.monotonic()-started > 300:
                raise RuntimeError("Ball generation exceeded 300 s; reduce patch size under the V9 performance contract")
        self.spawn_seconds = time.monotonic()-started

    def bind(self, sim):
        if self.disabled:
            return
        self.view = sim.physics_sim_view.create_rigid_body_view("/World/envs/env_*/Balls/b_*")
        if self.view.count != len(self.lanes)*self.count:
            raise ValueError("Rigid ball view population differs from the authored field")
        # PhysX does not promise numeric/lexicographic USD order. Resolve the
        # actual paths and assert complete per-lane, per-ball coverage once.
        lookup = np.full((len(self.s.role),self.count),-1,dtype=np.int64)
        lanes = set(self.lanes.tolist())
        for index,path in enumerate(self.view.prim_paths):
            match = re.fullmatch(r"/World/envs/env_(\d+)/Balls/b_(\d+)",path)
            if not match:
                raise ValueError("Unexpected ball view path: "+path)
            lane,ball = map(int,match.groups())
            if ball >= self.count or lookup[lane,ball] >= 0 or lane not in lanes:
                raise ValueError("Ball tensor view contains a duplicate or foreign body")
            lookup[lane,ball] = index
        if (lookup[self.lanes.cpu().numpy()] < 0).any():
            raise ValueError("Ball view omitted an authored body")
        self.indices = torch.as_tensor(lookup,device=self.s.device)
        self.transforms = torch.zeros(self.view.count,7,device=self.s.device)
        self.transforms[:,6] = 1.  # Native tensor API uses xyzw, unlike the USD spawner.
        self.velocities = torch.zeros(self.view.count,6,device=self.s.device)

    def reset(self, env_ids, terrain_origins):
        if self.view is None:
            return
        ids = env_ids[self.s.mask("balls")[env_ids]]
        if not len(ids):
            return
        n = len(ids)
        choice = (self.s.random(n)*len(self.layouts)).long().clamp_max(len(self.layouts)-1)
        turns = (self.s.random(n)*4).long().clamp_max(3)
        angle = turns*math.pi/2
        cos,sin = angle.cos()[:,None],angle.sin()[:,None]
        p = self.layouts[choice]
        positions = torch.stack((cos*p[...,0]-sin*p[...,1],sin*p[...,0]+cos*p[...,1]),-1)
        # Locked lanes practice at x=-40 while the full field rests at its own
        # patch. Only the active course samples a reduced coverage.
        coverage = torch.where(self.s.terrain.active[ids],self.s.terrain.coverage[ids],self.cfg["coverage_final"])
        number = (self.count*coverage/self.cfg["coverage_final"]).round().long()
        live = torch.arange(self.count,device=self.s.device)[None] < number[:,None]
        positions = torch.where(live[...,None],positions,self.parked)
        positions += terrain_origins[ids,None,:2]
        indices = self.indices[ids].flatten()
        self.transforms[indices,:2] = positions.flatten(0,1)
        self.transforms[indices,2] = (terrain_origins[ids,2,None]+self.cfg["diameter_m"]/2+.001).expand(-1,self.count).flatten()
        self.transforms[indices,3:6] = 0.
        self.transforms[indices,6] = 1.
        self.velocities[indices] = 0.
        self.view.set_transforms(self.transforms,indices.to(torch.int32))
        self.view.set_velocities(self.velocities,indices.to(torch.int32))

    def diagnostics(self, terrain_origins):
        if self.disabled:
            return dict(disabled_diagnostic=True)
        transform = self.view.get_transforms()
        velocity = self.view.get_velocities()
        ids = self.indices[self.lanes]
        return dict(count=self.view.count,balls_per_lane=self.count,spawn_seconds=self.spawn_seconds,
            minimum_z_m=float((transform[ids,2]-terrain_origins[self.lanes,2,None]).min()),
            maximum_speed_m_s=float(velocity[:,:3].norm(dim=-1).max()))

"""Native 1 ms v2.0.2 plant for the shared RMCS traditional controller.

Create after AppLauncher. No policy environment, action decimation, position
teleports during physics, or ground-truth controller velocity is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .full_robot_asset import ACTIVE, SPRINGS, digest
from .traditional_model import equilibrium_pose, geometry
from .traditional_asset import TraditionalAsset


class FullRobotTraditionalPlant:
    dt = .001
    # RMCS canonical order: left hip/crank, right hip/crank, left/right wheel.
    canonical_to_asset = [0, 1, 3, 4, 2, 5]

    def __init__(self, root, bundle, *, device="cpu", friction=.8, mass_scale=1., fixed_base=False,
                 physics_dt=.001):
        import torch
        import warp as wp
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt
        import isaaclab.sim as su
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab_physx.physics import PhysxCfg
        from .identified_wheel_response import IdentifiedWheelResponse

        self.torch, self.wp, self.device = torch, wp, device
        self.physics_dt = float(physics_dt)
        self.substeps = round(self.dt/self.physics_dt)
        if not 1 <= self.substeps <= 8 or abs(self.substeps*self.physics_dt-self.dt)>1e-12:
            raise ValueError("Physics dt must divide the 1 ms controller interval")
        self.root, self.bundle = Path(root), Path(bundle)
        self.asset = TraditionalAsset(self.bundle)
        manifest, self.spec = self.asset.manifest, self.asset.spec
        self.springs = self.asset.spring_component(self.root, device)
        wheel_path = str(self.bundle / "wheel_model.json")
        self.wheels = IdentifiedWheelResponse(1, device, self.physics_dt, dict(enabled=True,
            model_path=wheel_path, model_sha256=digest(self.root / wheel_path), apply_aggregate_delay=True), root=self.root)
        self.sim = su.SimulationContext(su.SimulationCfg(device=device, dt=self.physics_dt,
            render_interval=100, physics=PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)))
        material = su.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction,
            friction_combine_mode="average", restitution=0.)
        # Native Plane collision follows the installed PhysX ground-plane helper.
        # The finite quad below is for display only.
        UsdGeom.Xform.Define(self.sim.stage, "/World/Ground")
        ground = UsdGeom.Mesh.Define(self.sim.stage, "/World/Ground/geometry/mesh")
        ground.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(x,y,0.) for x,y in
            [(-50.,-10.), (50.,-10.), (50.,10.), (-50.,10.)]]))
        ground.CreateFaceVertexCountsAttr([3,3])
        ground.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
        ground.CreateDoubleSidedAttr(True)
        plane = UsdGeom.Plane.Define(self.sim.stage, "/World/Ground/CollisionPlane")
        plane.CreateAxisAttr("Z")
        plane.CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(plane.GetPrim()).CreateCollisionEnabledAttr(True)
        material.func("/World/Ground/Material", material)
        su.bind_physics_material(plane.GetPath().pathString, "/World/Ground/Material")

        def actuator(names, limit, **extra):
            parameters = dict(stiffness=0., damping=0., effort_limit=limit,
                effort_limit_sim=limit, velocity_limit_sim=self.asset.limits["passive_joint_velocity_limit_rad_s"])
            parameters.update(extra)
            return IdealPDActuatorCfg(joint_names_expr=names, **parameters)

        passive = [j["name"] for j in self.spec["joints"]
            if j["name"] not in ACTIVE + SPRINGS and j["type"] != "fixed"]
        native_wheel = self.wheels.native_parameters()
        native_wheel["friction"] = native_wheel.pop("frictionloss")
        native_wheel["viscous_friction"] = native_wheel.pop("damping")
        self.robot = Articulation(ArticulationCfg(prim_path="/World/Robot",
            spawn=su.UsdFileCfg(usd_path=str(self.bundle / "robot.usda"),
                physics_material=material, physics_material_path="contact_material",
                activate_contact_sensors=True,
                rigid_props=su.RigidBodyPropertiesCfg(max_depenetration_velocity=1.),
                articulation_props=su.ArticulationRootPropertiesCfg(fix_root_link=fixed_base,
                    enabled_self_collisions=False, solver_position_iteration_count=64,
                    solver_velocity_iteration_count=32)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., .4),
                joint_pos=manifest["nominal_joint_pos"], joint_vel={".*": 0.}),
            actuators=dict(legs=actuator([ACTIVE[i] for i in (0, 1, 3, 4)], 40.),
                wheels=actuator([ACTIVE[i] for i in (2, 5)], self.wheels.native_effort_limit,
                    **native_wheel),
                passive=actuator(passive, 100.), springs=actuator(SPRINGS, 1500.)),
            soft_joint_pos_limit_factor=1.))
        contact_materials = {}
        for prim in self.sim.stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            path = prim.GetPath().pathString
            if path != "/World/Ground/CollisionPlane" and not path.startswith("/World/Robot/"):
                continue
            bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            physical = UsdPhysics.MaterialAPI(bound.GetPrim())
            coefficients = [physical.GetStaticFrictionAttr().Get(), physical.GetDynamicFrictionAttr().Get()]
            combine = PhysxSchema.PhysxMaterialAPI(bound.GetPrim()).GetFrictionCombineModeAttr().Get()
            if not np.allclose(coefficients, [friction, friction], atol=1e-7, rtol=0.):
                raise ValueError("Contact material differs from configured friction: "+path)
            if combine != "average":
                raise ValueError("Unexpected friction combine mode: "+path)
            contact_materials[path] = dict(material=bound.GetPath().pathString,
                static_friction=coefficients[0], dynamic_friction=coefficients[1], combine_mode=combine)
        if not any("L_link3/" in n for n in contact_materials) or not any("R_link3/" in n for n in contact_materials):
            raise ValueError("Wheel contact material binding was not verified")
        loops = [p for p in self.sim.stage.Traverse() if p.IsA(UsdPhysics.SphericalJoint)]
        if len(loops) != 6:
            raise ValueError("Full model requires six external loop closures")
        for prim in self.sim.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                api.CreateContactOffsetAttr(.001)
                api.CreateRestOffsetAttr(0.)
        self.sim.reset()
        import carb
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        from isaaclab_physx.physics import PhysxManager
        self.contact_view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
            ["/World/Robot/L_link3", "/World/Robot/R_link3"],
            filter_patterns=[["/World/Ground/CollisionPlane"]]*2, max_contact_data_count=128)
        self.ids = [self.robot.joint_names.index(n) for n in ACTIVE]
        self.canonical_ids = [self.ids[i] for i in self.canonical_to_asset]
        self.gas_ids = [self.robot.joint_names.index(n) for n in SPRINGS]
        self.passive_ids = [self.robot.joint_names.index(n) for n in passive]
        if set(self.robot.joint_names) != set(manifest["tree_joint_names"]):
            raise ValueError("v2.0.2 native moving DOFs differ")
        fixed = [p for p in self.sim.stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)
            and p.GetName() in manifest["fixed_joint_names"]]
        if len(fixed) != 2:
            raise ValueError("v2.0.2 requires its two fixed-zero gimbal joints")
        self.limit_ids = [self.robot.joint_names.index(n) for n in ("L_joint2", "R_jonit2", *SPRINGS)]
        by_name = {j["name"]:j for j in self.spec["joints"]}
        self.mechanical_limits = np.array([[float(by_name[n]["limit"][k]) for k in ("lower","upper")]
            for n in ("L_joint2", "R_jonit2", *SPRINGS)])
        native_limits = wp.to_torch(self.robot.root_view.get_dof_limits())[0,self.limit_ids].cpu().numpy()
        if not np.allclose(native_limits,self.mechanical_limits,atol=2e-6,rtol=0.):
            raise ValueError("v2.0.2 native mechanical limits differ")
        masses = wp.to_torch(self.robot.root_view.get_masses())
        expected_mass = sum(b["mass"] for b in self.spec["bodies"])
        if len(self.robot.body_names) != 21 or abs(float(masses.sum()) - expected_mass) > 1e-4:
            raise ValueError("Native PhysX mass differs from the full CAD")
        coms = wp.to_torch(self.robot.root_view.get_coms())[0].cpu().numpy()
        inertias = wp.to_torch(self.robot.root_view.get_inertias())[0].cpu().numpy().reshape(-1,3,3)
        by_body = {b["name"]:b for b in self.spec["bodies"]}
        com_error = max(float(abs(coms[i,:3]-by_body[n]["com"]).max())
            for i,n in enumerate(self.robot.body_names))
        inertia_error = max(float(abs(inertias[i]-by_body[n]["inertia"]).max())
            for i,n in enumerate(self.robot.body_names))
        if com_error > 2e-7 or inertia_error > 3e-7:
            raise ValueError("v2.0.2 native COM/inertias differ from canonical link frames")
        if not .7 <= mass_scale <= 1.3:
            raise ValueError("Mass perturbation outside reviewed range")
        if mass_scale != 1.:
            body_ids = torch.arange(21, device=device)
            env_ids = torch.tensor([0], device=device)
            self.robot.set_masses_index(masses=masses.to(device)*mass_scale, body_ids=body_ids, env_ids=env_ids)
        # Verify the added wheel inertia is applied exactly once on the native joints.
        added = wp.to_torch(self.robot.root_view.get_dof_armatures())[0, [self.ids[2], self.ids[5]]].cpu().numpy()
        expected_added = [r["additional_armature_kg_m2"] for r in self.wheels.rows]
        if not np.allclose(added, expected_added, atol=1e-8):
            raise ValueError("Native wheel armature mismatch")
        self.receipt = dict(plant=manifest["model_kind"], body_count=21,
            asset_manifest_sha256=digest(self.bundle/"manifest.json"),
            control_frame=manifest["control_frame"], additional_root_rotation=False,
            tree_joint_count=len(self.robot.joint_names), external_loop_count=len(loops),
            mass_kg=float(masses.sum()), mass_scale=mass_scale, friction=friction,
            contact_materials=contact_materials, friction_combine_mode="average",
            physics_dt_s=self.physics_dt, control_dt_s=self.dt, action_hold_ticks=self.substeps,
            wheel_response=self.wheels.summary(), wheel_armature_readback=added.tolist(),
            gimbal_control="supplied v2.0.2 fixed-zero joints; full CAD masses retained",
            gas_spring="v2.0.2 bound force curve and declared C1 endpoint trend; right symmetry accepted by user",
            knee_inner_limits_deg=self.asset.limits["knee_inner_limits_deg"],
            closedchain_source="supplied v2.0.2 USD loop joints; no pin regeneration",
            native_mechanical_limits=self.mechanical_limits.tolist(),
            native_com_max_error_m=com_error, native_inertia_max_error_kg_m2=inertia_error,
            ground_representation="native infinite Plane at z=0; finite display-only quad",
            fixed_base=fixed_base, hardware_deployment_ready=False)
        self.previous_velocity = np.zeros(3)
        self.specific_force = 9.81
        self.tick = 0
        self.effort = torch.zeros_like(self.robot.data.joint_pos.torch)

    def reset(self, length=.334, pitch=0., lift=0.):
        torch = self.torch
        q = equilibrium_pose(self.spec, length)
        frames, height, _, _ = geometry(self.spec, q)
        delta = frames["L_link3"][:3, 3] - frames["R_link3"][:3, 3]
        roll = np.arctan2(-delta[2], delta[1])
        quat = Rotation.from_euler("xyz", [roll, pitch, 0.]).as_quat()
        root = torch.tensor([[0., 0., height+.0005+lift, *quat]], dtype=torch.float32, device=self.device)
        joints = torch.tensor([[q[n] for n in self.robot.joint_names]], dtype=torch.float32, device=self.device)
        self.robot.reset()
        self.robot.write_root_link_pose_to_sim_index(root_pose=root)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=self.device))
        self.robot.write_joint_position_to_sim_index(position=joints)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints))
        self.effort.zero_()
        self.robot.set_joint_effort_target_index(target=self.effort)
        self.robot.write_data_to_sim()
        self.sim.forward()
        self.robot.update(0.)
        self.wheels.reset()
        self.previous_velocity.fill(0.)
        self.specific_force = 9.81
        self.tick = 0
        return self.observe()

    def observe(self):
        data = self.robot.data
        pose = data.root_link_pose_w.torch[0].detach().cpu().numpy()
        rotation = Rotation.from_quat(pose[3:])
        roll, pitch, yaw = rotation.as_euler("xyz")
        angular = rotation.inv().apply(data.root_com_vel_w.torch[0, 3:].detach().cpu().numpy())
        q = data.joint_pos.torch[0, self.canonical_ids].detach().cpu().numpy().astype(np.float64)
        dq = data.joint_vel.torch[0, self.canonical_ids].detach().cpu().numpy().astype(np.float64)
        # The controller packet contains only motor encoders and synthetic IMU.
        packet = np.r_[q, dq, pitch, angular[1], yaw, angular[2], self.specific_force]
        truth_velocity = rotation.inv().apply(data.root_link_lin_vel_w.torch[0].detach().cpu().numpy())
        limited = data.joint_pos.torch[0,self.limit_ids].detach().cpu().numpy()
        excess = np.maximum(np.maximum(self.mechanical_limits[:,0]-limited,
            limited-self.mechanical_limits[:,1]),0.)
        bodies = data.body_link_pose_w.torch[0].detach().cpu().numpy()
        lengths = []
        for side in ("L", "R"):
            delta = rotation.inv().apply(bodies[self.robot.body_names.index(side+"_link3"), :3]
                - bodies[self.robot.body_names.index(side+"_link1"), :3])
            lengths.append(float(np.linalg.norm(delta[[0,2]])))
        return dict(packet=packet, time_ns=self.tick*1_000_000, pitch=float(pitch), roll=float(roll),
            yaw_rate=float(angular[2]), velocity=float(truth_velocity[0]), base_height=float(pose[2]),
            leg_length_m=lengths,
            knee_limit_excess_rad=float(excess[:2].max()), spring_limit_excess_m=float(excess[2:].max()),
            position=pose[:3].tolist(), loop_gap_m=self.loop_gap(),
            wheel_force_n=np.linalg.norm(self.wp.to_torch(
                self.contact_view.get_contact_force_matrix(dt=self.physics_dt)).cpu().numpy().reshape(2, 3), axis=1))

    def diagnostics(self):
        from pxr import Usd, UsdGeom
        from .full_robot_asset import fk
        q = dict(zip(self.robot.joint_names, self.robot.data.joint_pos.torch[0].cpu().numpy().astype(float)))
        root = self.robot.data.root_link_pose_w.torch[0].cpu().numpy()
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(root[3:]).as_matrix()
        transform[:3, 3] = root[:3]
        frames = fk(self.spec, q, transform)
        actual = self.robot.data.body_link_pose_w.torch[0].cpu().numpy()
        error = max(float(np.linalg.norm(frames[n][:3, 3]-actual[i, :3])) for i, n in enumerate(self.robot.body_names))
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
            self.sim.stage.GetPrimAtPath("/World/Ground")).ComputeAlignedRange()
        force = self.wp.to_torch(self.contact_view.get_net_contact_forces(dt=self.physics_dt)).cpu().numpy()
        return dict(max_fk_position_error_m=error, joint_positions=q,
            wheel_contact_force_n=force.tolist(), ground_bounds=[list(bounds.GetMin()), list(bounds.GetMax())],
            wheel_quaternions={n: actual[self.robot.body_names.index(n), 3:].tolist() for n in ("L_link3", "R_link3")},
            wheel_world_positions={n: actual[self.robot.body_names.index(n), :3].tolist() for n in ("L_link3", "R_link3")})

    def loop_gap(self):
        poses = self.robot.data.body_link_pose_w.torch[0].detach().cpu().numpy()
        frames = {n: (poses[i, :3], Rotation.from_quat(poses[i, 3:])) for i, n in enumerate(self.robot.body_names)}
        return max(float(np.linalg.norm(frames[c["body0"]][0] + frames[c["body0"]][1].apply(c["local_pos0_m"])
            - frames[c["body1"]][0] - frames[c["body1"]][1].apply(c["local_pos1_m"]))) for c in self.spec["constraints"])

    def step(self, canonical_effort):
        torch = self.torch
        values = np.asarray(canonical_effort, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("Expected six finite output-axis torques")
        asset_effort = torch.tensor(values[[0, 1, 4, 2, 3, 5]][None], dtype=torch.float32, device=self.device)
        for _ in range(self.substeps):
            # Active host PD and LQR torque stay held for exactly 1 ms. Only
            # physical forces and the measured motor response advance here.
            dq = self.robot.data.joint_vel.torch
            motor = self.wheels.apply_torque(asset_effort, dq[:, self.ids])
            self.effort.zero_()
            self.effort[:, self.ids] = motor
            self.effort[:, [self.ids[2], self.ids[5]]] += self.wheels.passive_effort
            q = self.robot.data.joint_pos.torch
            compression = q.new_tensor([self.spec["spring_binding"][n]["compression_at_q_zero_m"]
                for n in SPRINGS])-q[:,self.gas_ids]
            self.effort[:,self.gas_ids] = self.springs.physical_force(compression)
            self.effort[:, self.passive_ids] = -.002*dq[:, self.passive_ids]
            self.robot.set_joint_effort_target_index(target=self.effort)
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(self.physics_dt)
        velocity = self.robot.data.root_link_lin_vel_w.torch[0].detach().cpu().numpy().astype(np.float64)
        specific = (velocity-self.previous_velocity)/self.dt + np.array([0., 0., 9.81])
        self.specific_force = float(np.linalg.norm(specific))
        self.previous_velocity = velocity.copy()
        self.tick += 1
        return self.observe()

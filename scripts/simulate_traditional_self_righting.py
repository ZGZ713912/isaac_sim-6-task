#!/usr/bin/env python3
"""Independent native full-vehicle recovery; the exact RMCS C++ core drives it.

No learner, policy, ROS vehicle command, or state write after episode placement.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts"), str(ROOT/"tools")]
import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(path)


class Controller:
    def __init__(self, library, rows, settings):
        self.lib = ct.CDLL(str(library))
        pointer = ct.POINTER(ct.c_double)
        self.lib.wheelleg_recovery_create.argtypes = [pointer, ct.c_int, pointer, ct.c_int]
        self.lib.wheelleg_recovery_create.restype = ct.c_void_p
        self.lib.wheelleg_recovery_step.argtypes = [ct.c_void_p, pointer, ct.c_double, ct.c_int, ct.c_int, pointer]
        self.lib.wheelleg_recovery_destroy.argtypes = [ct.c_void_p]
        self.lib.wheelleg_recovery_compensation.argtypes = [pointer,pointer]
        if self.lib.wheelleg_recovery_abi_version() != 6:
            raise ValueError("Recovery ABI differs")
        rows, settings = [np.ascontiguousarray(x, dtype=np.float64) for x in (rows, settings)]
        self.handle = self.lib.wheelleg_recovery_create(self.ptr(rows), len(rows), self.ptr(settings), len(settings))
        if not self.handle:
            raise ValueError("C++ rejected recovery configuration")
        self.result = np.zeros(24)

    @staticmethod
    def ptr(a):
        return a.ctypes.data_as(ct.POINTER(ct.c_double))

    def step(self, feedback, dt, enabled):
        data = np.ascontiguousarray(feedback, dtype=np.float64)
        if data.shape != (20,):
            raise ValueError("Expected four axes, compensation, IMU and wheels")
        if self.lib.wheelleg_recovery_step(self.handle, self.ptr(data), dt, enabled, 1, self.ptr(self.result)):
            raise RuntimeError("Recovery step failed")
        return self.result.copy()

    def compensation(self, positions):
        positions=np.ascontiguousarray(positions,dtype=np.float64)
        effort=np.zeros(4,dtype=np.float64)
        if self.lib.wheelleg_recovery_compensation(self.ptr(positions),self.ptr(effort)):
            raise ValueError('C++ spring compensation rejected feedback')
        return effort

    def close(self):
        if self.handle:
            self.lib.wheelleg_recovery_destroy(self.handle)
            self.handle = None


def prepare(args, bundle, spec, manifest):
    from build_full_vehicle_training import ResetGeometry
    from wheeled_tasks.chassis.side_fall_study import leg_configuration, orbital_angles
    builder = ResetGeometry(spec, bundle)
    cache = ROOT/"contracts/full_vehicle_recovery_geometry_v1.json"
    if cache.exists() and json.loads(cache.read_text())["model_spec_sha256"] == digest(bundle/"model_spec.json"):
        geometry = json.loads(cache.read_text())
    else:
        rows = []
        for beta in np.linspace(41., 109., 35):
            q = leg_configuration(builder, (beta,beta), (0.,0.))
            rows.append([float(beta)]+[q[n] for n in ("L_joint1","LL_joint1","R_joint1","RR_joint1")])
        bank = json.loads((bundle/"pose_bank.json").read_text())
        stand = next(p for p in bank["poses"] if p["name"] == "stand")
        q = stand.get("joint_positions", stand.get("full_joint_positions"))
        if q is None:
            q = dict(zip(bank["tree_joint_names"], stand["joint_pos"]))
        zero = spec["passive_knee_zero_inner_rad"]
        geometry = dict(schema="full-vehicle-recovery-geometry/v1",
            model_spec_sha256=digest(bundle/"model_spec.json"), rows=rows,
            stand_knee_deg=math.degrees(q["L_joint2"]+zero["L_joint2"]),
            stand_orbit_deg=float(np.mean(orbital_angles(spec,q))),
            stand_joint_positions=q)
        save(cache, geometry)
    if args.starts:
        bank=json.loads(args.starts.read_text())
        if bank.get('model_spec_sha256')!=geometry['model_spec_sha256']:
            raise ValueError('Start bank belongs to a different vehicle model')
        poses=bank.get('poses') or [r['final_pose']for r in bank.get('records',[]) if 'final_pose'in r]
        starts=[]
        for pose in poses:
            if args.start_names and pose['name'] not in args.start_names: continue
            q=pose['full_joint_positions']
            dq=pose['full_joint_velocities']
            quaternion=pose['root_quaternion_xyzw']
            velocity=pose['root_com_velocity_world']
            values=list(q.values())+list(dq.values())+pose['root_position_m']+quaternion+velocity
            if not np.isfinite(values).all() or abs(np.linalg.norm(quaternion)-1.)>1e-4:
                raise ValueError('Invalid measured initial state')
            if set(q)!=set(manifest['tree_joint_names']) or set(dq)!=set(q) or len(velocity)!=6:
                raise ValueError('Measured initial state has incomplete feedback')
            starts.append(dict(name=pose['name'],family=pose['family'],legs='measured',
                root_position_m=pose['root_position_m'],quaternion_xyzw=quaternion,
                full_joint_positions=q,full_joint_velocities=dq,root_com_velocity_world=velocity,
                initial_condition='measured passive endpoint, including residual velocities',
                source_bank=str(args.starts.resolve()),source_bank_sha256=digest(args.starts),
                source_static_equilibrium_proven=pose.get('static_equilibrium_proven',False)))
        if not starts or (args.start_names and set(args.start_names)!={s['name']for s in starts}):
            raise ValueError('Start selection is empty or includes unknown names')
        return geometry,starts
    orientations = dict(prone=(0.,0.,0.),left=(-90.,0.,0.),right=(90.,0.,0.),
        inverted=(180.,0.,0.),front=(0.,90.,0.),rear=(0.,-90.,0.),stand=(0.,0.,0.))
    combinations = dict(ff=(90.,90.),rr=(-90.,-90.),fr=(90.,-90.),rf=(-90.,90.))
    starts = []
    rng = np.random.default_rng(args.seed)
    for family in args.families:
        for name in args.leg_combinations:
            q = leg_configuration(builder, tuple(args.knees), combinations[name])
            if family=='stand': q=geometry['stand_joint_positions'].copy()
            q["M_joint1"] = float(rng.uniform(-math.pi,math.pi)) if args.random_gimbal else 0.
            rotation = Rotation.from_euler("xyz", orientations[family], degrees=True)
            height = -min(v[2] for _,v in builder.bounds(q, rotation.as_matrix()))+.001
            if family=='stand':
                canonical=next(p for p in json.loads((bundle/'pose_bank.json').read_text())['poses']if p['name']=='stand')
                rotation=Rotation.from_quat(canonical['root_quaternion_xyzw'])
                height=canonical['root_position_m'][2]
            starts.append(dict(name=family+"_"+name, family=family, legs=name,
                root_position_m=[0.,0.,float(height)], quaternion_xyzw=rotation.as_quat().tolist(),
                full_joint_positions=q,initial_gravity_body=rotation.inv().apply([0.,0.,-1.]).tolist(),
                initial_condition='geometry placement; motors enabled after requested delay'))
    return geometry, starts


def run(args):
    plan = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4.json").read_text())
    cfg = json.loads((ROOT/plan["mechanism"]["file"]).read_text())
    from wheeled_tasks.chassis.full_vehicle_training import authenticate
    authenticate(cfg, ROOT)
    bundle = ROOT/cfg["asset_directory"]
    spec, manifest = [json.loads((bundle/n).read_text()) for n in ("model_spec.json","manifest.json")]
    args.output.mkdir(parents=True, exist_ok=False)
    geo, starts = prepare(args,bundle,spec,manifest)
    save(args.output/"starts.json",dict(starts=starts,geometry=geo))
    folder = args.rmcs/"rmcs_ws/src/rmcs_core/src/controller/wheelleg"
    source_files = [folder/n for n in ("self_righting_core.hpp","self_righting_abi.cpp","recovery_geometry_v202.hpp")]
    snapshots=args.output/'sources';snapshots.mkdir()
    for path in [Path(__file__),*source_files]:shutil.copy2(path,snapshots/path.name)
    if geo['model_spec_sha256']not in source_files[2].read_text():
        raise ValueError('C++ geometry was generated for a different vehicle')
    library = args.output/"libwheelleg_recovery.so"
    subprocess.run(["g++","-std=c++23","-O2","-Wall","-Wextra","-Werror","-fPIC","-shared",
        str(snapshots/source_files[1].name),"-o",str(library)], check=True)
    settings = [args.speed,args.acceleration,args.velocity_gain,args.position_gain,args.fold_knee,
        args.flip_knee,geo["stand_knee_deg"],geo["stand_orbit_deg"],args.flip_orbit,args.sweep_orbit,
        args.balance_kp,args.balance_kd,args.balance_kv,args.flip_asymmetry,args.seconds,40.,3.14,
        args.pitch_trim_gain,args.yaw_rate_gain,args.integral_gain,args.integral_limit,args.support_gain]
    names = ['speed','acceleration','velocity_gain','position_gain','fold_knee','flip_knee',
        'stand_knee','stand_orbit','flip_orbit','sweep_orbit','balance_kp','balance_kd','balance_kv',
        'flip_asymmetry','timeout','torque_limit','speed_limit','pitch_trim_gain','yaw_rate_gain',
        'integral_gain','integral_limit','support_gain']
    variants = json.loads(args.variants.read_text()) if args.variants else [dict(name='default',settings={})]
    expanded=[]
    for s in starts:
        for variant in variants:
            row=copy.deepcopy(s); row['variant']=variant['name']; row['controller_settings']=settings.copy()
            if args.variants: row['name']+='__'+variant['name']
            for key,value in variant['settings'].items():
                if key not in names: raise ValueError('Unknown variant setting: '+key)
                row['controller_settings'][names.index(key)]=float(value)
            expanded.append(row)
    starts=expanded
    if args.motor_servo=='implicit-velocity' and any(s['controller_settings'][2]!=args.velocity_gain for s in starts):
        raise ValueError('Implicit drive gain is common to this batch; vary other gains or run separate batches')
    save(args.output/'starts.json',dict(starts=starts,geometry=geo))
    receipt = dict(schema="traditional-self-righting-native/v1", status="prepared", cases=starts,
        model_spec_sha256=digest(bundle/"model_spec.json"),asset_manifest_sha256=digest(bundle/"manifest.json"),
        asset_directory=cfg['asset_directory'],backend='PhysX CPU',
        sources={str(p):digest(p) for p in [Path(__file__),*source_files]},
        settings=settings,geometry=geo,speed_limit_rad_s=3.14,physics_dt_s=args.dt,
        motor_speed_governor="native active-DOF limit plus measured-speed braking and overspeed fault",
        controller_dt_s=args.control_dt,leg_friction_scale=args.leg_friction_scale,
        leg_friction_integrator=args.friction_integrator,
        motor_servo=args.motor_servo,
        body_friction=args.body_friction,base_friction=args.base_friction,
        collision_geometry=args.collision_geometry,
        nonwheel_friction_combine=args.nonwheel_friction_combine,
        gimbal="both axes passive, yaw optionally randomized",ppo_updates=0,hardware_motion=False,
        records=[],hardware_deployment_ready=False)
    save(args.output/"receipt.json",receipt)
    if args.prepare_only:
        print(json.dumps(dict(prepared=len(starts),settings=settings)),flush=True)
        return
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(dict(headless=True,device="cpu",enable_cameras=False))
    controllers=[]
    exit_code=0
    try:
        import torch
        import warp as wp
        from pxr import Gf,PhysxSchema, UsdGeom, UsdPhysics
        import isaaclab.sim as su
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import IdealPDActuatorCfg,ImplicitActuatorCfg
        from isaaclab_physx.physics import PhysxCfg, PhysxManager
        from isaaclab.utils.math import quat_apply, quat_apply_inverse
        from wheeled_tasks.chassis.gas_spring_compensation import GasSpringCompensation
        from wheeled_tasks.chassis.leg_motor_friction import LegMotorFriction, declaration
        from wheeled_tasks.chassis.identified_wheel_response import IdentifiedWheelResponse
        from torch.utils.tensorboard import SummaryWriter
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Independent simulation interrupted with signal '+str(signum))
        # Kit otherwise exits directly with status zero, leaving a running
        # receipt. Preserve partial evidence and a nonzero status instead.
        signal.signal(signal.SIGINT,interrupted)
        signal.signal(signal.SIGTERM,interrupted)
        torch.set_num_threads(2)
        count=len(starts); dt=args.dt
        sim=su.SimulationContext(su.SimulationCfg(device="cpu",dt=dt,render_interval=1000,
            physics=PhysxCfg(solver_type=1,enable_external_forces_every_iteration=True)))
        import carb
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing",False)
        floor=su.CuboidCfg(size=(80.,80.,.1),collision_props=su.CollisionPropertiesCfg(contact_offset=.002,rest_offset=0.),
            physics_material=su.RigidBodyMaterialCfg(static_friction=.5,dynamic_friction=.5,restitution=0.))
        floor.func("/World/Ground",floor,translation=(20.,20.,-.05))
        origins=np.array([[i%8*4.,i//8*4.,0.] for i in range(count)])
        for i,origin in enumerate(origins):
            UsdGeom.Xform.Define(sim.stage,f"/World/envs/env_{i}").AddTranslateOp().Set(tuple(origin))
        active=manifest["control_joint_names"]; springs=manifest["spring_joint_names"]
        legs=["L_joint1","LL_joint1","R_joint1","RR_joint1"]
        gimbals=["M_joint1","M_joint2"]
        passive=[n for n in manifest["tree_joint_names"] if n not in active+springs+gimbals]
        friction=LegMotorFriction(count,"cpu",declaration(ROOT),root=ROOT)
        spring=GasSpringCompensation(cfg["v5_control"]["gas_spring_compensation"],root=ROOT,device="cpu")
        wheel_path=bundle/"wheel_model.json"
        wheels=IdentifiedWheelResponse(count,"cpu",dt,dict(enabled=True,model_path=str(wheel_path.relative_to(ROOT)),
            model_sha256=digest(wheel_path),apply_aggregate_delay=True),root=ROOT)
        native=wheels.native_parameters()
        def actuator(names,cap,speed=1e9,damping=0.,**extra):
            return IdealPDActuatorCfg(joint_names_expr=names,stiffness=0.,damping=damping,
                effort_limit=cap,effort_limit_sim=cap,velocity_limit_sim=speed,**extra)
        robot=Articulation(ArticulationCfg(prim_path="/World/envs/env_[0-9]+/Robot",
            spawn=su.UsdFileCfg(usd_path=str(bundle/manifest["usd"]),activate_contact_sensors=True,
                collision_props=su.CollisionPropertiesCfg(contact_offset=.002,rest_offset=0.),
                rigid_props=su.RigidBodyPropertiesCfg(max_depenetration_velocity=1.),
                articulation_props=su.ArticulationRootPropertiesCfg(enabled_self_collisions=args.self_collision,
                    solver_position_iteration_count=64,solver_velocity_iteration_count=32)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,1.),joint_pos=manifest["nominal_joint_pos"],joint_vel={".*":0.}),
            actuators=dict(legs=(ImplicitActuatorCfg(joint_names_expr=legs,stiffness=0.,damping=args.velocity_gain,
                    effort_limit_sim=40.,velocity_limit_sim=args.native_speed_limit,
                    friction=0.,dynamic_friction=0.,viscous_friction=0.) if args.motor_servo=='implicit-velocity'
                else actuator(legs,friction.native_effort_limit,args.native_speed_limit,
                    friction=0.,dynamic_friction=0.,viscous_friction=0.)),
                wheels=actuator([active[2],active[5]],wheels.native_effort_limit,
                    armature=native["armature"],friction=native["frictionloss"],
                    dynamic_friction=native["dynamic_friction"],viscous_friction=native["damping"]),
                gimbals=actuator(gimbals,10.),passive=actuator(passive,100.,damping=.002),gas=actuator(springs,1000.,damping=.002)),
            soft_joint_pos_limit_factor=1.))
        for name,mu,combine in (("Body",args.body_friction,args.nonwheel_friction_combine),
                ("Base",args.base_friction,args.nonwheel_friction_combine),("Wheel",.7,"average")):
            m=su.RigidBodyMaterialCfg(static_friction=mu,dynamic_friction=mu,restitution=0.,friction_combine_mode=combine)
            m.func("/World/"+name+"Material",m)
        if args.collision_geometry=='cad-decomposition':
            import trimesh
            cad_meshes=[]
            receipt['temporary_collision_source_sha256']={}
            for body in spec['bodies']:
                if body['name'] in ('L_link3','R_link3'):continue
                mesh_path=bundle/body['mesh']
                if not mesh_path.exists():mesh_path=bundle.parent/'isaac'/body['mesh']
                cad=trimesh.load(mesh_path,force='mesh',process=True)
                transform=np.array(body['mesh_origin'])
                cad_meshes.append((body['name'],np.asarray(cad.vertices)@transform[:3,:3].T+transform[:3,3],np.asarray(cad.faces)))
                receipt['temporary_collision_source_sha256'][str(mesh_path)]=digest(mesh_path)
            for prim in list(sim.stage.Traverse()):
                path=str(prim.GetPath())
                if '/Robot/'in path and prim.HasAPI(UsdPhysics.CollisionAPI) \
                        and not any('/'+n+'/'in path for n in ('L_link3','R_link3')):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            for i in range(count):
                for name,vertices,faces in cad_meshes:
                    mesh=UsdGeom.Mesh.Define(sim.stage,f'/World/envs/env_{i}/Robot/{name}/RecoveryCadCollision')
                    mesh.CreatePointsAttr([Gf.Vec3f(*v)for v in vertices]);mesh.CreateFaceVertexCountsAttr([3]*len(faces))
                    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
                    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
                    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexDecomposition')
                    decomposition=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh.GetPrim())
                    decomposition.CreateMaxConvexHullsAttr(64);decomposition.CreateHullVertexLimitAttr(64)
                    decomposition.CreateVoxelResolutionAttr(500000);decomposition.CreateErrorPercentageAttr(.1)
                    decomposition.CreateShrinkWrapAttr(True)
                    props=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
                    props.CreateContactOffsetAttr(args.cad_contact_offset);props.CreateRestOffsetAttr(0.)
            receipt['temporary_collision_decomposition']=dict(max_hulls=64,hull_vertices=64,voxels=500000,error_percentage=.1,
                shrink_wrap=True,contact_offset_m=args.cad_contact_offset,rest_offset_m=0.)
            save(args.output/'receipt.json',receipt)
        adjacent={b['name']:set() for b in spec['bodies']}
        for joint in spec['joints']:
            adjacent[joint['parent']].add(joint['child']);adjacent[joint['child']].add(joint['parent'])
        for constraint in spec['constraints']:
            adjacent[constraint['body0']].add(constraint['body1']);adjacent[constraint['body1']].add(constraint['body0'])
        # These two closed-link assembly surfaces also touch in nominal CAD.
        nominal_assembly=[('LLL_link2','LL_link3'),('RRR_link2','RR_link3')]
        for a,b in nominal_assembly:adjacent[a].add(b);adjacent[b].add(a)
        if args.filter_leg_assembly:
            for side in ('L','R'):
                assembly=[b['name']for b in spec['bodies']if b['name'].startswith(side)]
                for name in assembly:adjacent[name].update(set(assembly)-{name})
        for prim in sim.stage.Traverse():
            path=str(prim.GetPath())
            if "/Robot/" not in path: continue
            if args.motor_servo=='implicit-velocity' and prim.GetName() in legs and prim.IsA(UsdPhysics.RevoluteJoint):
                # This passive-study USD deliberately has no joint drives.
                # Tensor damping alone does not create an absent PhysX drive.
                drive=UsdPhysics.DriveAPI.Apply(prim,'angular')
                drive.CreateTypeAttr('force')
                drive.CreateStiffnessAttr(0.)
                drive.CreateDampingAttr(args.velocity_gain)
                drive.CreateMaxForceAttr(40.)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
                if args.self_collision:
                    base=path.rsplit('/',1)[0]
                    UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(
                        [base+'/'+n for n in sorted(adjacent[prim.GetName()])])
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                material="Wheel" if any("/"+n+"/" in path for n in ("L_link3","R_link3")) else "Base" if "/base_link/" in path else "Body"
                if material=='Wheel' and prim.IsA(UsdGeom.Cylinder):
                    cylinder=UsdGeom.Cylinder(prim)
                    radius,height=cylinder.GetRadiusAttr().Get(),cylinder.GetHeightAttr().Get()
                    segments=64
                    points=[Gf.Vec3f(radius*math.cos(2*math.pi*k/segments),radius*math.sin(2*math.pi*k/segments),z)
                        for z in (-height/2,height/2) for k in range(segments)]
                    mesh=UsdGeom.Mesh.Define(sim.stage,prim.GetPath());mesh.CreatePointsAttr(points)
                    faces=[[k,(k+1)%segments,(k+1)%segments+segments,k+segments]for k in range(segments)]
                    faces+=[list(reversed(range(segments))),list(range(segments,2*segments))]
                    mesh.CreateFaceVertexCountsAttr([len(f)for f in faces])
                    mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
                    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
                    PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh.GetPrim()).CreateHullVertexLimitAttr(256)
                su.bind_physics_material(path,"/World/"+material+"Material")
        sim.reset(); robot.update(0.)
        if not math.isclose(sim.get_physics_dt(),dt,abs_tol=1e-12): raise ValueError('Native physics dt differs')
        order=[int(p.split("/env_")[1].split("/")[0]) for p in robot.root_view.prim_paths]
        starts=[starts[i] for i in order]; origins=torch.tensor(origins[order],dtype=torch.float32)
        ids=[robot.joint_names.index(n) for n in active]
        leg_ids=[robot.joint_names.index(n) for n in legs]
        gas_ids=[robot.joint_names.index(n) for n in springs]
        q0=torch.tensor([[s["full_joint_positions"][n] for n in robot.joint_names] for s in starts],dtype=torch.float32)
        roots=torch.tensor([s["root_position_m"]+s["quaternion_xyzw"] for s in starts],dtype=torch.float32)
        roots[:,:3]+=origins
        robot.reset(); wheels.reset()
        robot.write_root_link_pose_to_sim_index(root_pose=roots)
        initial_velocity=torch.tensor([s.get('root_com_velocity_world',[0.]*6)for s in starts],dtype=torch.float32)
        initial_dq=torch.tensor([[s.get('full_joint_velocities',{}).get(n,0.)for n in robot.joint_names]for s in starts],dtype=torch.float32)
        robot.write_root_com_velocity_to_sim_index(root_velocity=initial_velocity)
        robot.write_joint_position_to_sim_index(position=q0)
        robot.write_joint_velocity_to_sim_index(velocity=initial_dq)
        robot.update(0.)
        view=PhysxManager.get_physics_sim_view()
        bodies=[p.split("/Robot")[0]+"/Robot/"+n for p in robot.root_view.prim_paths for n in robot.body_names]
        colliders=[str(p.GetPath()) for p in sim.stage.Traverse() if str(p.GetPath()).startswith("/World/Ground/") and p.HasAPI(UsdPhysics.CollisionAPI)]
        contact=view.create_rigid_contact_view(bodies,filter_patterns=[colliders for _ in bodies],max_contact_data_count=64*len(bodies))
        if not colliders or contact.filter_count != len(colliders): raise ValueError("Missing native ground contacts")
        contact_order=[contact.sensor_paths.index(p) for p in bodies]
        c0=torch.tensor([c["local_pos0_m"] for c in spec["constraints"]]); c1=torch.tensor([c["local_pos1_m"] for c in spec["constraints"]])
        b0=[robot.body_names.index(c["body0"]) for c in spec["constraints"]]
        b1=[robot.body_names.index(c["body1"]) for c in spec["constraints"]]
        wheel_ids=[robot.body_names.index(n) for n in ("L_link3","R_link3")]
        wheel_collisions=[next(b for b in spec['bodies']if b['name']==name)['collisions'][0]
            for name in ('L_link3','R_link3')]
        wheel_centers=torch.tensor(np.array([np.array(c['origin'])[:3,3]for c in wheel_collisions]),dtype=torch.float32)
        wheel_axes=torch.tensor(np.array([np.array(c['origin'])[:3,2]for c in wheel_collisions]),dtype=torch.float32)
        wheel_radius=torch.tensor([c['radius']for c in wheel_collisions])
        wheel_half_width=torch.tensor([c['length']*.5 for c in wheel_collisions])
        controllers=[Controller(library,geo["rows"],s['controller_settings']) for s in starts]
        effort=torch.zeros_like(q0); gravity0=torch.tensor([[0.,0.,-1.]]).expand(count,3)
        peaks=np.zeros((count,4)); active_peaks=peaks.copy(); reference_peaks=peaks.copy(); gaps=np.zeros(count)
        first_fault_seconds=[None]*count
        contact_peaks=np.zeros((count,len(robot.body_names)))
        stable=np.zeros(count); completion=[None]*count; faults=np.zeros(count,dtype=int)
        completion_speed=[None]*count;completion_gap=[None]*count
        trace=[]; body_trace=[]; body_trace_time=[]; start_time=time.monotonic()
        writer=SummaryWriter(str(args.output/"events"))
        native_friction=wp.to_torch(robot.root_view.get_dof_friction_properties()).clone()
        native_force_limits=wp.to_torch(robot.root_view.get_dof_max_forces()).clone()
        articulation_indices=wp.from_numpy(np.arange(count,dtype=np.uint32),dtype=wp.uint32,device='cpu')
        previous_leg_q=q0[:,leg_ids].numpy().copy(); leg_turn_offset=np.zeros_like(previous_leg_q)
        receipt.update(status="running",environment_order=order,native_joint_names=robot.joint_names,
            native_body_names=robot.body_names,leg_friction=friction.summary(),self_collision_enabled=args.self_collision,
            self_collision_filtered_body_pairs=sorted({tuple(sorted((a,b)))for a in adjacent for b in adjacent[a]}),
            native_active_velocity_limits=wp.to_torch(robot.root_view.get_dof_max_velocities())[:,leg_ids].tolist())
        receipt.update(contact_sensor_paths=contact.sensor_paths,contact_index_map=contact_order,
            native_rest_offsets=wp.to_torch(robot.root_view.get_rest_offsets()).tolist(),
            passive_rotary_and_gas_numerical_damping=.002,wheel_collision_segments=64,
            wheel_hull_maximum_radial_deficit_m=.06*(1.-math.cos(math.pi/64)),
            active_angular_drive_api_authored=args.motor_servo=='implicit-velocity')
        save(args.output/"receipt.json",receipt)
        steps=round((args.seconds+args.settle)/dt)
        control_every=round(args.control_dt/dt)
        commands=np.zeros((count,24))
        for tick in range(steps):
            q=robot.data.joint_pos.torch; dq=robot.data.joint_vel.torch
            root=robot.data.root_link_pose_w.torch
            gravity=quat_apply_inverse(root[:,3:],gravity0)
            gyro=quat_apply_inverse(root[:,3:],robot.data.root_com_vel_w.torch[:,3:])
            raw_leg_q=q[:,leg_ids].numpy()
            # PhysX continuous revolute coordinates wrap by 4*pi. Reconstruct
            # encoder turns before both compensation and control, without
            # changing any physical state.
            jumps=raw_leg_q-previous_leg_q
            leg_turn_offset-=np.round(jumps/(4.*math.pi))*(4.*math.pi)
            leg_q=raw_leg_q+leg_turn_offset
            previous_leg_q=raw_leg_q.copy()
            enabled=tick*dt>=args.settle
            if tick%control_every==0:
                commands=np.array([c.step(np.r_[leg_q[i],dq[i,leg_ids].numpy(),c.compensation(leg_q[i]),
                    gravity[i].numpy(),gyro[i].numpy(),dq[i,[ids[2],ids[5]]].numpy()],args.control_dt,enabled) for i,c in enumerate(controllers)])
            motor=torch.zeros(count,6)
            if args.motor_servo=='torque':
                motor[:,[0,1,3,4]]=torch.tensor(commands[:,8:12],dtype=torch.float32)
            else:
                gains=np.array([s['controller_settings'][2]for s in starts])[:,None]
                biased_target=commands[:,16:20]+commands[:,20:24]/gains
                robot.set_joint_velocity_target_index(target=torch.tensor(biased_target,dtype=torch.float32),joint_ids=leg_ids)
                native_force_limits[:,leg_ids]=torch.tensor((enabled&(commands[:,15]==0))[:,None]*np.ones((1,4))*40.,dtype=torch.float32)
                robot.root_view.set_dof_max_forces(wp.from_torch(native_force_limits),articulation_indices)
            motor[:,[2,5]]=torch.tensor(commands[:,12:14],dtype=torch.float32)
            actual=wheels.apply_torque(motor,dq[:,ids])
            effort.zero_(); effort[:,ids]=actual
            effort[:,[ids[2],ids[5]]]+=wheels.passive_effort
            if args.friction_integrator=='explicit':
                effort[:,leg_ids]+=args.leg_friction_scale*friction.effort(dq[:,leg_ids])
            else:
                # Implicit secant integration of the SAME regularized Coulomb
                # curve. This coefficient is numerical, not identified viscous
                # friction; native static and dynamic friction stay zero.
                native_friction[:,leg_ids,2]=(args.leg_friction_scale*friction.coulomb
                    /dq[:,leg_ids].abs().clamp_min(friction.band))
                robot.root_view.set_dof_friction_properties(wp.from_torch(native_friction),articulation_indices)
            compression=q.new_tensor([spec["spring_binding"][n]["compression_at_q_zero_m"] for n in springs])-q[:,gas_ids]
            effort[:,gas_ids]=spring.physical_force(compression)
            robot.set_joint_effort_target_index(target=effort)
            robot.write_data_to_sim(); sim.step(render=False); robot.update(dt)
            root=robot.data.root_link_pose_w.torch
            q=robot.data.joint_pos.torch; dq=robot.data.joint_vel.torch
            gravity=quat_apply_inverse(root[:,3:],gravity0)
            gyro=quat_apply_inverse(root[:,3:],robot.data.root_com_vel_w.torch[:,3:])
            poses=robot.data.body_link_pose_w.torch
            wheel_pose=poses[:,wheel_ids]
            wheel_center=wheel_pose[:,:,:3]+quat_apply(wheel_pose[:,:,3:],wheel_centers[None].expand(count,-1,-1))
            wheel_axis=quat_apply(wheel_pose[:,:,3:],wheel_axes[None].expand(count,-1,-1))
            wheel_bottom=(wheel_center[:,:,2]-wheel_radius*(1.-wheel_axis[:,:,2].square()).clamp_min(0.).sqrt()
                -wheel_half_width*wheel_axis[:,:,2].abs()).numpy()
            points0=poses[:,b0,:3]+quat_apply(poses[:,b0,3:],c0[None].expand(count,-1,-1))
            points1=poses[:,b1,:3]+quat_apply(poses[:,b1,3:],c1[None].expand(count,-1,-1))
            gap=(points0-points1).norm(dim=-1).max(-1).values.numpy()
            if not torch.isfinite(poses).all() or not torch.isfinite(dq).all() or gap.max()>.01 \
                    or root[:,2].max()>2. or root[:,2].min()<-.1:
                receipt['invalid_state_diagnostics']=dict(simulation_seconds=(tick+1)*dt,
                    case_names=[s['name']for s in starts],root_height_m=root[:,2].tolist(),loop_gap_m=gap.tolist())
                raise FloatingPointError('Invalid native plant state; inspect collision envelopes and closed chains')
            forces=wp.to_torch(contact.get_contact_force_matrix(dt=dt))[contact_order].reshape(count,len(robot.body_names),-1,3).sum(2).norm(dim=-1).numpy()
            if enabled:
                peaks=np.maximum(peaks,np.abs(dq[:,leg_ids].numpy()))
                active_peaks=np.maximum(active_peaks,np.where((faults==0)[:,None],np.abs(dq[:,leg_ids].numpy()),0.))
                reference_peaks=np.maximum(reference_peaks,np.abs(commands[:,4:8]))
                gaps=np.maximum(gaps,gap); contact_peaks=np.maximum(contact_peaks,forces)
                faults=np.maximum(faults,commands[:,15].astype(int))
                for i in range(count):
                    if first_fault_seconds[i] is None and faults[i]:
                        first_fault_seconds[i]=float(tick*dt-args.settle)
            quiet=(gravity[:,2].numpy()<-.9903)&(np.abs(root[:,2].numpy()-.305)<.03)\
                &(gyro.norm(dim=-1).numpy()<.5)&(robot.data.root_link_lin_vel_w.torch[:,:2].norm(dim=-1).numpy()<.12)\
                &(np.abs(wheel_bottom).max(-1)<=.003)&(np.abs(dq[:,leg_ids].numpy()).max(-1)<.6)&(faults==0)\
                &np.isin(commands[:,14],[4,5])
            stable=np.where(quiet&enabled,stable+dt,0.)
            for i in range(count):
                if completion[i] is None and stable[i]>=.5:
                    completion[i]=float((tick+1)*dt-args.settle)
                    completion_speed[i]=peaks[i].tolist();completion_gap[i]=float(gaps[i])
            row=np.c_[root.numpy(),q.numpy(),dq.numpy(),gravity.numpy(),gyro.numpy(),forces,commands,gap,wheel_bottom,
                robot.data.root_link_lin_vel_w.torch.numpy(),robot.data.root_com_vel_w.torch[:,3:].numpy()]
            trace.append(row.astype(np.float32))
            if tick%max(1,round(.05/dt))==0:
                body_trace.append(poses.numpy().copy()); body_trace_time.append((tick+1)*dt)
            if tick%max(1,round(1./dt))==0:
                for i,s in enumerate(starts):
                    prefix="TraditionalRecovery/"+s["name"]
                    writer.add_scalar(prefix+"/leg_speed_peak_rad_s",peaks[i].max(),tick)
                    writer.add_scalar(prefix+"/phase",commands[i,14],tick)
                    writer.add_scalar(prefix+"/height_m",float(root[i,2]),tick)
                    writer.add_scalar(prefix+"/loop_gap_m",float(gap[i]),tick)
                print(json.dumps(dict(simulation_seconds=round(tick*dt,3),phases=commands[:,14].astype(int).tolist(),
                    tilt_deg=np.rad2deg(np.arccos(np.clip(-gravity[:,2].numpy(),-1,1))).round(1).tolist(),
                    height_m=root[:,2].numpy().round(3).tolist(),speed_peak=peaks.max(-1).round(3).tolist(),
                    completion=completion)),flush=True)
        writer.close()
        np.savez_compressed(args.output/"trace.npz",values=np.asarray(trace),dt_s=dt,
            body_poses=np.asarray(body_trace),body_time_s=body_trace_time)
        for i,s in enumerate(starts):
            receipt["records"].append(dict(name=s["name"],completion_seconds=completion[i],
                final_stable_seconds=float(stable[i]),fault=int(faults[i]),leg_speed_peak_rad_s=peaks[i].tolist(),
                recovery_sequence_completed=bool(completion[i]is not None and max(completion_speed[i])<=3.140001
                    and completion_gap[i]<=.003),
                speed_peak_to_completion_rad_s=completion_speed[i],loop_gap_to_completion_m=completion_gap[i],
                first_fault_seconds=first_fault_seconds[i],
                speed_peak_until_first_fault_rad_s=active_peaks[i].tolist(),
                reference_speed_peak_rad_s=reference_peaks[i].tolist(),loop_gap_peak_m=float(gaps[i]),
                contact_force_peak_n=contact_peaks[i].tolist(),
                final_root_pose=root[i].tolist(),final_joint_positions=dict(zip(robot.joint_names,q[i].tolist())),
                passed=bool(args.leg_friction_scale==1. and completion[i] is not None and stable[i]>=.5
                    and faults[i]==0 and peaks[i].max()<=3.140001 and gaps[i]<=.003)))
        receipt.update(status="completed",all_cases_passed=all(r["passed"] for r in receipt["records"]),
            trace_sha256=digest(args.output/"trace.npz"),elapsed_wall_seconds=time.monotonic()-start_time,
            trace_layout=["root_xyzw7",f"positions{len(robot.joint_names)}",f"velocities{len(robot.joint_names)}",
                "gravity3","gyro3",f"body_contact_force{len(robot.body_names)}","controller24","loop_gap1","wheel_bottom2_m",
                'root_link_linear_velocity_world3','root_angular_velocity_world3'],
            success_rule='0.5s continuous in Capture/Complete: tilt<8deg, height305+/-30mm, gyro<0.5rad/s, translation<0.12m/s, both wheel surfaces within3mm, leg speed<0.6rad/s; plus whole controlled window speed<=3.14rad/s and loop gap<=3mm; CAD contact admission is separate',
            contact_interpretation='Contact force<=2N is a raw diagnostic, not a wheel-liftoff criterion')
        save(args.output/"receipt.json",receipt)
    except BaseException as error:
        exit_code=1
        receipt.update(status='interrupted'if isinstance(error,KeyboardInterrupt)else'failed',error_type=type(error).__name__,error=str(error))
        if 'trace'in locals() and trace:
            np.savez_compressed(args.output/'partial_trace.npz',values=np.asarray(trace),dt_s=args.dt)
            receipt['partial_trace_sha256']=digest(args.output/'partial_trace.npz')
        save(args.output/'receipt.json',receipt)
        print(json.dumps(dict(status='failed',error_type=type(error).__name__,error=str(error))),flush=True)
        raise
    finally:
        for c in controllers: c.close()
        launcher.app.close(exit_code=exit_code)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rmcs",type=Path,default=Path("/home/noir/.codex/worktrees/rmcs-traditional-recovery/RMCS"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--families",nargs="+",choices=["prone","left","right","inverted","front","rear","stand"],default=["prone","left","right","inverted"])
    p.add_argument("--leg-combinations",nargs="+",choices=["ff","rr","fr","rf"],default=["ff","rr","fr","rf"])
    p.add_argument("--knees",nargs=2,type=float,default=[75.,75.])
    for name,value in dict(seconds=30.,settle=0.,dt=.001,speed=1.5,acceleration=3.,velocity_gain=5.,position_gain=12.,
        fold_knee=75.,flip_knee=105.,flip_orbit=0.,sweep_orbit=180.,balance_kp=35.,balance_kd=5.,balance_kv=2.,
        flip_asymmetry=0.,body_friction=.7,base_friction=.7,native_speed_limit=2.8,
        pitch_trim_gain=0.,yaw_rate_gain=.6,integral_gain=50.,integral_limit=30.,
        control_dt=.001,leg_friction_scale=1.,support_gain=1.,cad_contact_offset=.00002).items():
        p.add_argument("--"+name.replace("_","-"),type=float,default=value)
    p.add_argument("--seed",type=int,default=261031)
    p.add_argument("--random-gimbal",action="store_true")
    p.add_argument('--self-collision',action='store_true',help='Native nonadjacent collision-envelope probe; filter joint/loop adjacent bodies and two nominal CAD assembly pairs')
    p.add_argument('--collision-geometry',choices=['model-spec','cad-decomposition'],default='model-spec',
        help='Experimental temporary CAD convex decomposition; mass, inertia and mechanism remain authenticated')
    p.add_argument('--filter-leg-assembly',action='store_true',
        help='Research mask: exclude internal pairs within each leg assembly; body, opposite leg and gimbal contacts remain. Requires separate CAD screening.')
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--variants",type=Path,help="JSON list of named C++ controller parameter overrides")
    p.add_argument('--starts',type=Path,help='Measured passive bank. Preserves all measured residual velocities.')
    p.add_argument('--start-names',nargs='+',help='Named subset of --starts')
    p.add_argument('--friction-integrator',choices=['explicit','implicit-secant'],default='explicit')
    p.add_argument('--motor-servo',choices=['torque','implicit-velocity'],default='implicit-velocity',
        help='Implicit mode integrates the same bounded Kd*(v_cmd-v)+feedforward servo inside PhysX; its bias target is not an extra speed command')
    p.add_argument('--nonwheel-friction-combine',choices=['max','average'],default='max')
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,float) and not math.isfinite(value):p.error(key+' must be finite')
    if not 0.<args.speed<3.14 or not 0.<args.dt<=.001: p.error("Speed must be below3.14;physics dt at most1ms")
    if args.seconds<=0. or args.settle<0. or args.native_speed_limit<=0. or min(args.knees)<40. or max(args.knees)>110.:
        p.error('Invalid duration, settling time, native speed limit or initial knee angle')
    if not 0.<args.cad_contact_offset<=.002:p.error('CAD contact offset must be positive and at most 2mm')
    if not 0.<args.control_dt<=.01 or not math.isclose(args.control_dt/args.dt,round(args.control_dt/args.dt),abs_tol=1e-9):
        p.error('Control dt must be an integer multiple of physics dt')
    if args.leg_friction_scale not in (0.,1.): p.error('Friction scale is 1, or 0 for an explicitly unqualified diagnostic')
    run(args)


if __name__ == "__main__": main()

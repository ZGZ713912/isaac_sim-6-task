#!/usr/bin/env python3
"""Place the current full vehicle near the floor and collect disabled rests.

CPU physics is the default so an existing GPU learner need not be interrupted.
All six chassis motors and both gimbal motors have zero command and zero PD;
gravity, contacts, passive gas springs and wheel losses remain physical.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts"), str(ROOT/"tools")]


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(path)


def main():
    from wheeled_tasks.chassis.fallen_pose_bank import SCHEMA, SETTLE, WINDOW, SHAPES, build_starts, digest, family, group_poses, native_clearance, posture_class, window_rest
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="New directory; existing results are never overwritten")
    p.add_argument("--plan", type=Path, default=ROOT/"contracts/wheel_leg_prone_motion_v4.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--tilt-step-deg", type=float, default=45.)
    p.add_argument("--azimuth-step-deg", type=float, default=45.)
    p.add_argument("--shapes", choices=SHAPES, nargs="+", default=list(SHAPES))
    p.add_argument("--random-placements", type=int, default=256, help="Additional uniformly random whole-vehicle orientations")
    p.add_argument("--seed", type=int, default=261031)
    p.add_argument("--placement-clearance-m", type=float, default=.002)
    p.add_argument("--settle-timeout-s", type=float, default=SETTLE["max_seconds"])
    p.add_argument("--physics-dt-s", type=float, choices=(.0005, .001, .002, .005), default=.001)
    p.add_argument("--self-collision", action="store_true", help="Enable native non-adjacent self-contact for this isolated study")
    p.add_argument("--wheel-convex-segments", type=int, choices=(0, 64), default=64,
        help="0 preserves USD Cylinder;64 uses an explicit inscribed cylinder hull for a collision-geometry diagnostic")
    p.add_argument("--replay-bank",type=Path,help="Gently re-place measured endpoints for a separate passive stability check")
    p.add_argument("--replay-indices",type=int,nargs="+",help="Native record indices from--replay-bank")
    p.add_argument("--placements",type=Path,help="Authenticated, CAD-screened paired side-study starts")
    p.add_argument("--leg-friction",choices=("off","moving","from-placements"),default="off")
    p.add_argument("--fixed-duration",action="store_true",help="Keep every pair in physics for the same requested duration")
    p.add_argument("--trace-per-batch",action="store_true",help="Write independent trace chunks for a larger sweep")
    p.add_argument("--prepare-only", action="store_true", help="Solve starts and provenance without starting Isaac")
    args = p.parse_args()
    if not 1 <= args.num_envs <= 64:
        p.error("--num-envs must be1..64")
    if not math.isfinite(args.settle_timeout_s) or not 2 <= args.settle_timeout_s <= 120:
        p.error("--settle-timeout-s must be2..120")
    if not math.isfinite(args.placement_clearance_m) or not 0 < args.placement_clearance_m <= .01:
        p.error("--placement-clearance-m must be positive and at most.01")
    args.plan = args.plan.resolve()
    import numpy as np
    from scipy.spatial.transform import Rotation
    from wheeled_tasks.chassis.full_vehicle_training import authenticate
    plan = json.loads(args.plan.read_text())
    cfg = json.loads((ROOT/plan["mechanism"]["file"]).read_text())
    authenticate(cfg, ROOT)
    bundle = ROOT/cfg["asset_directory"]
    spec = json.loads((bundle/"model_spec.json").read_text())
    manifest = json.loads((bundle/"manifest.json").read_text())
    base_bank = json.loads((ROOT/plan["pose_bank"]["file"]).read_text())
    if args.placements:
        if args.replay_bank or args.replay_indices or args.leg_friction != "from-placements":
            p.error("Paired placements require --leg-friction from-placements and cannot be replayed")
        from wheeled_tasks.chassis.side_fall_study import validate_placements
        starts = validate_placements(json.loads(args.placements.read_text()),digest(bundle/"model_spec.json"),digest(bundle/"manifest.json"))
        if not args.fixed_duration:
            p.error("Paired placements require --fixed-duration")
    elif args.replay_bank:
        if not args.replay_indices or len(args.replay_indices) != len(set(args.replay_indices)):
            p.error("Replay needs distinct--replay-indices")
        replay = json.loads(args.replay_bank.read_text())
        if (replay["status"] != "completed" or replay["model_spec_sha256"] != digest(bundle/"model_spec.json")
                or replay["asset_manifest_sha256"] != digest(bundle/"manifest.json")):
            p.error("Replay needs a completed bank with this asset identity")
        from build_full_vehicle_training import ResetGeometry, geometry as full_geometry
        geometry = ResetGeometry(spec,bundle)
        starts = []
        for i in args.replay_indices:
            record = replay["records"][i]
            if record["index"] != i or record["status"] not in ("settled","not_settled"):
                p.error("Replay record was mechanically rejected or has a different index")
            pose = record["final_pose"]
            rotation = Rotation.from_quat(pose["root_quaternion_xyzw"])
            minimum = min(v[2] for _,v in geometry.bounds(pose["full_joint_positions"],rotation.as_matrix()))
            starts.append(dict(name=f"replay_record_{i:05d}",sampling="measured_endpoint_replacement",
                source_record=i,leg_shape="measured",gimbal_yaw_rad=pose["full_joint_positions"]["M_joint1"],
                quaternion_xyzw=pose["root_quaternion_xyzw"],spawn_family=family(pose["gravity_body"]),
                root_position_m=[0.,0.,-minimum+args.placement_clearance_m],full_joint_positions=pose["full_joint_positions"],
                min_collision_clearance_m=args.placement_clearance_m,
                initial_loop_gap_m=float(np.linalg.norm(full_geometry.closure_error(spec,pose["full_joint_positions"]).reshape(-1,3),axis=1).max())))
    else:
        if args.replay_indices:
            p.error("--replay-indices requires--replay-bank")
        starts = build_starts(spec, bundle, base_bank, args.shapes, tilt_step_deg=args.tilt_step_deg,
            azimuth_step_deg=args.azimuth_step_deg, clearance_m=args.placement_clearance_m,
            random_count=args.random_placements, seed=args.seed)
    if args.leg_friction == "from-placements" and not args.placements:
        p.error("--leg-friction from-placements needs --placements")
    from wheeled_tasks.chassis.leg_motor_friction import declaration, load as load_leg_friction
    friction_cfg = declaration(ROOT) if args.leg_friction != "off" else dict(enabled=False)
    _, friction_model = load_leg_friction(friction_cfg,ROOT)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source_names = [str(Path(__file__).relative_to(ROOT)), "scripts/build_full_vehicle_training.py",
        "tools/full_robot_v202.py", "src/wheeled_tasks/chassis/fallen_pose_bank.py",
        "src/wheeled_tasks/chassis/v5_control.py", "src/wheeled_tasks/chassis/gas_spring_compensation.py",
        "src/wheeled_tasks/chassis/identified_wheel_response.py", "src/wheeled_tasks/chassis/full_robot_v202.py",
        "src/wheeled_tasks/chassis/full_vehicle_training.py", "src/wheeled_tasks/chassis/control_contract.py",
        "src/wheeled_tasks/chassis/gimbal_motion.py", plan["mechanism"]["file"]]
    if args.plan.is_relative_to(ROOT):
        source_names.append(str(args.plan.relative_to(ROOT)))
    source_names += ["src/wheeled_tasks/chassis/leg_motor_friction.py","src/wheeled_tasks/chassis/side_fall_study.py"]
    if friction_model:
        source_names += [friction_cfg["model_path"]]+[str(Path(friction_cfg["model_path"]).parent/n) for n in friction_model["files_sha256"]]
    sources = {}
    for name in source_names:
        source = ROOT/name
        if source.is_file():
            sources[name] = digest(source)
            destination = args.output/"source"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    report = dict(schema=SCHEMA, status="prepared", started_utc=datetime.now(timezone.utc).isoformat(),
        device=args.device, physics_dt_s=args.physics_dt_s, solver_iterations=[64, 32],
        asset_directory=cfg["asset_directory"], asset_manifest_sha256=digest(bundle/"manifest.json"),
        model_spec_sha256=digest(bundle/"model_spec.json"), plan_sha256=digest(args.plan),
        tree_joint_names=manifest["tree_joint_names"], source_hashes=sources,
        placement_clearance_m=args.placement_clearance_m, settle_contract=SETTLE,
        actual_timeout_s=args.settle_timeout_s,
        window_rest_contract=WINDOW,
        random_placements=args.random_placements, seed=args.seed,
        temporary_nonwheel_friction=.7, body_material_combine_mode="max", gimbal_initial_yaw="uniform(-pi,pi)",
        all_eight_motor_servos_disabled=True, motor_effort_peak_nm=0.,
        wheel_collision_segments=args.wheel_convex_segments,
        wheel_hull_maximum_radial_deficit_m=(.06*(1.-math.cos(math.pi/args.wheel_convex_segments)) if args.wheel_convex_segments else 0.),
        passive_gas_springs="retained", passive_wheel_losses="retained_once",
        gimbal="both axes passive, including pitch stops; no servo or pose hold",
        state_writes_during_physics=0, ppo_updates=0, hardware_motion=False,
        self_collision_scope="native non-adjacent self-contact enabled" if args.self_collision else "inherited disabled self-collision; endpoints require a later CAD self-contact audit",
        coverage_scope="finite placement sweep on a homogeneous flat floor; not exhaustive falls or recovery qualification",
        behavior_qualified=False, requested_cases=len(starts), completed_cases=0,
        starts=starts, records=[], poses=[], family_counts={})
    report.update(leg_friction_mode=args.leg_friction,leg_motor_friction=friction_cfg,
        fixed_duration=args.fixed_duration,leg_friction_passive_power_peak_w=0.)
    if args.placements:
        report.update(placements_path=str(args.placements.resolve()),placements_sha256=digest(args.placements),
            random_placements=0,gimbal_initial_yaw="randomized per leg configuration;identical across each pair")
    path = args.output/"bank.json"
    if args.replay_bank:
        report.update(replay_bank=str(args.replay_bank.resolve()),replay_bank_sha256=digest(args.replay_bank),
            random_placements=0,gimbal_initial_yaw="measured endpoint yaw retained for replacement verification;original sweep randomized yaw")
    save(path, report)
    if args.prepare_only:
        print(json.dumps(dict(status="prepared", cases=len(starts), output=str(path))), flush=True)
        return
    launcher = None
    beginning = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False))
        import torch
        import warp as wp
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
        import isaaclab.sim as su
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab_physx.physics import PhysxCfg, PhysxManager
        from isaaclab.utils.math import quat_apply
        from wheeled_tasks.chassis.full_robot_v202 import FullRobotV202
        from wheeled_tasks.chassis.identified_wheel_response import IdentifiedWheelResponse
        from wheeled_tasks.chassis.leg_motor_friction import LegMotorFriction, JOINTS as LEG_JOINTS
        from wheeled_tasks.chassis.fallen_pose_bank import rest_mask, chassis_rest_mask
        from wheeled_tasks.chassis.v5_control import V5Control
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        torch.set_num_threads(4)
        dt = args.physics_dt_s
        sample_steps, log_steps = round(.01/dt), round(1./dt)
        count = min(args.num_envs, len(starts))
        parent = FullRobotV202()
        prior = load_chassis_control(ROOT/cfg["control_math_source"])
        fit = json.loads((ROOT/"model/纯底盘_v7_flat_base107/urdf/fit_10mpa.json").read_text())
        control = V5Control(manifest, spec, fit, prior, cfg["v5_control"], args.device, root=ROOT)
        wheel_cfg = dict(enabled=True, model_path=str((bundle/"wheel_model.json").relative_to(ROOT)),
            model_sha256=digest(bundle/"wheel_model.json"), apply_aggregate_delay=True)
        response = IdentifiedWheelResponse(count, args.device, dt, wheel_cfg, root=ROOT)
        leg_friction = LegMotorFriction(count,args.device,friction_cfg,root=ROOT)
        native = response.native_parameters()
        sim = su.SimulationContext(su.SimulationCfg(device=args.device, dt=dt, gravity=(0., 0., -9.81),
            render_interval=1000, physics=PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)))
        import carb.settings
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        floor = su.CuboidCfg(size=(40., 40., .1), collision_props=su.CollisionPropertiesCfg(), physics_material=su.RigidBodyMaterialCfg(
            static_friction=plan["friction"]["ground_nominal"], dynamic_friction=plan["friction"]["ground_nominal"],
            restitution=0., friction_combine_mode="average"))
        floor.func("/World/Ground", floor, translation=(14., 14., -.05))
        origins = []
        for i in range(count):
            origin = (float(i%8)*4., float(i//8)*4., 0.)
            origins.append(origin)
            UsdGeom.Xform.Define(sim.stage, f"/World/envs/env_{i}").AddTranslateOp().Set(origin)
        active, springs = manifest["control_joint_names"], manifest["spring_joint_names"]
        gimbals = ["M_joint1", "M_joint2"]
        passive = [n for n in manifest["tree_joint_names"] if n not in active+springs+gimbals]
        def actuator(names, cap, damping=0., **kwargs):
            return IdealPDActuatorCfg(joint_names_expr=names, stiffness=0., damping=damping,
                effort_limit=cap, effort_limit_sim=cap, velocity_limit_sim=1e9, **kwargs)
        robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_[0-9]+/Robot",
            spawn=su.UsdFileCfg(usd_path=str(bundle/manifest["usd"]), activate_contact_sensors=True,
                rigid_props=su.RigidBodyPropertiesCfg(max_depenetration_velocity=1.),
                articulation_props=su.ArticulationRootPropertiesCfg(enabled_self_collisions=args.self_collision,
                    solver_position_iteration_count=64, solver_velocity_iteration_count=32)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 1.),
                joint_pos=manifest["nominal_joint_pos"], joint_vel={".*": 0.}),
            actuators=dict(legs=actuator([active[i] for i in (0, 1, 3, 4)],
                    leg_friction.native_effort_limit if leg_friction.enabled else 40., friction=0.,dynamic_friction=0.,viscous_friction=0.),
                wheels=actuator([active[2], active[5]], response.native_effort_limit,
                    armature=native["armature"], friction=native["frictionloss"],
                    dynamic_friction=native["dynamic_friction"], viscous_friction=native["damping"]),
                gimbals=actuator(gimbals, 10.), passive=actuator(passive, 100., .002), gas=actuator(springs, 1000., .002)),
            soft_joint_pos_limit_factor=1.))
        for name, coefficient, combine in (("Body", .7, "max"), ("Wheel", plan["friction"]["wheel"], "average")):
            material = su.RigidBodyMaterialCfg(static_friction=coefficient, dynamic_friction=coefficient,
                restitution=0., friction_combine_mode=combine)
            material.func("/World/"+name+"Material", material)
        loops = 0
        adjacent = {b["name"]: set() for b in spec["bodies"]}
        for j in spec["joints"]:
            adjacent[j["parent"]].add(j["child"]); adjacent[j["child"]].add(j["parent"])
        for c in spec["constraints"]:
            adjacent[c["body0"]].add(c["body1"]); adjacent[c["body1"]].add(c["body0"])
        for prim in sim.stage.Traverse():
            if "/Robot/" not in str(prim.GetPath()):
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
                if args.self_collision:
                    base = str(prim.GetPath()).rsplit("/", 1)[0]
                    UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(
                        [base+"/"+n for n in sorted(adjacent[prim.GetName()])])
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                s = str(prim.GetPath())
                material = "Wheel" if any("/"+n+"/" in s for n in ("L_link3", "R_link3")) else "Body"
                if args.wheel_convex_segments and material == "Wheel" and prim.IsA(UsdGeom.Cylinder):
                    cylinder = UsdGeom.Cylinder(prim)
                    radius, height = cylinder.GetRadiusAttr().Get(), cylinder.GetHeightAttr().Get()
                    n = args.wheel_convex_segments
                    points = [Gf.Vec3f(radius*math.cos(2*math.pi*k/n), radius*math.sin(2*math.pi*k/n), z)
                        for z in (-height/2, height/2) for k in range(n)]
                    mesh = UsdGeom.Mesh.Define(sim.stage, prim.GetPath())
                    mesh.CreatePointsAttr(points)
                    faces = [[k, (k+1)%n, (k+1)%n+n, k+n] for k in range(n)]
                    faces += [list(reversed(range(n))), list(range(n, 2*n))]
                    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
                    mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
                    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
                    PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh.GetPrim()).CreateHullVertexLimitAttr(256)
                su.bind_physics_material(s, "/World/"+material+"Material")
            if prim.IsA(UsdPhysics.SphericalJoint):
                if not UsdPhysics.Joint(prim).GetExcludeFromArticulationAttr().Get():
                    raise ValueError("Loop was included in articulation tree")
                loops += 1
        if loops != 6*count:
            raise ValueError("Unexpected external loop count")
        sim.reset(); robot.update(0.)
        if set(robot.joint_names) != set(manifest["tree_joint_names"]):
            raise ValueError("Unexpected native joint set")
        order = [int(p.split("/env_")[1].split("/")[0]) for p in robot.root_view.prim_paths]
        origins = torch.tensor([origins[i] for i in order], device=args.device, dtype=torch.float32)
        ids = [robot.joint_names.index(n) for n in active]
        leg_ids = [robot.joint_names.index(n) for n in LEG_JOINTS]
        if leg_friction.enabled:
            native_leg_loss = wp.to_torch(robot.root_view.get_dof_friction_properties())[:,leg_ids]
            if bool((native_leg_loss != 0.).any()):
                raise ValueError("Identified leg friction would overlap native leg friction/damping")
            report["leg_motor_friction_summary"] = leg_friction.summary()
            report["leg_native_friction_properties"] = native_leg_loss.cpu().tolist()
        gas_ids = [robot.joint_names.index(n) for n in springs]
        motor_ids = ids+[robot.joint_names.index(n) for n in gimbals]
        slider_ids = [robot.joint_names.index(j["name"]) for j in spec["joints"] if j["type"] == "prismatic"]
        view = PhysxManager.get_physics_sim_view()
        body_paths = [p.split("/Robot")[0]+"/Robot/"+n for p in robot.root_view.prim_paths for n in robot.body_names]
        # Validate the real floor path rather than treating a missing filter as no support.
        ground_colliders = [str(p.GetPath()) for p in sim.stage.Traverse()
            if str(p.GetPath()).startswith("/World/Ground/") and p.HasAPI(UsdPhysics.CollisionAPI)]
        contact = view.create_rigid_contact_view(body_paths, filter_patterns=[ground_colliders for _ in body_paths],
            max_contact_data_count=64*len(body_paths))
        if not ground_colliders or contact.filter_count != len(ground_colliders):
            raise ValueError("Missing terrain contact filter")
        masses = wp.to_torch(robot.root_view.get_masses()).to(args.device)
        native_materials = wp.to_torch(robot.root_view.get_material_properties()).to(args.device)
        if not torch.allclose(native_materials[..., :2], torch.full_like(native_materials[..., :2], .7), atol=1e-6, rtol=0.):
            raise ValueError("Temporary0.7 body materials did not reach native PhysX")
        weight = masses.sum(-1)*9.81
        c0 = torch.tensor([c["local_pos0_m"] for c in spec["constraints"]], device=args.device)
        c1 = torch.tensor([c["local_pos1_m"] for c in spec["constraints"]], device=args.device)
        b0 = [robot.body_names.index(c["body0"]) for c in spec["constraints"]]
        b1 = [robot.body_names.index(c["body1"]) for c in spec["constraints"]]
        if not torch.allclose(masses.sum(-1), torch.full_like(weight, manifest["total_mass_kg"]), atol=1e-5, rtol=0.):
            raise ValueError("Native mass differs from authenticated full vehicle")
        report.update(status="running", native=dict(body_names=robot.body_names, joint_names=robot.joint_names,
            environment_order=order,environment_origins_world_m=origins.cpu().tolist(),
            external_loops=loops, masses_kg=masses.cpu().tolist(), floor_collision_paths=ground_colliders,
            body_friction=.7, body_friction_combine_mode="max", wheel_friction=plan["friction"]["wheel"],
            ground_friction=plan["friction"]["ground_nominal"], base_guide_friction=.7,
            material_static_dynamic_restitution=native_materials.cpu().tolist(),
            parent_manifest_sha256=digest(parent.bundle/"manifest.json")),
            versions={n: importlib.metadata.version(n) for n in ("isaacsim", "isaaclab", "isaaclab-physx", "torch")})
        from build_full_vehicle_training import ResetGeometry, geometry as full_geometry
        geometry = ResetGeometry(spec, bundle)
        chassis_ids = [i for i,n in enumerate(robot.body_names) if n not in ("M_link1", "M_link2")]
        report["trace_columns"] = ["root_x", "root_y", "root_z", "quat_x", "quat_y", "quat_z", "quat_w"]+[
            "q/"+n for n in robot.joint_names]+["dq/"+n for n in robot.joint_names]+[
            "body_linear_speed_max_m_s", "body_angular_speed_max_rad_s"]+["contact_n/"+n for n in robot.body_names]+[
            "gap_m", "static_hold_s", "chassis_static_hold_s", "support_weight_fraction",
            "chassis_linear_speed_max_m_s", "chassis_angular_speed_max_rad_s"]
        if leg_friction.enabled:
            report["trace_columns"] += ["contact_z_n/"+n for n in robot.body_names]+["leg_loss_nm/"+n for n in LEG_JOINTS]
        save(path, report)
        all_trace = []
        report["trace_files"] = []
        for offset in range(0, len(starts), count):
            batch = starts[offset:offset+count]
            valid_count = len(batch)
            batch = batch+[batch[-1]]*(count-valid_count)
            q0 = torch.tensor([[s["full_joint_positions"][n] for n in robot.joint_names] for s in batch], device=args.device, dtype=torch.float32)
            roots = torch.tensor([s["root_position_m"]+s["quaternion_xyzw"] for s in batch], device=args.device, dtype=torch.float32)
            roots[:, :3] += origins
            robot.reset(); response.reset()
            robot.write_root_link_pose_to_sim_index(root_pose=roots)
            robot.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(count, 6, device=args.device))
            robot.write_joint_position_to_sim_index(position=q0)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
            robot.update(0.)
            hold = torch.zeros(count, device=args.device)
            chassis_hold = torch.zeros(count, device=args.device)
            bad = torch.zeros(count, dtype=torch.bool, device=args.device)
            gap_peak = torch.zeros(count, device=args.device)
            motor_peak = torch.zeros_like(hold)
            leg_mask = torch.tensor([s.get("leg_friction_enabled",args.leg_friction == "moving") for s in batch],device=args.device)[:,None]
            leg_loss = torch.zeros(count,4,device=args.device)
            leg_energy = torch.zeros_like(hold)
            leg_peak = torch.zeros_like(hold)
            trace = []
            tau0 = torch.zeros(count, 6, device=args.device)
            effort = torch.zeros_like(q0)
            for tick in range(round(args.settle_timeout_s/dt)):
                q, dq = robot.data.joint_pos.torch, robot.data.joint_vel.torch
                motor = response.apply_torque(tau0, dq[:, ids])
                if motor.abs().max() != 0.:
                    raise ValueError("Disabled collector produced a motor command")
                effort.zero_()
                effort[:, [ids[2], ids[5]]] = response.passive_effort
                if leg_friction.enabled:
                    leg_loss.copy_(leg_friction.effort(dq[:,leg_ids])*leg_mask)
                    power = leg_loss*dq[:,leg_ids]
                    if bool((power > 1e-6).any()):
                        raise ValueError("Passive leg loss injected positive power at its evaluation velocity")
                    leg_energy -= power.sum(-1)*dt
                    leg_peak = torch.maximum(leg_peak,leg_loss.abs().amax(-1))
                    effort[:,leg_ids] = leg_loss
                effort[:, gas_ids] = control.spring_efforts(q[:, gas_ids])
                robot.set_joint_effort_target_index(target=effort)
                robot.write_data_to_sim(); sim.step(render=False); robot.update(dt)
                pos, quat = robot.data.body_pos_w.torch, robot.data.body_quat_w.torch
                gap = (pos[:, b0]+quat_apply(quat[:, b0], c0[None].expand(count, -1, -1))
                       -pos[:, b1]-quat_apply(quat[:, b1], c1[None].expand(count, -1, -1))).norm(dim=-1).amax(-1)
                gap_peak = torch.maximum(gap_peak, gap)
                bad |= ~torch.isfinite(robot.data.joint_pos.torch).all(-1) | (gap > SETTLE["loop_gap_m"])
                # Applied wheel effort includes separately modelled passive loss.
                # Subtract that exact loss to audit zero motor torque downstream of actuators.
                applied = robot.data.applied_torque.torch[:, motor_ids].clone()
                applied[:, [2, 5]] -= response.passive_effort
                applied[:, [0,1,3,4]] -= leg_loss
                motor_peak = torch.maximum(motor_peak, applied.abs().amax(-1))
                if tick%sample_steps != sample_steps-1:
                    continue
                force = wp.to_torch(contact.get_contact_force_matrix(dt=dt)).reshape(count, len(robot.body_names), -1, 3).sum(2)
                body_vel = robot.data.body_com_vel_w.torch
                hold = torch.where(rest_mask(body_vel, robot.data.joint_vel.torch, slider_ids, force[:, :, 2].sum(-1), weight),
                    hold+dt*sample_steps, 0.)
                chassis_hold = torch.where(chassis_rest_mask(body_vel, robot.data.joint_vel.torch,
                    robot.body_names, robot.joint_names, force[:, :, 2].sum(-1), weight), chassis_hold+dt*sample_steps, 0.)
                root = robot.data.root_link_pose_w.torch.clone(); root[:, :3] -= origins
                sample = torch.cat((root, robot.data.joint_pos.torch, robot.data.joint_vel.torch,
                    body_vel[..., :3].norm(dim=-1).amax(-1)[:, None],
                    body_vel[..., 3:].norm(dim=-1).amax(-1)[:, None],
                    force.norm(dim=-1), gap[:, None], hold[:, None], chassis_hold[:, None],
                    (force[:, :, 2].sum(-1)/weight)[:, None],
                    body_vel[:, chassis_ids, :3].norm(dim=-1).amax(-1)[:, None],
                    body_vel[:, chassis_ids, 3:].norm(dim=-1).amax(-1)[:, None]), -1)
                if leg_friction.enabled:
                    sample = torch.cat((sample,force[:,:,2],leg_loss),-1)
                trace.append(sample.cpu().numpy())
                elapsed = (tick+1)*dt
                if tick%log_steps == log_steps-1:
                    print(json.dumps(dict(batch_start=offset, passive_s=elapsed,
                        support_weight_fraction=(force[:, :, 2].sum(-1)/weight).cpu().tolist(),
                        net_contact_peak_n=float(wp.to_torch(contact.get_net_contact_forces(dt=dt)).abs().max()),
                        static_hold_s=hold.cpu().tolist())), flush=True)
                if not args.fixed_duration and elapsed >= SETTLE["min_seconds"] and bool(((hold >= SETTLE["hold_seconds"]) | bad).all()):
                    break
            # No freezing/repositioning is used when one lane settles before its neighbours.
            root = robot.data.root_link_pose_w.torch.clone(); root[:, :3] -= origins
            q, dq = robot.data.joint_pos.torch, robot.data.joint_vel.torch
            all_trace.append(np.stack(trace)[:, :valid_count])
            for i in range(valid_count):
                index = offset+i
                quaternion = root[i, 3:].cpu().numpy().astype(float)
                quaternion /= np.linalg.norm(quaternion)
                rotation = Rotation.from_quat(quaternion)
                values = dict(zip(robot.joint_names, q[i].cpu().tolist()))
                bodies = robot.data.body_link_pose_w.torch[i].cpu().numpy().astype(float).copy()
                bodies[:, :3] -= origins[i].cpu().numpy()
                bodies[:, :2] -= root[i, :2].cpu().numpy()
                clearance = native_clearance(spec, geometry.meshes, robot.body_names, bodies)
                frame = np.eye(4); frame[:3, :3] = rotation.as_matrix(); frame[2, 3] = float(root[i, 2])
                predicted = full_geometry.fk(spec, values, frame)
                fk_error = max(float(np.linalg.norm(predicted[n][:3, 3]-bodies[k, :3])) for k,n in enumerate(robot.body_names))
                support = [n for n, f in zip(robot.body_names, force[i, :, 2].cpu().tolist()) if f > 2.]
                limits_ok = True
                for j in spec["joints"]:
                    if j["type"] in ("prismatic", "revolute") and "lower" in j["limit"]:
                        tolerance = 2e-4 if j["type"] == "prismatic" else .003
                        limits_ok &= float(j["limit"]["lower"])-tolerance <= values[j["name"]] <= float(j["limit"]["upper"])+tolerance
                status = ("mechanical_guard" if bool(bad[i]) else "motor_not_disabled" if float(motor_peak[i]) > 1e-6
                    else "joint_limit" if not limits_ok else "penetration" if clearance < -SETTLE["ground_penetration_m"]
                    else "settled" if float(hold[i]) >= SETTLE["hold_seconds"] else "not_settled")
                gravity = rotation.inv().apply([0., 0., -1.]).tolist()
                # Express roll/pitch for compatibility; measured family uses the gravity vector.
                roll, pitch, _ = rotation.as_euler("xyz", degrees=True)
                zero = spec["passive_knee_zero_inner_rad"]
                final = dict(name=f"passive_rest_{index:04d}", family=family(gravity),
                    roll_deg=float(roll), pitch_deg=float(pitch), gravity_body=gravity,
                    root_position_m=[0., 0., float(root[i, 2])], root_quaternion_xyzw=quaternion.tolist(),
                    full_joint_positions=values, full_joint_velocities=dict(zip(robot.joint_names, dq[i].cpu().tolist())),
                    native_body_poses_world=dict(zip(robot.body_names, bodies.tolist())),
                    native_body_com_velocities_world=dict(zip(robot.body_names, body_vel[i].cpu().tolist())),
                    reconstructed_fk_position_error_m=fk_error,
                    body_linear_speed_max_m_s=float(body_vel[i, :, :3].norm(dim=-1).max()),
                    body_angular_speed_max_rad_s=float(body_vel[i, :, 3:].norm(dim=-1).max()),
                    support_weight_fraction=float(force[i, :, 2].sum()/weight[i]),
                    root_com_velocity_world=robot.data.body_com_vel_w.torch[i, robot.body_names.index("base_link")].cpu().tolist(),
                    knee_deg=[math.degrees(values["L_joint2"]+zero["L_joint2"]), math.degrees(zero["R_jonit2"]-values["R_jonit2"])],
                    support_bodies=sorted(support), support_force_world_n=force[i].cpu().tolist(),
                    gap_m=float(gap[i]), min_collision_clearance_m=clearance,
                    stable_hold_seconds=float(hold[i]), native_settled=status == "settled",
                    chassis_stable_hold_seconds=float(chassis_hold[i]),
                    chassis_settled=(status in ("settled", "not_settled") and float(chassis_hold[i]) >= SETTLE["hold_seconds"]),
                    collector_record_index=index, reset_velocity="zero; measured residual velocities retained above")
                final["posture_class"] = posture_class(final)
                final["window_rest"] = window_rest(all_trace[-1][:, i], report["trace_columns"], robot.joint_names)
                final["chassis_window_rest"] = window_rest(all_trace[-1][:, i], report["trace_columns"], robot.joint_names, chassis_only=True)
                final["window_rest"]["mechanically_valid"] = status in ("settled", "not_settled")
                final["chassis_window_rest"]["mechanically_valid"] = status in ("settled", "not_settled")
                report["records"].append(dict(index=index, spawn=starts[index]["name"], spawn_family=starts[index]["spawn_family"],
                    status=status, elapsed_s=elapsed, loop_gap_peak_m=float(gap_peak[i]),
                    motor_effort_peak_nm=float(motor_peak[i]), final_pose=final,
                    leg_friction_enabled=bool(leg_mask[i,0]),leg_loss_peak_nm=float(leg_peak[i]),
                    leg_loss_dissipation_j=float(leg_energy[i]),physics_origin_world_m=origins[i].cpu().tolist()))
            report["completed_cases"] += valid_count
            report["motor_effort_peak_nm"] = max(report["motor_effort_peak_nm"], float(motor_peak[:valid_count].max()))
            save(path, report)
            if args.trace_per_batch:
                trace_path = args.output/f"trace_batch_{offset//count:03d}.npz"
                np.savez_compressed(trace_path,trace=all_trace[-1])
                report["trace_files"].append(dict(path=trace_path.name,sha256=digest(trace_path),
                    first_record=offset,count=valid_count))
                all_trace.clear()
                save(path,report)
            else:
                np.savez_compressed(args.output/"traces.npz", **{f"batch_{i}": a for i, a in enumerate(all_trace)})
            print(json.dumps(dict(completed=report["completed_cases"], total=len(starts),
                settled=sum(r["status"] == "settled" for r in report["records"]), elapsed_wall_s=round(time.monotonic()-beginning, 1))), flush=True)
        groups = group_poses(report["records"], robot.joint_names)
        report["groups"] = groups
        report["poses"] = [dict(report["records"][g["representative_record"]]["final_pose"],
            group_record_indices=g["record_indices"]) for g in groups]
        report["family_counts"] = {f: sum(p["family"] == f for p in report["poses"])
            for f in ("upright", "front", "back", "left", "right", "inverted", "compound")}
        report["outcome_counts"] = {s: sum(r["status"] == s for r in report["records"]) for s in sorted({r["status"] for r in report["records"]})}
        report.update(status="completed", elapsed_wall_s=time.monotonic()-beginning)
        if not args.trace_per_batch:
            report["trace_sha256"] = digest(args.output/"traces.npz")
        save(path, report)
        print(json.dumps(dict(status=report["status"], outcomes=report["outcome_counts"], groups=len(groups),
            families=report["family_counts"], bank=str(path))), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), elapsed_wall_s=time.monotonic()-beginning)
        save(path, report)
        raise
    finally:
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()

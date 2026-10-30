#!/usr/bin/env python3
"""Native full-vehicle component probe; no policy, PPO or hardware IO."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "model/wheel_leg_v2.0.2/isaac")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000, help="1 ms steps per reset case")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps < 300:
        parser.error("Use at least 300 physical steps per case")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "probe.source.py").write_bytes(Path(__file__).read_bytes())
    import numpy as np
    from scipy.spatial.transform import Rotation
    from full_robot_v202 import ACTIVE, GIMBAL, closure_error, solve_pose
    from wheeled_tasks.chassis.full_robot_v202 import FullRobotV202, digest
    asset = FullRobotV202(args.bundle)
    manifest, spec, runtime = asset.manifest, asset.spec, asset.runtime
    report = dict(status="prepared", started_at=datetime.now(timezone.utc).isoformat(),
        fixture="fixed root, zero gravity, no floor/contact; unrestrained closed-chain legs",
        physics_dt_s=.001, policy_dt_s=.01, leg_motor_armature_kg_m2=0.,
        reset_pose_writes_per_case=1, passive_pose_writes_during_physics=0,
        ppo_updates=0, hardware_motion=False, behavior_qualified=False, device=args.device,
        asset_manifest_sha256=digest(args.bundle / "manifest.json"),
        model_spec_sha256=asset.spec_sha256, script_sha256=digest(__file__),
        versions={n:importlib.metadata.version(n) for n in ("isaacsim", "isaaclab", "isaaclab-physx", "torch")},
        cases=[])
    path = args.output / "report.json"
    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    save()
    started, launcher = time.monotonic(), None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False))
        import torch
        import warp as wp
        from pxr import UsdPhysics
        import isaaclab.sim as su
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab.utils.math import quat_apply
        torch.set_num_threads(4)
        control, response = asset.components(1, args.device)
        observer = asset.encoder_geometry(args.device)
        native = response.native_parameters()
        sim = su.SimulationContext(su.SimulationCfg(device=args.device, dt=.001, gravity=(0.,0.,0.),
            render_interval=100, physics=PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)))
        active, springs = manifest["control_joint_names"], manifest["spring_joint_names"]
        wheel_names = [active[2], active[5]]
        leg_names = [active[i] for i in control.LEGS]
        passive = [n for n in manifest["tree_joint_names"] if n not in active+springs]
        def actuator(names, cap, damping=0., **kwargs):
            return IdealPDActuatorCfg(joint_names_expr=names, stiffness=0., damping=damping,
                effort_limit=cap, effort_limit_sim=cap, velocity_limit_sim=100., **kwargs)
        robot = Articulation(ArticulationCfg(prim_path="/World/Robot",
            spawn=su.UsdFileCfg(usd_path=str(args.bundle / "robot.usda"),
                articulation_props=su.ArticulationRootPropertiesCfg(fix_root_link=True,
                    enabled_self_collisions=False, solver_position_iteration_count=64, solver_velocity_iteration_count=32)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,2.),
                joint_pos=manifest["nominal_joint_pos"], joint_vel={".*":0.}),
            actuators=dict(legs=actuator(leg_names,40.,armature=0.),
                wheels=actuator(wheel_names,response.native_effort_limit,armature=native["armature"],
                    friction=native["frictionloss"],dynamic_friction=native["dynamic_friction"],
                    viscous_friction=native["damping"]),
                passive=actuator(passive,100.,damping=.002), gas=actuator(springs,1500.,damping=.002)),
            soft_joint_pos_limit_factor=1.))
        loops = [p for p in sim.stage.Traverse() if p.IsA(UsdPhysics.SphericalJoint)]
        fixed = [p for p in sim.stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)
                 and p.GetName() in GIMBAL]
        if len(loops) != 6 or len(fixed) != 2:
            raise ValueError("Native topology must have six external loops and two fixed gimbal joints")
        if not all(UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get() for p in loops):
            raise ValueError("Closed loops must remain outside the articulation tree")
        sim.reset()
        robot.update(0.)
        if set(robot.joint_names) != set(manifest["tree_joint_names"]):
            raise ValueError(f"Native DOF mismatch: {robot.joint_names}")
        if set(robot.body_names) != set(manifest["rigid_body_names"]):
            raise ValueError(f"Native rigid body mismatch: {robot.body_names}")
        ids = [robot.joint_names.index(n) for n in active]
        spring_ids = [robot.joint_names.index(n) for n in springs]
        wheel_ids = [robot.joint_names.index(n) for n in wheel_names]
        knee_ids = [robot.joint_names.index(n) for n in ("L_joint2","R_jonit2")]
        as_numpy = lambda value: wp.to_torch(value).cpu().numpy()[0]
        masses = as_numpy(robot.root_view.get_masses())
        coms = as_numpy(robot.root_view.get_coms())
        inertias = as_numpy(robot.root_view.get_inertias()).reshape(-1,3,3)
        armatures = as_numpy(robot.root_view.get_dof_armatures())
        friction = as_numpy(robot.root_view.get_dof_friction_properties())
        limits = as_numpy(robot.root_view.get_dof_limits())
        max_mass_error = max_com_error = max_inertia_error = 0.
        by_name = {b["name"]:b for b in spec["bodies"]}
        for i, name in enumerate(robot.body_names):
            b = by_name[name]
            rotation = Rotation.from_quat(coms[i,3:]).as_matrix()
            tensor = rotation @ inertias[i] @ rotation.T
            max_mass_error = max(max_mass_error,abs(float(masses[i])-b["mass"]))
            max_com_error = max(max_com_error,float(abs(coms[i,:3]-b["com"]).max()))
            max_inertia_error = max(max_inertia_error,float(abs(tensor-b["inertia"]).max()))
        if max_mass_error > 3e-6 or max_com_error > 2e-7 or max_inertia_error > 3e-7:
            raise ValueError(f"Native CAD mass properties differ: {max_mass_error}, {max_com_error}, {max_inertia_error}")
        for n in wheel_names:
            k = robot.joint_names.index(n)
            expected = [native[key][n] for key in ("frictionloss","dynamic_friction","damping")]
            if abs(float(armatures[k])-native["armature"][n]) > 1e-8 or not np.allclose(friction[k],expected,atol=1e-7,rtol=0.):
                raise ValueError("Measured wheel armature/friction did not reach PhysX")
        expected_limits = [[float(j["limit"][key]) for key in ("lower","upper")]
                           for name in ("L_joint2","R_jonit2",*springs)
                           for j in spec["joints"] if j["name"]==name]
        actual_limits = limits[knee_ids+spring_ids]
        if not np.allclose(actual_limits,expected_limits,atol=2e-7,rtol=0.):
            raise ValueError("Passive knee or gas-spring stroke limits differ")
        report["native_readback"] = dict(body_names=robot.body_names, joint_names=robot.joint_names,
            body_masses_kg=masses.tolist(), total_mass_kg=float(masses.sum()),
            max_mass_error_kg=max_mass_error,max_com_error_m=max_com_error,max_inertia_error_kg_m2=max_inertia_error,
            body_com_poses_xyzw=coms.tolist(),body_inertias_mass_frame=inertias.tolist(),
            armatures_kg_m2=armatures.tolist(),friction_static_dynamic_viscous=friction.tolist(),
            knee_and_spring_limits=actual_limits.tolist(),external_loops=len(loops),fixed_gimbal_joints=len(fixed))
        report["wheel_response"] = response.summary()
        save()
        c0 = torch.tensor([c["local_pos0_m"] for c in spec["constraints"]],device=args.device,dtype=torch.float32)
        c1 = torch.tensor([c["local_pos1_m"] for c in spec["constraints"]],device=args.device,dtype=torch.float32)
        b0 = [robot.body_names.index(c["body0"]) for c in spec["constraints"]]
        b1 = [robot.body_names.index(c["body1"]) for c in spec["constraints"]]
        def gap():
            p,q = robot.data.body_pos_w.torch,robot.data.body_quat_w.torch
            a = p[:,b0]+quat_apply(q[:,b0],c0[None])
            b = p[:,b1]+quat_apply(q[:,b1],c1[None])
            return float((a-b).norm(dim=-1).max())
        traces = {}
        solved = spec["nominal_joint_pos"]
        for beta in (42.,50.,75.,100.,106.):
            prescribed = dict(L_joint1=.42,R_joint1=-.42,L_joint3=0.,R_joint3=0.,
                L_joint2=np.deg2rad(beta)-spec["passive_knee_zero_inner_rad"]["L_joint2"],
                R_jonit2=-(np.deg2rad(beta)-spec["passive_knee_zero_inner_rad"]["R_jonit2"]))
            solved = solve_pose(spec,prescribed,solved)
            q0 = torch.tensor([[solved[n] for n in robot.joint_names]],device=args.device,dtype=torch.float32)
            robot.reset(); response.reset()
            robot.write_joint_position_to_sim_index(position=q0)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
            robot.update(0.)
            leg_target = q0[:,ids][:,list(control.LEGS)].clone()
            effort = torch.zeros_like(q0)
            rows, max_gap, max_motor, max_passive = [], gap(), 0., 0.
            for step in range(args.steps):
                q,dq = robot.data.joint_pos.torch,robot.data.joint_vel.torch
                # Equal model signs test both drive directions; an airborne
                # fixture cannot qualify vehicle-forward ground wiring.
                speed = 8. if args.steps//3 <= step < 2*args.steps//3 else 0.
                reference = torch.tensor([[speed,-speed]],device=args.device)
                requested = control.motor_efforts(q[:,ids],dq[:,ids],leg_target,reference)
                motor = response.apply_torque(requested,dq[:,ids])
                effort.zero_();effort[:,ids] = motor
                effort[:,wheel_ids] += response.passive_effort
                effort[:,spring_ids] = control.spring_efforts(q[:,spring_ids])
                if not torch.isfinite(effort).all():
                    raise ValueError("Nonfinite full-vehicle effort")
                max_motor = max(max_motor,float(motor.abs().max()))
                max_passive = max(max_passive,float(response.passive_effort.abs().max()))
                robot.set_joint_effort_target_index(target=effort)
                robot.write_data_to_sim();sim.step(render=False);robot.update(.001)
                current_gap = gap()
                max_gap = max(max_gap,current_gap)
                qn,dqn = robot.data.joint_pos.torch,robot.data.joint_vel.torch
                if not torch.isfinite(qn).all() or not torch.isfinite(dqn).all() or current_gap > .003:
                    raise ValueError(f"Full-vehicle closure/finite-state guard at beta={beta},step={step},gap={current_gap}")
                if step%5 == 0:
                    state = observer(qn[:,ids][:,list(control.LEGS)],dqn[:,ids][:,list(control.LEGS)])
                    beta_actual = torch.rad2deg(qn[0,knee_ids]*qn.new_tensor([1.,-1.])
                        +qn.new_tensor(list(spec["passive_knee_zero_inner_rad"].values())))
                    error = (leg_target-qn[:,ids][:,list(control.LEGS)])[0]
                    rows.append([step*.001,current_gap,*error.tolist(),*beta_actual.tolist(),
                        *dqn[0,wheel_ids].tolist(),*motor[0].tolist(),float(state["valid"].all())])
            trace = np.asarray(rows)
            traces[f"beta_{int(beta)}"] = trace
            peak_error = float(abs(trace[:,2:6]).max())
            report["cases"].append(dict(beta_reference_deg=beta,physical_steps=args.steps,
                max_closure_gap_m=max_gap,max_leg_error_rad=peak_error,
                wheel_speed_min_rad_s=trace[:,8:10].min(axis=0).tolist(),
                wheel_speed_max_rad_s=trace[:,8:10].max(axis=0).tolist(),
                max_motor_torque_nm=max_motor,max_extra_passive_wheel_effort_nm=max_passive,
                encoder_valid_all_samples=bool(trace[:,-1].all()),
                passed=bool(max_gap<.003 and peak_error<.02 and trace[:,-1].all())))
            save()
            print(json.dumps(report["cases"][-1]),flush=True)
        np.savez_compressed(args.output/"traces.npz",**traces)
        report.update(status="completed",passed=all(c["passed"] for c in report["cases"]),
            physical_steps=args.steps*len(report["cases"]),elapsed_s=time.monotonic()-started,
            traces_sha256=digest(args.output/"traces.npz"),
            trace_columns=["t_s","closure_gap_m","error_LH_rad","error_LA_rad","error_RH_rad","error_RA_rad",
                "beta_left_deg","beta_right_deg","wheel_left_rad_s","wheel_right_rad_s",*["motor_"+n+"_nm" for n in ACTIVE],"encoder_valid"],
            scope="native component wiring and bounded airborne fixture; no balance, contact, policy or hardware acceptance")
        save()
        if not report["passed"]:
            raise RuntimeError("Full-vehicle component probe did not meet fixture criteria")
    except Exception as exc:
        report.update(status="failed",passed=False,error=str(exc),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-started)
        save()
        raise
    finally:
        if launcher is not None:
            launcher.app.close()


if __name__=="__main__":
    main()

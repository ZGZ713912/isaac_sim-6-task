#!/usr/bin/env python3
"""Independent zero-PPO native wiring/reset probe, including every v5 bank pose."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--num-envs",type=int,default=32)
    parser.add_argument("--steps",type=int,default=20)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--performance-backend",choices=("eager","cuda_graph"),default="eager")
    parser.add_argument("--wiring-suite",action="store_true",help="Also verify the standard training native gate without constructing a learner")
    parser.add_argument("--profiles",nargs="+",choices=("nominal","requested"),default=["nominal","requested"])
    args = parser.parse_args()
    if not 20 <= args.num_envs <= 80 or args.steps < 16:
        parser.error("Use20..80 environments and at least16 policy intervals to observe actor enable")
    args.output.mkdir(parents=True,exist_ok=False)
    from train_prone_motion import configuration
    from train_scratch_standing import write_json,digest
    plan = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v5.json").read_text())
    bank_data = json.loads((ROOT/plan["pose_bank"]["file"]).read_text())
    report = dict(schema="multi-pose-recovery-native-probe-v1",status="starting",device=args.device,
        performance_backend=args.performance_backend,
        new_PPO=0,optimizer_attempts=0,checkpoint_reads=[],behavior_qualified=False,hardware_motion=False,
        bank_sha256=digest(ROOT/plan["pose_bank"]["file"]),profiles=[],
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),
            ROOT/"scripts/train_prone_motion.py",ROOT/"scripts/evaluate_prone_motion.py",
            ROOT/"contracts/wheel_leg_prone_motion_v5.json",
            ROOT/"src/wheeled_tasks/chassis/prone_motion_env.py",ROOT/"src/wheeled_tasks/chassis/robust_getup_env.py",
            ROOT/"src/wheeled_tasks/chassis/multi_pose_recovery.py",ROOT/"src/wheeled_tasks/chassis/recovery_impact.py"]})
    report_path = args.output/"report.json"
    write_json(report_path,report)
    launcher = env = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True,device=args.device,enable_cameras=False))
        import torch
        from isaaclab.sim import SimulationContext
        from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.scut_observation import POLICY_FROM_CONTROL
        import train_scratch_standing as base
        base.torch = torch
        torch.set_num_threads(4)
        def forbidden_load(*a,**kw):
            report["checkpoint_reads"].append(str(a[0] if a else kw.get("f")))
            raise RuntimeError("Native probe cannot load policy, teacher or donor weights")
        torch.load = forbidden_load
        for profile in args.profiles:
            directory = args.output/profile
            cfg,settings,bank = configuration(directory,plan,profile=profile,strength=float(profile != "nominal"),
                num_envs=args.num_envs,device=args.device,performance_backend=args.performance_backend)
            manifest = json.loads((ROOT/cfg["asset_directory"]/"manifest.json").read_text())
            env = ProneMotionEnv(cfg,manifest,load_chassis_control(ROOT/cfg["control_math_source"]),ROOT,
                stage_name="speed",num_envs=args.num_envs,device=args.device,seed=plan["seed"],level=1.,
                run_role="train",bank=bank,settings=settings)
            write_json(directory/"startup.json",env.startup_report)
            wiring = None
            if args.wiring_suite:
                from evaluate_prone_motion import probe
                wiring = probe(env,plan)
                write_json(directory/"training_native_gate.json",wiring)
            checks = dict(all_bank_roots_and_joints_read_back=True,all_bank_velocities_read_back=True,
                finite_transitions=True,all_eight_motor_efforts_zero_while_disabled=True,
                suffix_and_impact_diagnostics_present=True,friction_passive_and_exactly_once=True,
                native_initial_closure_within3mm=True,standing_roots_grounded_within_declared_reset_correction=True)
            root_error_peak = dict(position_m=0.,quaternion_distance=0.)
            rows = []
            pose_ids = list(range(len(bank_data["poses"])))
            for offset in range(0,len(pose_ids),args.num_envs):
                indices = pose_ids[offset:offset+args.num_envs]
                cases = []
                for lane in range(args.num_envs):
                    i = indices[lane % len(indices)]
                    p = bank_data["poses"][i]
                    recovery = "recovery" in p or i == env.prone_index
                    h = p["root_position_m"][2]
                    cases.append(dict(role="prone" if recovery else "stand",recovery=recovery,pose_index=i,
                        birth_height=h,pitch_deg=0.,target=[0.,0.,.305],command=[0.,0.,.305]))
                env.evaluation_cases = cases
                env.reset(torch.arange(env.num_envs,device=env.device))
                actual_q = env.robot.data.joint_pos.torch
                actual_dq = env.robot.data.joint_vel.torch
                expected_q = env.bank_joints[env.pose_index]
                expected_dq = env.bank_velocities[env.pose_index]
                expected_root = env.bank_roots[env.pose_index].clone()
                expected_root[:,:3] += env.origins
                expected_velocity = expected_root.new_tensor([
                    [*bank_data["poses"][c["pose_index"]].get("root_linear_velocity_body_m_s",[0.,0.,0.]),
                     *bank_data["poses"][c["pose_index"]].get("root_angular_velocity_body_rad_s",[0.,0.,0.])]
                    for c in cases]).reshape(args.num_envs,2,3)
                xyz,w = expected_root[:,None,3:6].expand_as(expected_velocity),expected_root[:,None,6:7]
                cross = torch.linalg.cross(xyz,expected_velocity)
                expected_velocity += 2.*(w*cross+torch.linalg.cross(xyz,cross))
                actual_root = env.robot.data.root_link_pose_w.torch
                recovery = env.recovery_lane
                position_error = (actual_root[:,:3]-expected_root[:,:3]).abs().amax(-1)
                quaternion_error = torch.minimum((actual_root[:,3:]-expected_root[:,3:]).norm(dim=-1),
                    (actual_root[:,3:]+expected_root[:,3:]).norm(dim=-1))
                if bool(recovery.any()):
                    root_error_peak["position_m"] = max(root_error_peak["position_m"],float(position_error[recovery].max()))
                    root_error_peak["quaternion_distance"] = max(root_error_peak["quaternion_distance"],float(quaternion_error[recovery].max()))
                checks["all_bank_roots_and_joints_read_back"] &= bool(
                    torch.allclose(actual_q,expected_q,atol=2e-5,rtol=0.) and
                    (position_error[recovery] <= 2e-5).all() and (quaternion_error[recovery] <= 2e-5).all())
                # The three legacy standing births deliberately use the existing
                # reset-only wheel grounding correction, rather than exact bank Z.
                standing = (~recovery).nonzero().flatten()
                if len(standing):
                    from wheeled_tasks.chassis.grounded_standing_reset import grounded_pose
                    wheels = env.robot.data.body_link_pose_w.torch[standing][:,env.wheel_ids]
                    grounded,_,_ = grounded_pose(actual_root[standing],env.wheel_centers()[standing],
                        wheels[:,:,3:],env.origins[standing,2])
                    checks["standing_roots_grounded_within_declared_reset_correction"] &= bool(
                        torch.allclose(grounded,actual_root[standing],atol=2e-5,rtol=0.) and
                        (position_error[standing] <= .025).all() and (quaternion_error[standing] <= .02).all())
                checks["all_bank_velocities_read_back"] &= bool(
                    torch.allclose(actual_dq,expected_dq,atol=2e-5,rtol=0.) and
                    torch.allclose(env.robot.data.root_com_vel_w.torch,expected_velocity.reshape(args.num_envs,6),atol=2e-5,rtol=0.))
                checks["native_initial_closure_within3mm"] &= bool((env.closure_gap() <= .003).all())
                rows_batch = [dict(pose_index=i,name=bank_data["poses"][i]["name"],
                    enabled=False,terminated=False,closure_peak_m=0.,impact_peak_n=0.,suffix_seen=False) for i in indices]
                for _ in range(args.steps):
                    # Test action only; never constructs an actor, learner, optimizer or trajectory script.
                    actions = env.actions[:,list(POLICY_FROM_CONTROL)].clone()
                    observation,reward,done,info = env.step(actions)
                    state,physical = info["diagnostics"],env.prone_motion_physical
                    checks["finite_transitions"] &= bool(torch.isfinite(reward).all() and
                        all(torch.isfinite(t).all() for t in observation.values()))
                    disabled = physical["wheel_disabled"]
                    checks["all_eight_motor_efforts_zero_while_disabled"] &= bool(
                        (physical["motor_effort"][disabled] == 0.).all() and
                        (physical["gimbal_effort"][disabled] == 0.).all())
                    loss,speed = physical["leg_friction_passive_effort"],physical["leg_friction_velocity"]
                    expected = -speed.clamp(-.05,.05)/.05*speed.new_tensor(env.leg_friction.model["coulomb_nm"])
                    checks["friction_passive_and_exactly_once"] &= bool(torch.allclose(loss,expected,atol=2e-6,rtol=1e-6)
                        and (loss*speed <= 1e-7).all()
                        and torch.allclose(physical["leg_net_effort"],physical["leg_applied_effort"],atol=3e-6,rtol=1e-6))
                    checks["suffix_and_impact_diagnostics_present"] &= all(k in state for k in
                        ("recovery_enabled_at","recovery_entry_leg_class","recovery_force_peak_n","recovery_impulse_peak_ns"))
                    for lane,row in enumerate(rows_batch):
                        row["enabled"] |= bool(torch.isfinite(state["recovery_enabled_at"][lane]))
                        row["terminated"] |= bool(done[lane])
                        row["suffix_seen"] |= bool(state["recovery_suffix_entry_event"][lane])
                        row["closure_peak_m"] = max(row["closure_peak_m"],float(state["gap"][lane]))
                        row["impact_peak_n"] = max(row["impact_peak_n"],float(state["recovery_force_peak_n"][lane].max()))
                rows.extend(rows_batch)
                print(json.dumps(dict(profile=profile,checked_poses=len(rows),total=len(pose_ids))),flush=True)
            result = dict(profile=profile,checks=checks,all_checks_pass=all(checks.values()),poses=rows,
                initial_state_count=len(rows),policy_steps_per_pose=args.steps,physics_dt_s=env.dt,policy_dt_s=env.policy_dt,
                recovery_root_readback_error_peak=root_error_peak)
            if wiring is not None:
                result["training_native_gate"] = wiring
                result["all_checks_pass"] &= wiring["all_checks_pass"]
            write_json(directory/"native.json",result)
            report["profiles"].append(result)
            write_json(report_path,report)
            env.close(); env = None
            SimulationContext.clear_instance(); gc.collect()
        report.update(status="completed",passed=all(r["all_checks_pass"] for r in report["profiles"]))
    except Exception:
        report.update(status="failed",passed=False,error=traceback.format_exc())
        raise
    finally:
        write_json(report_path,report)
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()

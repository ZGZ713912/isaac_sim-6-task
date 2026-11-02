#!/usr/bin/env python3
"""Train the original Wheelbipe V14.2 PPO with the v2.0.2 vehicle adapter."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")


def save(path,value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("preflight","train","evaluate"),default="train")
    parser.add_argument("--task",choices=("flat","rough"),default="flat")
    parser.add_argument("--num-envs",type=int,default=4096)
    parser.add_argument("--iterations",type=int,default=20000)
    parser.add_argument("--save-interval",type=int,default=500)
    parser.add_argument("--seed",type=int,default=107)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--headless",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=500)
    parser.add_argument("--randomization-profile",default="requested")
    parser.add_argument("--randomization-strength",type=float,default=1.)
    parser.add_argument("--solver-position-iterations",type=int,default=12)
    parser.add_argument("--solver-velocity-iterations",type=int,default=6)
    parser.add_argument("--external-forces-every-iteration",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--physics-hz",type=int,choices=(200,500,1000,2000),default=200,
        help="Explicit integration/feedback frequency; the original policy stays at 50Hz")
    parser.add_argument("--leg-armature",type=float,default=.015795,
        help="Additional inertia on each of the four driven leg axes, kg*m^2")
    parser.add_argument("--leg-kp",type=float,default=160.)
    parser.add_argument("--leg-kd",type=float,default=2.5)
    parser.add_argument("--leg-target-projection-penalty",type=float,default=1.,
        help="Cost weight for unreachable leg targets in rad^2; 0 reproduces the historical objective")
    parser.add_argument("--max-constraint-gap-mm",type=float,default=None,
        help="Optional historical policy-sampled gap limit; omitted means diagnostics only")
    parser.add_argument("--checkpoint",type=Path,help="Read-only evaluation only; never initializes training")
    parser.add_argument("--managed-control-dir",type=Path,
        help="Internal binding created by trainctl wheelbipe init/start")
    parser.add_argument("--resume",type=Path,
        help="Full learning state from a cleanly stopped managed attempt; requires --managed-control-dir")
    args = parser.parse_args()
    from wheeled_tasks.wheelbipe_port.training_contract import (
        constraint_limit_m,check_constraint_gap,leg_armature_kg_m2,CONSTRAINT_DIAGNOSTIC_SCOPE,
        projection_penalty,
        leg_pd_gains,
    )
    try:
        gap_limit = constraint_limit_m(args.max_constraint_gap_mm)
        args.leg_armature = leg_armature_kg_m2(args.leg_armature)
        args.leg_target_projection_penalty = projection_penalty(args.leg_target_projection_penalty)
        args.leg_kp,args.leg_kd = leg_pd_gains(args.leg_kp,args.leg_kd)
    except ValueError as error:
        parser.error(str(error))
    if args.num_envs<1 or args.iterations<1 or args.steps<1 or not 0.<=args.randomization_strength<=1.:
        parser.error("Invalid positive budget or randomization strength")
    if not 1<=args.solver_position_iterations<=255 or not 0<=args.solver_velocity_iterations<=255:
        parser.error("PhysX solver iterations must be within 1..255 positions / 0..255 velocities")
    if (args.mode=="evaluate") != (args.checkpoint is not None):
        parser.error("--checkpoint is required only for read-only evaluate mode")
    if args.resume is not None and args.managed_control_dir is None:
        parser.error("--resume requires a pinned managed session; legacy model-only checkpoints cannot cold-resume")
    if args.managed_control_dir is not None and args.mode!="train":
        parser.error("Only PPO training can use a managed control binding")
    args.output = args.output.resolve()
    managed = None
    endpoint_phase = None
    if args.managed_control_dir is not None:
        from wheeled_algo.managed_training import ManagedWorker, ManagedStop
        managed = ManagedWorker(args.managed_control_dir,args.output,args.resume)
    args.output.mkdir(parents=True,exist_ok=False)
    receipt = dict(status="preparing",started_at=datetime.now(timezone.utc).isoformat(),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        hardware_io=False,behavior_qualified=False,completed_updates=0,constraint_gap_limit_m=gap_limit,
        constraint_gap_hard_limit_enabled=gap_limit is not None,
        diagnostic_sampling_scope=CONSTRAINT_DIAGNOSTIC_SCOPE)
    path = args.output/"receipt.json"
    save(path,receipt)
    app = None
    wrapped = None
    runner = None
    started = time.monotonic()
    try:
        from wheeled_tasks.wheelbipe_port.model import expose_source, BUNDLE, TABLE, ACTIVE, BODY_ALIASES, JOINT_ALIASES
        expose_source()
        versions = {n:importlib.metadata.version(n) for n in ("isaacsim","isaaclab","torch","rsl-rl-lib")}
        if versions["rsl-rl-lib"]!="3.0.1" or not versions["isaacsim"].startswith("5.1."):
            raise ValueError(f"This source-faithful experiment requires Sim5.1/RSL-RL3.0.1: {versions}")
        receipt.update(versions=versions,source_snapshot_sha256=sha(ROOT/"third_party/wheelbipe/provenance.json"),
            asset_manifest_sha256=sha(BUNDLE/"manifest.json"),reset_lookup_sha256=sha(TABLE),
            collision_support_sha256=sha(TABLE.parent/"collision_support.json"))
        receipt["ground_plane_sha256"] = sha(TABLE.parent/"ground_plane.usda")
        implementation = [*sorted((ROOT/"src").rglob("*.py")),
            ROOT/"scripts/train_wheelbipe_port.py",ROOT/"scripts/export_wheelbipe_port_onnx.py"]
        receipt["implementation_files_sha256"] = {str(p.relative_to(ROOT)):sha(p) for p in implementation}
        deployment = ROOT/"deployment.json"
        if deployment.is_file():
            receipt["deployment"] = json.loads(deployment.read_text())
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=args.headless,device=args.device,enable_cameras=False))
        app = launcher.app
        import torch
        torch.set_num_threads(4)
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from agent_tasks.direct.wheelbipe.agents.rsl_rl_ppo_cfg import (
            WheelbipeV14FlatPPORunnerCfg,WheelbipeV14RoughPPORunnerCfg,
        )
        from wheeled_tasks.wheelbipe_port.config import WheelbipePortFlatCfg,WheelbipePortRoughCfg
        from wheeled_tasks.wheelbipe_port.environment import WheelbipePortEnv
        cfg = (WheelbipePortFlatCfg if args.task=="flat" else WheelbipePortRoughCfg)()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.sim.dt,cfg.decimation = 1./args.physics_hz,args.physics_hz//50
        cfg.sim.render_interval = cfg.decimation
        cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = args.solver_position_iterations
        cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = args.solver_velocity_iterations
        cfg.robot_cfg.actuators["legs_act"].armature = args.leg_armature
        cfg.port_leg_kp,cfg.port_leg_kd = args.leg_kp,args.leg_kd
        cfg.sim.physx.enable_external_forces_every_iteration = args.external_forces_every_iteration
        cfg.seed = args.seed
        cfg.port_randomization_profile = args.randomization_profile
        cfg.port_randomization_strength = args.randomization_strength
        cfg.port_record_peaks = args.mode in ("preflight","train")
        cfg.port_leg_target_projection_penalty = args.leg_target_projection_penalty
        cfg.rewards["port_leg_target_projection"] = -args.leg_target_projection_penalty
        if args.mode=="evaluate":
            cfg.use_leg_random_start = False
            cfg.events.reset_base.params["pose_range"] = {}
            cfg.events.reset_base.params["velocity_range"] = {}
            cfg.commands.special_modes = {}
        env = WheelbipePortEnv(cfg)
        wrapped = RslRlVecEnvWrapper(env)
        observations = wrapped.get_observations()
        dimensions = {name:list(value.shape) for name,value in observations.items()}
        if dimensions["policy"][-1]!=35 or dimensions["critic"][-1]!=78:
            raise ValueError(f"Reference observation dimensions differ: {dimensions}")
        leg_ids = [env.motor_ids[i] for i in (0,1,3,4)]
        leg_armatures = torch.as_tensor(env.robot.root_physx_view.get_dof_armatures())[:,leg_ids]
        if not torch.allclose(leg_armatures,torch.full_like(leg_armatures,args.leg_armature),atol=1.e-8,rtol=0.):
            raise ValueError("Native leg armature differs from the requested per-axis inertia")
        native = dict(actor_dim=35,critic_dim=78,action_dim=6,observation_shapes=dimensions,
            cad_joint_names=env.robot.cad_joint_names,policy_joint_names=env.robot.joint_names,
            cad_body_names=env.robot.cad_body_names,policy_body_names=env.robot.body_names,
            active_joint_order=[env.robot.cad_joint_names[i] for i in env._actuate_idx],
            policy_action_order=[env.robot.cad_joint_names[i] for i in env._actuate_idx],
            default_joint_pos=env.robot.data.default_joint_pos[0].cpu().tolist(),
            leg_action_scale=env.leg_action_scale,wheel_action_scale=env.wheel_action_scale,
            policy_dt_s=env.step_dt,physics_dt_s=env.physics_dt,domain=env.domain_receipt,
            leg_armature_kg_m2=args.leg_armature,leg_armature_joint_names=[ACTIVE[i] for i in (0,1,3,4)],
            leg_kp_Nm_per_rad=env.control.settings["leg_kp"],
            leg_kd_Nm_per_rad_s=env.control.settings["leg_kd"],
            leg_armature_native_readback_kg_m2=leg_armatures[0].cpu().tolist(),
            leg_armature_readback_verified_envs=args.num_envs,
            diagnostic_sampling_scope=CONSTRAINT_DIAGNOSTIC_SCOPE,
            paired_target_signed_delta_bounds_rad=env.paired_targets.bounds.cpu().tolist(),
            leg_target_projection_penalty=args.leg_target_projection_penalty,
            leg_target_projection_cost="sum of squared discarded pair targets, rad^2; shared winding unpenalized",
            joint_velocity_guard_scope="asset solver guard; not an identified motor-speed limit",
            spring_model_sha256=env.control.gas_compensation.cfg["model_sha256"],
            wheel_model_sha256=env.wheel_response.cfg["model_sha256"],
            friction_model_sha256=env.leg_friction.cfg["model_sha256"],
            gimbal="CAD mass retained; both joints fixed at zero",
            body_masses_kg=env.robot.root_physx_view.get_masses().cpu().tolist())
        from pxr import UsdPhysics
        loops = [p for p in env.sim.stage.Traverse() if p.IsA(UsdPhysics.SphericalJoint)]
        native["external_loop_count"] = len(loops)
        if len(loops)!=6*args.num_envs:
            raise ValueError(f"Expected six external loop joints per environment; got {len(loops)}")
        native["loop_excluded_from_articulation"] = all(UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get() for p in loops)
        if not native["loop_excluded_from_articulation"]:
            raise ValueError("A loop was inserted into the reduced-coordinate tree")
        if env.robot.num_joints!=18 or env.robot.num_bodies!=21:
            raise ValueError("Full v2.0.2 topology differs")
        receipt.update(status="native_started",native=native)
        save(path,receipt)
        agent = (WheelbipeV14FlatPPORunnerCfg if args.task=="flat" else WheelbipeV14RoughPPORunnerCfg)()
        agent.seed,agent.device = args.seed,args.device
        agent.save_interval,agent.max_iterations = args.save_interval,args.iterations
        dump_yaml(str(args.output/"params/env.yaml"),cfg)
        dump_yaml(str(args.output/"params/agent.yaml"),agent)
        if args.mode=="preflight":
            peak_gap = 0.
            minimum_spring = float("inf")
            maximum_spring = 0.
            terminated = 0
            calls_before = env.port_physics_calls
            for step in range(args.steps):
                actions = torch.empty(args.num_envs,6,device=env.device).uniform_(-.2,.2)
                observations,reward,done,extras = wrapped.step(actions)
                state = torch.cat((env.robot.data.joint_pos,env.robot.data.joint_vel,
                                   env.robot.data.root_state_w,env.motor_effort,env.spring_force),-1)
                if not torch.isfinite(state).all() or not torch.isfinite(reward).all():
                    raise ValueError(f"Nonfinite native state/reward at policy step {step}")
                if not all(torch.isfinite(v).all() for v in observations.values()):
                    raise ValueError("Nonfinite source observations")
                if env._obs_raw_policy_has_nonfinite.any() or env._obs_raw_critic_has_nonfinite.any():
                    raise ValueError("Source observation sanitization hid a nonfinite raw input")
                peak_gap = max(peak_gap,float(env.port_constraint_gap_peak))
                minimum_spring = min(minimum_spring,float(env.spring_force.min()))
                maximum_spring = max(maximum_spring,float(env.spring_force.max()))
                terminated += int(done.sum())
                if env.motor_effort[:,(0,1,3,4)].abs().max()>40.00001 or env.motor_effort[:,(2,5)].abs().max()>4.50001:
                    raise ValueError("Electrical torque exceeded its delivered cap")
            receipt.update(status="preflight_complete",native_steps=args.steps,
                physical_steps=args.steps*cfg.decimation,max_constraint_gap_m=peak_gap,
                spring_force_range_N=[minimum_spring,maximum_spring],resets=terminated,
                all_state_reward_observation_finite=True)
            receipt["peak_diagnostic"] = env.port_peak_diagnostic
            receipt["measured_controller_and_spring_calls"] = env.port_physics_calls-calls_before
            if receipt["measured_controller_and_spring_calls"]!=args.steps*cfg.decimation:
                raise ValueError("Feedback PD and spring must run once per physics tick")
            check_constraint_gap(peak_gap,gap_limit,"Preflight")
        elif args.mode=="train":
            runner = OnPolicyRunner(wrapped,agent.to_dict(),log_dir=str(args.output),device=args.device)
            if runner.alg.optimizer.state and args.resume is None:
                raise ValueError("A fresh experiment must start with empty Adam")
            runner._prepare_logging_writer()
            if managed is not None:
                managed.attach(runner,env,args.resume)
            start_completed = managed.completed if managed is not None else 0
            receipt["started_from_completed_updates"] = start_completed
            receipt["cold_resume"] = args.resume is not None
            receipt["physical_episode_restored"] = False
            initial = {k:v.detach().cpu().clone() for k,v in runner.alg.policy.state_dict().items()}
            runner.save(str(args.output/"model_initial.pt"))
            receipt["initial_checkpoint_sha256"] = sha(args.output/"model_initial.pt")
            original_log = runner.log
            def log_progress(values,*log_args,**log_kwargs):
                original_log(values,*log_args,**log_kwargs)
                # Keep timing in the same event stream as the original RSL
                # collection/learning metrics; setup/checkpoint IO is separate.
                iteration_time = values["collection_time"]+values["learn_time"]
                runner.writer.add_scalar("Perf/iteration_time_s",iteration_time,values["it"])
                runner.writer.add_scalar("Perf/physics_steps_per_update",24*cfg.decimation,values["it"])
                runner.writer.add_scalar("Port/constraint_gap_hard_limit_enabled",int(gap_limit is not None),values["it"])
                if gap_limit is not None:
                    runner.writer.add_scalar("Port/constraint_gap_limit_m",gap_limit,values["it"])
                completed = int(values["it"])+1
                peak = float(env.port_constraint_gap_peak)
                receipt.update(status="training",completed_updates=completed,
                    elapsed_s=time.monotonic()-started,max_constraint_gap_m=peak,
                    physical_steps=env.port_physics_calls,
                    joint_speed_running_peak_rad_s=float(env.port_joint_speed_peak),
                    body_angular_speed_running_peak_rad_s=float(env.port_body_angular_speed_peak),
                    peak_diagnostic=getattr(env,"port_peak_diagnostic",None),
                    last_collection_time_s=values["collection_time"],last_learning_time_s=values["learn_time"],
                    last_iteration_time_s=iteration_time)
                if completed==start_completed+1:
                    updated = runner.alg.policy.state_dict()
                    receipt["first_update_changed_policy_tensors"] = [k for k,v in initial.items()
                        if not torch.equal(v,updated[k].detach().cpu())]
                    receipt["first_update_optimizer_state_count"] = len(runner.alg.optimizer.state)
                if completed==1 or completed%10==0:
                    save(path,receipt)
                if not all(torch.isfinite(torch.as_tensor(v)).all() for v in values["loss_dict"].values()):
                    raise ValueError("Nonfinite PPO losses")
                check_constraint_gap(peak,gap_limit,"PPO")
                if managed is not None:
                    managed.after_update(completed)
                    active,paused = managed.control.elapsed()
                    runner.writer.add_scalar("TrainControl/active_seconds",active,completed)
                    runner.writer.add_scalar("TrainControl/paused_seconds",paused,completed)
            runner.log = log_progress
            original_save = runner.save
            def save_progress(filename,infos=None):
                original_save(filename,infos)
                receipt.update(status="training",completed_updates=runner.current_learning_iteration+1,
                    latest_checkpoint=str(filename),elapsed_s=time.monotonic()-started)
                save(path,receipt)
            runner.save = save_progress
            if managed is None:
                env.set_training_progress(iteration=0)
            runner.learn(num_learning_iterations=args.iterations-start_completed,
                init_at_random_ep_len=args.resume is None)
            final = runner.alg.policy.state_dict()
            changed = [k for k,v in initial.items() if not torch.equal(v,final[k].detach().cpu())]
            finite = all(torch.isfinite(v).all() for v in final.values())
            if not finite or not any(k.startswith("actor.") for k in changed) or not any(k.startswith("critic.") for k in changed):
                raise ValueError("Fresh PPO did not produce finite actor and critic updates")
            last = args.output/f"model_{runner.current_learning_iteration}.pt"
            receipt.update(status="training_complete",completed_updates=args.iterations,
                latest_checkpoint=str(last),checkpoint_sha256=sha(last),
                fresh_initial_optimizer_state_count=0 if args.resume is None else None,
                final_optimizer_state_count=len(runner.alg.optimizer.state),
                changed_policy_tensors=changed,policy_tensors_finite=bool(finite),
                max_constraint_gap_m=float(env.port_constraint_gap_peak),
                physical_steps=env.port_physics_calls)
            if managed is not None:
                endpoint_phase = "completed"
        else:
            runner = OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=args.device)
            runner.load(str(args.checkpoint),load_optimizer=False)
            actor = runner.get_inference_policy(device=args.device)
            # Cold command cases use the same source observation/reward path.
            cases = [("stand_low",0.,0.,.23),("stand_mid",0.,0.,.305),("stand_high",0.,0.,.42),
                     ("forward",.5,0.,.305),("reverse",-.5,0.,.305),
                     ("spin_positive",0.,100.*2.*torch.pi/60.,.305),
                     ("spin_negative",0.,-100.*2.*torch.pi/60.,.305)]
            rows = []
            for name,vx,wz,height in cases:
                env.eval_command = (vx,wz,height)
                with torch.inference_mode():
                    wrapped.reset()
                    observations = wrapped.get_observations()
                velocities,angular,heights,gaps = [],[],[],[]
                reset_count = 0
                for step in range(args.steps):
                    with torch.inference_mode():
                        observations,reward,done,extras = wrapped.step(actor(observations))
                    reset_count += int(done.sum())
                    velocities.append(env.robot.data.root_lin_vel_b[:,0].clone())
                    angular.append(env.robot.data.root_ang_vel_b[:,2].clone())
                    heights.append(env.robot.data.root_pos_w[:,2].clone())
                    gaps.append(env.constraint_gaps().max())
                rows.append(dict(case=name,command=[vx,wz,height],duration_s=args.steps*env.step_dt,
                    envs=args.num_envs,resets=reset_count,actual_vx_mean_m_s=float(torch.stack(velocities).mean()),
                    actual_yaw_mean_rad_s=float(torch.stack(angular).mean()),
                    actual_height_mean_m=float(torch.stack(heights).mean()),max_gap_m=float(torch.stack(gaps).max())))
            receipt.update(status="evaluation_complete",checkpoint_sha256=sha(args.checkpoint),cases=rows,
                acceptance="diagnostic cold command cases; complete acceptance suite remains separate")
        receipt["elapsed_s"] = time.monotonic()-started
        receipt["max_constraint_gap_m"] = float(env.port_constraint_gap_peak)
        check_constraint_gap(receipt["max_constraint_gap_m"],gap_limit,args.mode)
        receipt["reset_initial_penetration_peak_m"] = float(env.reset_penetration_peak)
        receipt["peak_diagnostic"] = getattr(env,"port_peak_diagnostic",None)
        receipt["joint_speed_running_peak_rad_s"] = float(env.port_joint_speed_peak)
        receipt["body_angular_speed_running_peak_rad_s"] = float(env.port_body_angular_speed_peak)
        save(path,receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k not in ("native","arguments")},indent=2))
    except BaseException as exc:
        if managed is not None and isinstance(exc,ManagedStop):
            endpoint_phase = "stopped"
            receipt.update(status="training_stopped",completed_updates=managed.completed,
                elapsed_s=time.monotonic()-started,cold_resume_available_after_parent_exit=True,
                physical_episode_restored=False)
            save(path,receipt)
        else:
            if managed is not None:
                managed.control.fail(exc)
            receipt.update(status="failed",error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-started)
            if wrapped is not None:
                receipt["peak_diagnostic"] = getattr(env,"port_peak_diagnostic",None)
                receipt["reset_initial_penetration_peak_m"] = float(env.reset_penetration_peak)
                receipt["joint_speed_running_peak_rad_s"] = float(env.port_joint_speed_peak)
                receipt["body_angular_speed_running_peak_rad_s"] = float(env.port_body_angular_speed_peak)
            save(path,receipt)
            raise
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        if managed is not None:
            if endpoint_phase is not None:
                checkpoint = managed.commit_endpoint(endpoint_phase)
                receipt["durable_checkpoint"] = str(checkpoint)
                save(path,receipt)
            managed.close()
        if wrapped is not None:
            wrapped.close()
        if app is not None:
            # The documented immediate headless shutdown avoids importing the
            # graphics-only Replicator path on WSL after all checkpoints,
            # receipts and event writers have already been flushed.
            app.close(skip_cleanup=args.headless)


if __name__=="__main__":
    main()

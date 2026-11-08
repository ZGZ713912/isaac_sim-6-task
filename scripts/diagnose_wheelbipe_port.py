#!/usr/bin/env python3
"""Zero-PPO clone/anchor audit and controlled dynamics comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--num-envs",type=int,default=32)
    parser.add_argument("--steps",type=int,default=500)
    parser.add_argument("--seed",type=int,default=111)
    parser.add_argument("--leg-kp",type=float,default=160.)
    parser.add_argument("--leg-kd",type=float,default=2.5)
    parser.add_argument("--spring-damping",type=float,default=.002)
    parser.add_argument("--passive-armature",type=float,default=0.)
    parser.add_argument("--randomization-profile",choices=("nominal","requested"),default="nominal")
    parser.add_argument("--external-forces-every-iteration",action="store_true")
    parser.add_argument("--solver",nargs=2,type=int,default=(64,32))
    parser.add_argument("--actions",choices=("zero","random"),default="random")
    parser.add_argument("--checkpoint",type=Path,help="Read-only deterministic actor diagnostic, no PPO")
    parser.add_argument("--command",type=float,nargs=3,metavar=("VX","YAW","HEIGHT"))
    parser.add_argument("--hold-leg-length",type=float,help="Fixed CAD IK target, replaces policy actions")
    parser.add_argument("--audit-reset-cache",action="store_true",help="Isolated 200m teleport/reset consistency audit, no physics step")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing diagnostic")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    root = args.source_root.resolve()
    sys.path.insert(0,str(root/"src"))
    from wheeled_tasks.wheelbipe_port.model import expose_source
    expose_source()
    from isaaclab.app import AppLauncher
    app = AppLauncher(dict(headless=True,device="cuda:0",enable_cameras=False)).app
    started = time.monotonic()
    env = None
    result = dict(status="preparing",arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        ppo_updates=0,hardware_io=False,gap_scope="policy observation samples, without a distance threshold")
    def save():
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    try:
        import torch
        from pxr import UsdPhysics
        from wheeled_tasks.wheelbipe_port.config import WheelbipePortFlatCfg
        from wheeled_tasks.wheelbipe_port.environment import WheelbipePortEnv
        torch.set_num_threads(4)
        cfg = WheelbipePortFlatCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed = args.seed
        cfg.sim.dt,cfg.decimation = .005,4
        cfg.sim.render_interval = 4
        cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = args.solver[0]
        cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = args.solver[1]
        cfg.sim.physx.enable_external_forces_every_iteration = args.external_forces_every_iteration
        cfg.robot_cfg.actuators["spring"].damping = args.spring_damping
        cfg.robot_cfg.actuators["legs_inact"].armature = args.passive_armature
        cfg.port_randomization_profile = args.randomization_profile
        cfg.port_record_peaks = True
        if args.command is not None:
            cfg.use_leg_random_start = False
            cfg.events.reset_base.params["pose_range"] = {}
            cfg.events.reset_base.params["velocity_range"] = {}
            cfg.commands.special_modes = {}
        env = WheelbipePortEnv(cfg)
        env.control.settings["leg_kp"],env.control.settings["leg_kd"] = args.leg_kp,args.leg_kd
        actor = None
        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint,map_location=env.device,weights_only=False)
            state = checkpoint["model_state_dict"]
            if any("obs_normalizer" in k for k in state):
                raise ValueError("This diagnostic expects the source actor without normalization")
            layers = []
            for i,(inputs,outputs) in enumerate(((35,256),(256,128),(128,64),(64,6))):
                layer = torch.nn.Linear(inputs,outputs,device=env.device)
                layer.weight.data.copy_(state[f"actor.{i*2}.weight"])
                layer.bias.data.copy_(state[f"actor.{i*2}.bias"])
                layers.append(layer)
                if i<3:
                    layers.append(torch.nn.ELU())
            actor = torch.nn.Sequential(*layers).eval()
            result["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if args.command is not None:
            env.eval_command = tuple(args.command)
        obs,_ = env.reset()
        loops = [p for p in env.sim.stage.Traverse() if p.IsA(UsdPhysics.SphericalJoint)]
        bad,examples = [],[]
        for prim in loops:
            joint = UsdPhysics.Joint(prim)
            owner = str(prim.GetPath()).split("/Robot/")[0]
            targets = [str(v) for rel in (joint.GetBody0Rel(),joint.GetBody1Rel()) for v in rel.GetTargets()]
            row = dict(joint=str(prim.GetPath()),bodies=targets,
                excluded=joint.GetExcludeFromArticulationAttr().Get())
            if len(targets)!=2 or any(not v.startswith(owner+"/Robot/") for v in targets):
                bad.append(row)
            if len(examples)<12:
                examples.append(row)
        result.update(status="sampling",external_loop_count=len(loops),bad_clone_ownership=bad,
            joint_examples=examples,body_names=env.robot.cad_body_names,
            body_masses_kg=env.robot.root_physx_view.get_masses()[0].cpu().tolist())
        if args.audit_reset_cache:
            from isaaclab.utils.math import quat_apply
            def anchor_readback():
                transforms = env.robot.root_physx_view.get_link_transforms().clone()
                positions = transforms[:,:,:3]
                quaternions = transforms[:,:,(6,3,4,5)]
                a = positions[:,env._constraint_body0]+quat_apply(quaternions[:,env._constraint_body0],
                    env._constraint_points0[None].expand(args.num_envs,-1,-1))
                b = positions[:,env._constraint_body1]+quat_apply(quaternions[:,env._constraint_body1],
                    env._constraint_points1[None].expand(args.num_envs,-1,-1))
                return dict(cached_gap_m=env.constraint_gaps()[0].cpu().tolist(),
                    physx_gap_m=(a-b).norm(dim=-1)[0].cpu().tolist())
            ids = torch.tensor([0],device=env.device)
            result["reset_cache_audit"] = dict(initial=anchor_readback())
            root_state = env.robot.data.root_state_w[:1].clone()
            root_state[:,0] += 200.
            env.robot.write_root_state_to_sim(root_state,ids)
            env.sim.forward()
            result["reset_cache_audit"]["teleported"] = anchor_readback()
            env._reset_idx(ids)
            result["reset_cache_audit"]["immediate_reset"] = anchor_readback()
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(0.)
            result["reset_cache_audit"]["after_forward"] = anchor_readback()
            result["reset_cache_audit"]["physics_steps_during_audit"] = 0
        save()
        max_gap = 0.
        max_root_speed = 0.
        max_joint_speed = 0.
        resets = 0
        for step in range(args.steps):
            actions = torch.zeros(args.num_envs,6,device=env.device)
            if actor is not None:
                with torch.inference_mode():
                    actions = actor(obs["policy"])
            elif args.hold_leg_length is not None:
                length = torch.full((args.num_envs,2),args.hold_leg_length,device=env.device)
                target = env.reset_lookup.sample(length,torch.zeros_like(length))[:,env._legs_act_idx]
                actions[:,:4] = (target-env.robot.data.default_joint_pos[:,env._legs_act_idx])/env.leg_action_scale
            elif args.actions=="random":
                actions.normal_().clamp_(-100.,100.)
            with torch.inference_mode():
                obs,reward,terminated,truncated,_ = env.step(actions)
                done = terminated|truncated
            state = torch.cat((env.robot.data.root_state_w,env.robot.data.joint_pos,env.robot.data.joint_vel),-1)
            if not torch.isfinite(state).all() or not torch.isfinite(reward).all():
                raise ValueError(f"Nonfinite native state at policy sample {step+1}")
            gaps = env.constraint_gaps()
            gap = float(gaps.max())
            if gap>max_gap:
                lane,pin = divmod(int(gaps.argmax()),6)
                max_gap = gap
                result["max_gap_sample"] = dict(policy_sample=step+1,lane=lane,pin=pin,
                    gap_m=gap,reset_on_sample=bool(done[lane]),
                    root_state=env.robot.data.root_state_w[lane].cpu().tolist(),
                    joint_vel=env.robot.data.joint_vel[lane].cpu().tolist())
            max_root_speed = max(max_root_speed,float(env.robot.data.root_lin_vel_w.norm(dim=-1).max()))
            max_joint_speed = max(max_joint_speed,float(env.robot.data.joint_vel.abs().max()))
            resets += int(done.sum())
            if (step+1)%50==0:
                result.update(policy_samples=step+1,max_gap_m=max_gap,
                    max_joint_speed_rad_s=max_joint_speed,resets=resets,
                    elapsed_s=time.monotonic()-started)
                save()
            if step in (0,args.steps//2,args.steps-1):
                result.setdefault("state_samples",[]).append(dict(policy_sample=step+1,
                    command=env.command[0].cpu().tolist(),height_cmd_m=float(env.height_cmd[0]),
                    effective_height_cmd_m=float(env._get_effective_height_cmd()[0]),
                    reward_height_target_m=float(env._get_height_reward_target_height()[0]),
                    actor_observation=obs["policy"][0].cpu().tolist(),
                    action=actions[0].cpu().tolist(),
                    decoded_leg_target=env.leg_actions[0].cpu().tolist(),
                    paired_target=env.paired_targets(env.leg_actions[:,env.leg_target_order],
                        env.robot.data.joint_pos[:,[env.motor_ids[i] for i in (0,1,3,4)]])[0].cpu().tolist(),
                    motor_joint_order=list(env.asset.manifest["control_joint_names"]),
                    motor_positions=env.robot.data.joint_pos[0,env.motor_ids].cpu().tolist(),
                    motor_effort_Nm=env.motor_effort[0].cpu().tolist(),
                    spring_feedforward_Nm=env.control.gas_compensation_effort[0].cpu().tolist(),
                    base_height_m=float(env.robot.data.root_pos_w[0,2]),
                    root_gravity=env.robot.data.projected_gravity_b[0].cpu().tolist()))
        result.update(status="complete",policy_samples=args.steps,max_gap_m=max_gap,
            max_root_speed_m_s=max_root_speed,max_joint_speed_rad_s=max_joint_speed,resets=resets,
            elapsed_s=time.monotonic()-started,final_base_height_mean_m=float(env.robot.data.root_pos_w[:,2].mean()))
    except BaseException as error:
        result.update(status="failed",error=repr(error),elapsed_s=time.monotonic()-started)
        raise
    finally:
        save()
        if env is not None:
            env.close()
        app.close(skip_cleanup=True)


if __name__=="__main__":
    main()

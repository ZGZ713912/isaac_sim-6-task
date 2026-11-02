#!/usr/bin/env python3
"""Zero-PPO clone/anchor audit and controlled dynamics comparisons."""
from __future__ import annotations

import argparse
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
    parser.add_argument("--actions",choices=("zero","random"),default="random")
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
        cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = 64
        cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 32
        cfg.robot_cfg.actuators["spring"].damping = args.spring_damping
        cfg.robot_cfg.actuators["legs_inact"].armature = args.passive_armature
        cfg.port_randomization_profile = "nominal"
        cfg.port_record_peaks = True
        env = WheelbipePortEnv(cfg)
        env.control.settings["leg_kp"],env.control.settings["leg_kd"] = args.leg_kp,args.leg_kd
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
        save()
        max_gap = 0.
        max_root_speed = 0.
        max_joint_speed = 0.
        resets = 0
        for step in range(args.steps):
            actions = torch.zeros(args.num_envs,6,device=env.device)
            if args.actions=="random":
                actions.normal_().clamp_(-100.,100.)
            with torch.inference_mode():
                _,reward,terminated,truncated,_ = env.step(actions)
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

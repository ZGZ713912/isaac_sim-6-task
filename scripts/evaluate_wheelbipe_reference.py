#!/usr/bin/env python3
"""Read-only source-asset baseline; never imports its policy into our learner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--asset",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--num-envs",type=int,default=32)
    parser.add_argument("--steps",type=int,default=200)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite reference evidence")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.source_root.resolve()/"src"))
    from wheeled_tasks.wheelbipe_port.model import expose_source
    expose_source()
    from isaaclab.app import AppLauncher
    app = AppLauncher(dict(headless=True,device="cuda:0")).app
    started = time.monotonic()
    env = None
    result = dict(status="preparing",ppo_updates=0,hardware_io=False,behavior_qualified=False,
        comparison_scope="existing August checkpoint with current vendored V14 task; not an exact historical replay",
        checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        asset=str(args.asset),asset_sha256=hashlib.sha256(args.asset.read_bytes()).hexdigest(),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    def save():
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    try:
        import torch
        from pxr import UsdPhysics
        from isaaclab.utils.math import quat_apply
        from agent_tasks.direct.wheelbipe.wheelbipe_V14.env import WheelbipeV14Env
        from agent_tasks.direct.wheelbipe.wheelbipe_V14.env_cfg import WheelbipeV14FlatEnvCfg
        from wheeled_tasks.wheelbipe_port.environment import OfflineTerrainImporter
        class ReferenceEnv(WheelbipeV14Env):
            def _on_command_updated(self):
                super()._on_command_updated()
                if hasattr(self,"eval_command"):
                    vx,wz,height = self.eval_command
                    self.command[:,0],self.command[:,1],self.command[:,2] = vx,0.,wz
                    self.height_cmd.fill_(height)
        cfg = WheelbipeV14FlatEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed = 110
        cfg.robot_cfg.spawn.usd_path = str(args.asset.resolve())
        cfg.terrain.class_type = OfflineTerrainImporter
        cfg.sim.dt,cfg.decimation = .005,4
        cfg.sim.render_interval = 4
        cfg.use_leg_random_start = False
        cfg.commands.special_modes = {}
        cfg.events.reset_base.params["pose_range"] = {}
        cfg.events.reset_base.params["velocity_range"] = {}
        for name,value in vars(cfg.events).items():
            if hasattr(value,"mode") and name!="reset_base":
                setattr(cfg.events,name,None)
        cfg.use_self_obs_noise,cfg.use_self_act_noise = False,False
        cfg.use_obs_delay,cfg.use_act_delay = False,False
        cfg.spring_settings["random_force"] = None
        cfg.play_wheel_material_debug = False
        cfg.play_wheel_motor_z_axis_align_debug = False
        env = ReferenceEnv(cfg)
        state = torch.load(args.checkpoint,map_location=env.device,weights_only=False)["model_state_dict"]
        layers = []
        for i,(inputs,outputs) in enumerate(((35,256),(256,128),(128,64),(64,6))):
            layer = torch.nn.Linear(inputs,outputs,device=env.device)
            layer.weight.data.copy_(state[f"actor.{i*2}.weight"])
            layer.bias.data.copy_(state[f"actor.{i*2}.bias"])
            layers.append(layer)
            if i<3:
                layers.append(torch.nn.ELU())
        actor = torch.nn.Sequential(*layers).eval()
        constraints = []
        for prim in env.sim.stage.Traverse():
            if not str(prim.GetPath()).startswith("/World/envs/env_0/") or not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            if not joint.GetExcludeFromArticulationAttr().Get():
                continue
            names = [str(p).split("/")[-1] for rel in (joint.GetBody0Rel(),joint.GetBody1Rel()) for p in rel.GetTargets()]
            if len(names)==2 and all(n in env.robot.body_names for n in names):
                constraints.append(dict(path=str(prim.GetPath()),bodies=names,
                    local0=list(joint.GetLocalPos0Attr().Get()),local1=list(joint.GetLocalPos1Attr().Get())))
        if not constraints:
            raise ValueError("No measurable excluded loop joints in the reference asset")
        idx0 = [env.robot.body_names.index(c["bodies"][0]) for c in constraints]
        idx1 = [env.robot.body_names.index(c["bodies"][1]) for c in constraints]
        pos0 = torch.tensor([c["local0"] for c in constraints],device=env.device)
        pos1 = torch.tensor([c["local1"] for c in constraints],device=env.device)
        result.update(status="evaluating",constraints=constraints,policy_hz=50,physics_hz=200,
            solver=[cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count,
                    cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count],
            policy_joint_order=[env.robot.joint_names[i] for i in env._actuate_idx],
            cases=[])
        save()
        cases = [("stand_low",0.,0.,.23),("stand_mid",0.,0.,.305),("stand_high",0.,0.,.42),
                 ("forward",.5,0.,.305),("reverse",-.5,0.,.305),
                 ("spin_positive",0.,100.*2.*torch.pi/60.,.305),
                 ("spin_negative",0.,-100.*2.*torch.pi/60.,.305)]
        for name,vx,wz,height in cases:
            env.eval_command = vx,wz,height
            with torch.inference_mode():
                obs,_ = env.reset()
            values,reset_count = [],0
            for _ in range(args.steps):
                with torch.inference_mode():
                    obs,reward,terminated,truncated,_ = env.step(actor(obs["policy"]))
                a = env.robot.data.body_pos_w[:,idx0]+quat_apply(env.robot.data.body_quat_w[:,idx0],pos0.expand(args.num_envs,-1,-1))
                b = env.robot.data.body_pos_w[:,idx1]+quat_apply(env.robot.data.body_quat_w[:,idx1],pos1.expand(args.num_envs,-1,-1))
                reset_count += int((terminated|truncated).sum())
                values.append(torch.stack((env.robot.data.root_lin_vel_b[:,0].mean(),
                    env.robot.data.root_ang_vel_b[:,2].mean(),env.robot.data.root_pos_w[:,2].mean(),
                    (a-b).norm(dim=-1).max())))
            data = torch.stack(values)
            if not torch.isfinite(data).all():
                raise ValueError("Nonfinite reference diagnostic")
            result["cases"].append(dict(case=name,command=[vx,wz,height],envs=args.num_envs,
                duration_s=args.steps*env.step_dt,resets=reset_count,
                actual_vx_mean_m_s=float(data[:,0].mean()),actual_yaw_mean_rad_s=float(data[:,1].mean()),
                actual_height_mean_m=float(data[:,2].mean()),policy_sample_gap_max_m=float(data[:,3].max())))
            save()
        result.update(status="evaluation_complete",elapsed_s=time.monotonic()-started)
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

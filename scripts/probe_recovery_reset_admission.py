#!/usr/bin/env python3
"""Read-only native passive-reset admission; no actor, optimizer or checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"scripts"),str(ROOT/"src")]
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--plan",type=Path,default=ROOT/"contracts/wheel_leg_prone_motion_v4.json")
    p.add_argument("--seconds",type=float,default=.15)
    p.add_argument("--height-shift-m",type=float,default=0.)
    p.add_argument("--headings",type=float,nargs="+",default=[0.])
    p.add_argument("--pose-indices",type=int,nargs="+")
    p.add_argument("--trace-physics",action="store_true")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from train_prone_motion import configuration
    from train_scratch_standing import write_json,digest
    plan=json.loads(a.plan.read_text())
    bank=json.loads((ROOT/plan["pose_bank"]["file"]).read_text())
    report=dict(schema="recovery-passive-admission-v1",status="starting",new_PPO=0,checkpoint_reads=[],
        seconds=a.seconds,height_shift_m=a.height_shift_m,plan_sha256=digest(a.plan),
        bank_sha256=digest(ROOT/plan["pose_bank"]["file"]),profiles="nominal",cases=[],
        policy_hz=100,physics_hz=200,closure_limit_m=.003,hardware_ready=False)
    launcher=env=None
    try:
        from isaaclab.app import AppLauncher
        launcher=AppLauncher(dict(headless=True,device="cuda:0",enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import torch
        import numpy as np
        import warp as wp
        from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        torch.set_num_threads(4)
        n=256
        cfg,settings,bankpath=configuration(a.output/"config",plan,num_envs=n,device="cuda:0")
        # Continue measuring the original disabled placement without actor enable.
        settings.update(passive_min_seconds=a.seconds+1,passive_max_seconds=a.seconds+2)
        env=ProneMotionEnv(cfg,json.loads((ROOT/cfg["asset_directory"]/"manifest.json").read_text()),
            load_chassis_control(ROOT/cfg["control_math_source"]),ROOT,stage_name="speed",num_envs=n,
            device="cuda:0",seed=plan["seed"],level=1.,run_role="train",bank=bankpath,settings=settings)
        if a.height_shift_m:env.bank_roots[:,2] += a.height_shift_m
        limits=wp.to_torch(env.robot.root_view.get_dof_limits())
        spec=json.loads((ROOT/cfg["asset_directory"]/"model_spec.json").read_text())
        report["live_knee_limits"]={name:dict(model_rad=limits[0,j].tolist(),
            included_deg=sorted((sign*limits[0,j]+spec["passive_knee_zero_inner_rad"][name]).rad2deg().tolist()))
            for name,sign,j in zip(("L_joint2","R_jonit2"),(1.,-1.),env.knee_ids)}
        report["live_solver_iterations"]=env.startup_report["solver_iterations"]
        report["runtime_closure_guard_m"]=env.getup_cfg["closure_gap_m"]
        indices=a.pose_indices if a.pose_indices is not None else [i for i,r in enumerate(bank["poses"]) if r.get("recovery")]
        allcases=[dict(pose_index=i,heading_rad=h,recovery=True,role="prone",pitch_deg=0.,
            target=[0.,0.,.305],command=[0.,0.,.305],birth_height=bank["poses"][i]["root_position_m"][2])
            for i in indices for h in a.headings]
        with torch.inference_mode():
            for offset in range(0,len(allcases),n):
                batch=allcases[offset:offset+n];env.evaluation_cases=[batch[i%len(batch)] for i in range(n)]
                env.reset(torch.arange(n,device=env.device))
                rows=[dict(pose_index=c["pose_index"],heading_rad=c["heading_rad"],
                    recovery=bank["poses"][c["pose_index"]].get("recovery",{}),terminated=False,
                    closure_peak_m=0.,passive_motor_peak_nm=0.,mechanical=False,first_failure_s=None)
                    for c in batch]
                active=torch.ones(n,dtype=torch.bool,device=env.device)
                samples=[]
                original_gap=env.closure_gap
                def measured_gap():
                    gap=original_gap()
                    if a.trace_physics:
                        force=env.contact_force[:len(batch)].norm(dim=-1)
                        samples.append(dict(gap=gap[:len(batch)].cpu().numpy().copy(),
                            knee_q=env.robot.data.joint_pos.torch[:len(batch),env.knee_ids].cpu().numpy().copy(),
                            knee_dq=env.robot.data.joint_vel.torch[:len(batch),env.knee_ids].cpu().numpy().copy(),
                            body_force=force.cpu().numpy().copy(),
                            passive=env.passive[:len(batch)].cpu().numpy().copy()))
                    return gap
                env.closure_gap=measured_gap
                for _ in range(round(a.seconds/env.policy_dt)):
                    _,_,done,extra=env.step(env.actions[:,[0,1,3,4,2,5]].clone())
                    d=extra["diagnostics"]
                    for i,row in enumerate(rows):
                        if not bool(active[i]):continue
                        row["closure_peak_m"]=max(row["closure_peak_m"],float(d["gap"][i]))
                        row["passive_motor_peak_nm"]=max(row["passive_motor_peak_nm"],float(d["passive_motor_peak"][i]))
                        row["mechanical"] |= bool(d["mechanical"][i])
                        row["terminated"] |= bool(done[i])
                        if bool(done[i]):row["first_failure_s"]=float(d["time"][i])
                    active &= ~done
                for row in rows:
                    row["passed"]=not row["terminated"] and row["closure_peak_m"]<=.003 and row["passive_motor_peak_nm"]==0.
                env.closure_gap=original_gap
                if samples:
                    np.savez_compressed(a.output/f"physics_{offset:04d}.npz",**{
                        key:np.stack([sample[key] for sample in samples]) for key in samples[0]})
                    report["trace_metadata"]=dict(physics_dt=env.dt,knee_names=["L_joint2","R_jonit2"],
                        knee_zero_inner_rad=spec["passive_knee_zero_inner_rad"],body_names=env.robot.body_names)
                report["cases"].extend(rows)
                print(json.dumps(dict(measured=len(report["cases"]),passed=sum(r["passed"] for r in report["cases"]))),flush=True)
        report.update(status="completed",all_pass=all(r["passed"] for r in report["cases"]),
            policy_hz=1/env.policy_dt,physics_hz=1/env.dt)
    except BaseException:
        report.update(status="failed",error=traceback.format_exc());raise
    finally:
        write_json(a.output/"report.json",report)
        if env is not None:env.close()
        if launcher is not None:launcher.app.close()


if __name__=="__main__":main()

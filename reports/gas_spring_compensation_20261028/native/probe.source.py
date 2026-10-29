#!/usr/bin/env python3
"""Compare the real V5 torque controller with/without gas FF in native PhysX.

No policy or hardware. Fixed root, zero gravity, no ground/contact/friction;
closed-chain passive coordinates are written only once per case at reset.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
sys.path.insert(0,str(ROOT / "tools"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"probe.source.py").write_bytes(Path(__file__).read_bytes())
    import numpy as np
    from identify_gas_spring import geometry
    cfg = json.loads((ROOT / "contracts/scratch_standing_base_mechanism_v2.json").read_text())
    bundle = ROOT/cfg["asset_directory"]
    manifest, spec, fit = (json.loads((bundle/n).read_text()) for n in ("manifest.json","model_spec.json","fit_10mpa.json"))
    prior = json.loads((ROOT/cfg["control_math_source"]).read_text())
    model_path = ROOT/"model/gas_spring_compensation_20261028/model.json"
    setting = dict(enabled=True,model_file=str(model_path.relative_to(ROOT)),model_sha256=digest(model_path),
        gain=1.,force_extrapolation="positive_c1_endpoint_trend_v1",
        physical_curve_outside_geometry="hold_geometry_endpoint")
    betas = [42.,50.,60.,75.,90.,98.,104.,106.5]
    solved = geometry(bundle,betas)
    report = dict(status="starting",started_at=datetime.now(timezone.utc).isoformat(),
        fixture="native PhysX fixed root, no gravity, ground, contact or friction; unrestrained closed-chain legs",
        gravity=[0.,0.,0.],policy_used=False,hardware_used=False,dt_s=.001,seconds_per_case=1.5,
        fixture_motor_armature_kg_m2=.02,armature_identified=False,motor_limit_nm=40.,
        passive_pose_writes_during_physics=0,reset_pose_writes_per_case=1,
        model_sha256=digest(model_path),asset_manifest_sha256=digest(bundle/"manifest.json"),
        model_spec_sha256=digest(bundle/"model_spec.json"),right_leg_symmetry_assumed=True,
        script_sha256=digest(__file__),versions={n:importlib.metadata.version(n)
            for n in ("isaacsim","isaaclab","isaaclab-physx","torch")},cases=[])
    start=time.monotonic()
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(dict(headless=True,device="cpu",enable_cameras=False))
    try:
        import torch
        import warp as wp
        from pxr import UsdPhysics
        import isaaclab.sim as su
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab_physx.physics import PhysxCfg
        from wheeled_tasks.chassis.v5_control import V5Control
        torch.set_num_threads(4)
        settings=deepcopy(cfg["v5_control"])
        settings["gas_spring_compensation"]=setting
        on=V5Control(manifest,spec,fit,prior,settings,"cpu",root=ROOT)
        off=V5Control(manifest,spec,fit,prior,cfg["v5_control"],"cpu",root=ROOT)
        sim=su.SimulationContext(su.SimulationCfg(device="cpu",dt=.001,gravity=(0.,0.,0.),
            render_interval=100,physics=PhysxCfg(solver_type=1,enable_external_forces_every_iteration=True)))
        active=manifest["control_joint_names"];springs=manifest["spring_joint_names"]
        passive=[n for n in manifest["tree_joint_names"] if n not in active+springs]
        def actuator(names,limit,armature=0.):
            return IdealPDActuatorCfg(joint_names_expr=names,stiffness=0.,damping=0.,armature=armature,
                effort_limit=limit,effort_limit_sim=limit,velocity_limit_sim=1000.)
        robot=Articulation(ArticulationCfg(prim_path="/World/Robot",
            spawn=su.UsdFileCfg(usd_path=str(bundle/"robot.usda"),
                articulation_props=su.ArticulationRootPropertiesCfg(fix_root_link=True,
                    enabled_self_collisions=False,solver_position_iteration_count=64,solver_velocity_iteration_count=32)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,2.),joint_pos=manifest["nominal_joint_pos"],joint_vel={".*":0.}),
            actuators=dict(active=actuator(active,40.,.02),passive=actuator(passive,100.),gas=actuator(springs,1500.)),
            soft_joint_pos_limit_factor=1.))
        loops=[p for p in sim.stage.Traverse() if p.IsA(UsdPhysics.SphericalJoint)]
        assert len(loops)==6
        sim.reset()
        friction=wp.to_torch(robot.root_view.get_dof_friction_properties())
        assert float(friction.abs().max())<1e-12
        report.update(external_loop_constraints=len(loops),asset_friction_abs_max=float(friction.abs().max()))
        ids=[robot.joint_names.index(n) for n in active]
        spring_ids=[robot.joint_names.index(n) for n in springs]
        knee_ids=[robot.joint_names.index(n) for n in ("L_joint2","R_jonit2")]
        traces={}
        for k,beta in enumerate(betas):
            pose=dict(zip(solved["names"].tolist(),solved["poses"][k]))
            q0=torch.tensor([[pose[n] for n in robot.joint_names]],dtype=torch.float32)
            for enabled,control in ((False,off),(True,on)):
                label=f"beta{int(beta)}_{'compensated' if enabled else 'baseline'}"
                robot.reset()
                robot.write_joint_position_to_sim_index(position=q0)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
                robot.update(0.)
                target=q0[:,ids][:,list(control.LEGS)].clone()
                effort=torch.zeros_like(q0)
                rows=[];peak=0.
                for step in range(1500):
                    q=robot.data.joint_pos.torch
                    dq=robot.data.joint_vel.torch
                    torque=control.motor_efforts(q[:,ids],dq[:,ids],target,torch.zeros(1,2))
                    assert torch.isfinite(torque).all() and float(torque.abs().max())<=40.
                    peak=max(peak,float(torque.abs().max()))
                    effort.zero_();effort[:,ids]=torque
                    effort[:,spring_ids]=on.spring_efforts(q[:,spring_ids])
                    robot.set_joint_effort_target_index(target=effort)
                    robot.write_data_to_sim();sim.step(render=False);robot.update(.001)
                    if step%5==0:
                        qn=robot.data.joint_pos.torch
                        errors=(target-qn[:,ids][:,list(control.LEGS)])[0]
                        actual_beta=torch.rad2deg(qn[0,knee_ids]*torch.tensor([1.,-1.])+torch.pi-2.3573)
                        rows.append([step*.001,*errors.tolist(),*actual_beta.tolist(),*torque[0,list(control.LEGS)].tolist()])
                trace=np.asarray(rows)
                traces[label]=trace
                steady=trace[trace[:,0]>=1.25]
                report["cases"].append(dict(reference_beta_deg=beta,compensation_enabled=enabled,
                    rms_error_rad=np.sqrt(np.mean(steady[:,1:5]**2,axis=0)).tolist(),
                    mean_actual_beta_deg=steady[:,5:7].mean(axis=0).tolist(),
                    mean_motor_torque_nm=steady[:,7:11].mean(axis=0).tolist(),peak_torque_nm=peak))
        before=np.array([v["rms_error_rad"] for v in report["cases"] if not v["compensation_enabled"]])
        after=np.array([v["rms_error_rad"] for v in report["cases"] if v["compensation_enabled"]])
        report.update(status="completed",baseline_joint_rms_rad=float(np.sqrt(np.mean(before**2))),
            compensated_joint_rms_rad=float(np.sqrt(np.mean(after**2))),
            improvement_ratio=float(np.sqrt(np.mean(after**2)/np.mean(before**2))),
            physical_steps=round(1500*2*len(betas)),control="production V5Control.motor_efforts + V5Control.spring_efforts",
            elapsed_s=time.monotonic()-start)
        report["passed"]=bool(report["improvement_ratio"]<.05 and report["compensated_joint_rms_rad"]<.002)
        np.savez_compressed(args.output/"traces.npz",**traces)
        report["traces_sha256"]=digest(args.output/"traces.npz")
        report["trace_columns"]=["t_s","error_LH_rad","error_LA_rad","error_RH_rad","error_RA_rad",
            "beta_left_deg","beta_right_deg","tau_LH_nm","tau_LA_nm","tau_RH_nm","tau_RA_nm"]
        (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
        print(json.dumps({key:report[key] for key in ("status","passed","baseline_joint_rms_rad",
            "compensated_joint_rms_rad","improvement_ratio")}),flush=True)
        if not report["passed"]:
            raise RuntimeError("Native gas compensation residual exceeds fixture criteria")
    finally:
        launcher.app.close()


if __name__=="__main__":
    main()

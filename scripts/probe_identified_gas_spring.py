#!/usr/bin/env python3
"""Bounded, gravity-free native PhysX force-to-motor calibration and validation.

No neural policy, production configuration, ground contact, spring feedforward,
or passive joint pose writes during physics. Each case resets a solved pose once
then uses explicit motor PD to reach free static equilibrium under spring force.
"""
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

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from identify_gas_spring import BUNDLE, digest, force


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--mode",choices=("unit","curves"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=1.2)
    p.add_argument("--betas",type=float,nargs="+",help="Actual-pose sampling references in degrees")
    a=p.parse_args()
    assert .6 <= a.seconds <= 3.
    g=np.load(a.data/"geometry.npz")
    meta=json.loads((a.data/"geometry.json").read_text());bundle=Path(meta["bundle"])
    assert all(digest(bundle/n)==value for n,value in meta["files_sha256"].items())
    manifest=json.loads((bundle/"manifest.json").read_text())
    spec=json.loads((bundle/"model_spec.json").read_text())
    catalogue=json.loads((bundle/"fit_10mpa.json").read_text())
    if a.mode=="unit":
        cases=[dict(model="unit",reference_beta_deg=b,constant_force_n=f) for b in (a.betas or [55.,74.,99.]) for f in (0.,100.)]
    else:
        fitted=json.loads((a.data/"identified_model.json").read_text())
        cases=[dict(model=m,reference_beta_deg=b,constant_force_n=None)
               for m in ("catalogue","monotone_cubic","effective_monotone_table")
               for b in (a.betas or [56.,60.,70.,80.,90.,95.,97.,98.])]
    assert all(g["beta_deg"][0] <= c["reference_beta_deg"] <= g["beta_deg"][-1] for c in cases)
    if a.mode=="curves":
        assert all(56. <= c["reference_beta_deg"] <= 101. for c in cases)
    report=dict(status="starting",mode=a.mode,started_at=datetime.now(timezone.utc).isoformat(),
                pid=os.getpid(),interpreter=sys.executable,gravity=[0.,0.,0.],
                fixture="fixed base, no ground, no self-contact",motor_pd=[160.,2.5],
                fixture_motor_armature_kg_m2=.02,
                armature_scope="measurement fixture only; not an identified robot inertia; equilibrium torque unchanged",
                motor_limit_nm=40.,dt_s=.001,spring_feedforward=False,
                passive_pose_writes_during_physics=0,cases=[],
                asset_manifest_sha256=digest(bundle/"manifest.json"),script_sha256=digest(__file__),
                versions={n:importlib.metadata.version(n) for n in ("isaacsim","isaaclab","isaaclab-physx","torch")})
    report["geometry_sha256"]=digest(a.data/"geometry.npz")
    report["identified_model_sha256_at_start"]=digest(a.data/"identified_model.json") if a.mode=="curves" else None
    report["numerical_domain_tolerance_m"]=2e-5
    report["trace_columns"]=["beta_deg","compression_m","hip_motor_nm","knee_motor_nm",
                             "hip_velocity_rad_s","knee_velocity_rad_s","force_n",
                             "hip_incoming_projected_nm","knee_incoming_projected_nm"]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix(".source.py").write_text(Path(__file__).read_text())
    launcher=None;start=time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        launcher=AppLauncher(dict(headless=True,device="cpu",enable_cameras=False))
        import torch
        from wheeled_tasks.chassis.identified_gas_spring import IdentifiedGasSpringCurve
        import warp as wp
        from pxr import UsdPhysics
        from scipy.spatial.transform import Rotation
        import isaaclab.sim as su
        from isaaclab.assets import Articulation,ArticulationCfg
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab_physx.physics import PhysxCfg
        torch.set_num_threads(4)
        identified_curve=IdentifiedGasSpringCurve.from_json(a.data/"identified_model.json") if a.mode=="curves" else None
        report["identified_force_evaluator"]="IdentifiedGasSpringCurve.force (torch)" if a.mode=="curves" else None
        dt=.001
        sim=su.SimulationContext(su.SimulationCfg(device="cpu",dt=dt,gravity=(0.,0.,0.),
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
        ids=[robot.joint_names.index(n) for n in active]
        gas_id=robot.joint_names.index("L_spring_slide")
        knee_id=robot.joint_names.index("L_joint2")
        s0=spec["spring_binding"]["L_spring_slide"]["compression_at_q_zero_m"]
        report["joint_names"]=robot.joint_names
        report["external_loop_constraints"]=len(loops)
        report["solver_iterations"]=[64,32]
        friction=wp.to_torch(robot.root_view.get_dof_friction_properties()).numpy()
        report["asset_friction_abs_max"]=float(abs(friction).max())
        assert abs(friction).max()<1e-12
        traces=[]
        for number,case in enumerate(cases):
            beta_ref=case["reference_beta_deg"]
            pose={n:np.interp(beta_ref,g["beta_deg"],g["poses"][:,i]) for i,n in enumerate(g["names"])}
            q0=torch.tensor([[pose[n] for n in robot.joint_names]],dtype=torch.float32)
            robot.reset()
            robot.write_joint_position_to_sim_index(position=q0)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
            robot.update(0.)
            target=q0[:,ids].clone()
            if a.mode=="curves":
                compression0=s0-pose["L_spring_slide"]
                if case["model"]=="catalogue":
                    force0=float(np.polynomial.polynomial.polyval(compression0/.08,catalogue["monomial_coefficients_n"]))
                else:
                    force0=float(force(fitted["models"][case["model"]],compression0))
                jac0=np.array([np.interp(beta_ref,g["beta_deg"],g["jacobian_m_per_rad"][:,i]) for i in range(2)])
                # Seed a constant POSITION reference offset to sample the desired
                # actual pose. No online torque FF is added; the recorded motor
                # effort remains the PD holding effort after free equilibrium.
                target[0,:2]+=torch.tensor(-force0*jac0/160.,dtype=torch.float32)
                case["reference_preload_offset_rad"]=(-force0*jac0/160.).tolist()
            effort=torch.zeros_like(q0)
            rows=[];clamps=0;peak_tau=0.;max_gap=0.
            for step in range(round(a.seconds/dt)):
                q=robot.data.joint_pos.torch
                dq=robot.data.joint_vel.torch
                compression=float(s0-q[0,gas_id])
                if a.mode=="unit":
                    F=case["constant_force_n"]
                elif case["model"]=="catalogue":
                    F=float(np.polynomial.polynomial.polyval(np.clip(compression/.08,0.,.9),catalogue["monomial_coefficients_n"]))
                else:
                    model=fitted["models"][case["model"]]
                    lo,hi=model["compression_domain_m"]
                    outside=not lo-2e-5<=compression<=hi+2e-5
                    if outside and step*dt>a.seconds-.25:clamps+=1
                    # Declared numerical/transient endpoint hold; final-domain
                    # violations are reported and excluded from valid comparisons.
                    bounded=np.clip(compression,lo,hi)
                    if case["model"]=="effective_monotone_table":
                        F=float(identified_curve.force(torch.tensor(bounded,dtype=torch.float64)))
                    else:
                        F=float(force(model,bounded))
                if a.mode=="unit":F*=min(1.,step*dt/.15)
                motor=(160.*(target-q[:,ids])-2.5*dq[:,ids]).clamp(-40.,40.)
                peak_tau=max(peak_tau,float(motor.abs().max()))
                effort.zero_();effort[:,ids]=motor;effort[0,gas_id]=F
                robot.set_joint_effort_target_index(target=effort)
                robot.write_data_to_sim();sim.step(render=False);robot.update(dt)
                if step*dt>=a.seconds-.25 and step%5==0:
                    qn=robot.data.joint_pos.torch[0].numpy();vn=robot.data.joint_vel.torch[0].numpy()
                    beta=np.rad2deg(qn[knee_id]+np.pi-2.3573)
                    projected=wp.to_torch(robot.root_view.get_dof_projected_joint_forces()).numpy()[0,ids[:2]]
                    rows.append([beta,s0-qn[gas_id],*motor[0,:2].numpy(),*vn[ids[:2]],F,*projected])
                if step%100==0:
                    poses=robot.data.body_link_pose_w.torch[0].numpy()
                    frames={n:(poses[i,:3],Rotation.from_quat(poses[i,3:])) for i,n in enumerate(robot.body_names)}
                    for c in spec["constraints"]:
                        p0,r0=frames[c["body0"]];p1,r1=frames[c["body1"]]
                        gap=float(np.linalg.norm(p0+r0.apply(c["local_pos0_m"])-p1-r1.apply(c["local_pos1_m"])))
                        max_gap=max(max_gap,gap)
            rows=np.asarray(rows);mean=rows.mean(axis=0)
            record=dict(**case,actual_beta_deg=float(mean[0]),compression_m=float(mean[1]),
                        holding_torque_nm=mean[2:4].tolist(),velocity_rms_rad_s=float(np.sqrt(np.mean(rows[:,4:6]**2))),
                        force_n=float(mean[6]),projected_joint_effort_nm=mean[7:9].tolist(),
                        holding_torque_std_nm=rows[:,2:4].std(axis=0).tolist(),max_loop_gap_m=max_gap,
                        peak_motor_torque_nm=peak_tau,final_domain_clamped_samples=clamps,
                        final_samples=len(rows),settled=bool(np.sqrt(np.mean(rows[:,4:6]**2))<.02 and max_gap<.001 and peak_tau<39.9 and clamps==0))
            report["cases"].append(record);traces.append(rows)
            report["status"]="running";a.output.write_text(json.dumps(report,indent=2)+"\n")
            print("CASE",number+1,"/",len(cases),json.dumps(record),flush=True)
        np.savez_compressed(a.output.with_suffix(".npz"),**{f"case_{i}":x for i,x in enumerate(traces)})
        report["status"]="completed"
        report["all_cases_settled"]=all(c["settled"] for c in report["cases"])
    except Exception:
        report.update(status="failed",error=traceback.format_exc())
        print(report["error"],flush=True)
        raise
    finally:
        report["wall_seconds"]=time.monotonic()-start
        a.output.write_text(json.dumps(report,indent=2)+"\n")
        if launcher is not None:launcher.app.close()


if __name__=="__main__":main()

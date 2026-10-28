#!/usr/bin/env python3
"""Freeze the gas-compensated research mechanism and its bounded native evidence.

This publishes configuration files only. No training or hardware is started.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/"model/gas_spring_compensation_20261028"
REPORT=ROOT/"reports/gas_spring_compensation_20261028"
sys.path.insert(0,str(ROOT/"src"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def publish():
    from wheeled_tasks.chassis.preflight import design_preflight
    parent=ROOT/"contracts/scratch_standing_base_mechanism_v2.json"
    parent_design=ROOT/"contracts/scratch_standing_base_design_v2.json"
    cfg=json.loads(parent.read_text())
    native_path=REPORT/"native_final/report.json"
    native=json.loads(native_path.read_text())
    assert native["status"]=="completed" and native["passed"]
    assert native["model_sha256"]==digest(PACKAGE/"model.json")
    assert native["script_sha256"]==digest(ROOT/"scripts/probe_gas_spring_compensation.py")
    for path,sha in native["source_files_sha256"].items():
        assert digest(ROOT/path)==sha,path
    settings=dict(enabled=True,model_file=relative(PACKAGE/"model.json"),
        model_sha256=digest(PACKAGE/"model.json"),gain=1.,force_extrapolation="positive_c1_endpoint_trend_v1",
        physical_curve_outside_geometry="hold_geometry_endpoint",
        relative_soft_limits=dict(minimum=-.140638855,maximum=1.3889088544,margin=.03,projection="outward_relative_torque_v1"))
    changes=dict(gas_spring_compensation=settings,spring_pressure_mpa=None,
        spring_reserve_model="measured_effective_force_with_C1_endpoint_trend_extension")
    cfg["v5_control"].update(deepcopy(changes))
    cfg["research_profile_id"]="gas-spring-compensated-full-soft-limits-20261028-v2"
    evidence_files=[PACKAGE/"model.json",native_path,REPORT/"native_final/traces.npz",
        REPORT/"native_final/probe.source.py",ROOT/"scripts/probe_gas_spring_compensation.py",
        ROOT/"src/wheeled_tasks/chassis/v5_control.py",ROOT/"src/wheeled_tasks/chassis/gas_spring_compensation.py",
        ROOT/"src/wheeled_tasks/chassis/env.py",
        ROOT/"model/gas_spring_identified_symmetric_20261028/model.json"]
    evidence=dict(schema="gas-spring-compensation-evidence-v1",status="validated_for_research",
        hardware_release=False,settings=settings,native_report=relative(native_path),
        scope="V7 gravity-free native closed-chain static compensation/sign test only; endpoint extension is an unmeasured trend prior; no real-robot or policy qualification",
        files_sha256={relative(path):digest(path) for path in evidence_files})
    write(PACKAGE/"evidence.json",evidence)
    binding=dict(file=relative(PACKAGE/"evidence.json"),sha256=digest(PACKAGE/"evidence.json"))
    cfg["gas_spring_compensation_evidence"]=binding
    design=json.loads(parent_design.read_text())
    design["scope"]="Gas-spring output-torque compensation with same identified physical spring, fresh-training mechanism"
    design["review_basis"]=dict(parent_files_sha256={relative(path):digest(path) for path in (parent,parent_design)},
        change="Gas force model, feedforward and matching RMCS relative effort projection; other parent task/behavior/budget expectations preserved",
        model_sha256=settings["model_sha256"],native_report_sha256=digest(native_path),
        endpoint_extension_measured=False,right_leg_force_symmetry_assumed=True)
    for stage in design["stages"].values():
        for row in stage["roles"].values():
            row["contract"]["v5_control"].update(deepcopy(changes))
            row["contract"]["gas_spring_compensation_evidence"]=deepcopy(binding)
            row["runtime"]["modules"]["gas_spring_compensation"]=True
            row["runtime"]["gas_spring_compensation"]=deepcopy(settings)
            row["runtime"]["gas_spring_force_each_iteration"]=True
    write(PACKAGE/"design.json",design)
    cfg["design_preflight"]=dict(file=relative(PACKAGE/"design.json"),sha256=digest(PACKAGE/"design.json"),stage="speed")
    write(PACKAGE/"mechanism.json",cfg)
    control=json.loads((ROOT/cfg["control_math_source"]).read_text())
    invocation=design["stages"]["speed"]["roles"]["train"]["invocation"]
    receipt=design_preflight(cfg,control,ROOT,stage="speed",role="train",
        seed=invocation["seed"]["value"],num_envs=invocation["num_envs"])
    receipt.update(ppo_updates=0,hardware_motion=False,default_scratch_entrypoint_enabled=True,
        full_robot_policy_qualified=False,component_native_report=relative(native_path))
    write(REPORT/"preflight.json",receipt)
    return receipt


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from wheeled_tasks.chassis.gas_spring_compensation import GasSpringCompensation
    cfg=json.loads((PACKAGE/"mechanism.json").read_text())
    spring=GasSpringCompensation(cfg["v5_control"]["gas_spring_compensation"],root=ROOT,device="cpu",dtype=torch.float64)
    model=spring.model
    beta=np.arange(40.,108.001,.1)
    left,right=(model["sides"][side] for side in ("left","right"))
    dl=np.interp(beta,left["beta_deg"],left["delta_rad"])
    dr=np.interp(beta,right["beta_deg"],right["delta_rad"])
    q=torch.tensor(np.column_stack((np.zeros(len(beta)),dl,np.zeros(len(beta)),-dr)),dtype=torch.float64)
    tau,info=spring.evaluate(q)
    measured=(beta>=56.)&(beta<=101.)
    fig,axes=plt.subplots(2,1,figsize=(9,6.4),sharex=True,layout="constrained")
    for ax,y,label in ((axes[0],info["force_n"][:,0].numpy(),"Axial extension force (N)"),
                        (axes[1],tau[:,0].numpy(),"Left hip holding effort (Nm)")):
        ax.plot(beta[~measured & (beta<56.)],y[~measured & (beta<56.)],"--",color="#d97706",lw=2,label="Endpoint-trend estimate")
        ax.plot(beta[measured],y[measured],color="#1d4ed8",lw=2,label="Measured-domain fitted curve")
        ax.plot(beta[beta>101.],y[beta>101.],"--",color="#d97706",lw=2)
        ax.axvspan(40.,56.,color="#f59e0b",alpha=.1)
        ax.axvspan(101.,108.,color="#f59e0b",alpha=.1)
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.set_xlim(40.,108.)
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Thigh-shank included angle beta (deg)")
    fig.suptitle("Gas spring compensation over the configured soft-limit range\nRight axial curve assumed symmetric; torque directions use each side's geometry")
    fig.savefig(REPORT/"full_range_curve.png",dpi=160)
    plt.close(fig)
    samples=[]
    for b in (40.,50.,56.,60.,75.,90.,98.,101.,104.,108.):
        i=int(round((b-40.)/.1))
        samples.append(dict(beta_deg=b,axial_force_n=float(info["force_n"][i,0]),
            model_holding_effort_nm=tau[i].tolist(),estimated=b<56. or b>101.))
    write(REPORT/"curve_samples.json",samples)


if __name__=="__main__":
    print(json.dumps(publish(),ensure_ascii=False),flush=True)
    plot()

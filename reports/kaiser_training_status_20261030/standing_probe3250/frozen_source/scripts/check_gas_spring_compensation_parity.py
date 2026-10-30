#!/usr/bin/env python3
"""Compare the built RMCS evaluator to Torch across both complete soft limits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from wheeled_tasks.chassis.gas_spring_compensation import GasSpringCompensation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmcs",type=Path,default=Path("/home/noir/Documents/workspace/RMCS"))
    parser.add_argument("--container",default="rmcs_devcontainer-rmcs-develop-1")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/gas_spring_compensation_20261028/parity.json")
    args=parser.parse_args()
    cfg=json.loads((ROOT/"model/gas_spring_compensation_20261028/mechanism.json").read_text())
    setting=cfg["v5_control"]["gas_spring_compensation"]
    spring=GasSpringCompensation(setting,root=ROOT,device="cpu",dtype=torch.float64)
    limits=setting["relative_soft_limits"]
    delta=np.linspace(limits["minimum"],limits["maximum"],2001)
    common=np.random.default_rng(20261028).uniform(-8*np.pi,8*np.pi,(len(delta),2))
    q=np.column_stack((common[:,0],common[:,0]+delta,common[:,1],common[:,1]-delta))
    lines="\n".join(" ".join(format(v,".17g") for v in row) for row in q)+"\n"
    probe="/workspaces/RMCS/rmcs_ws/build/rmcs_core/wheel_leg_gas_spring_parity"
    result=subprocess.run(["docker","exec","-i",args.container,probe],input=lines,text=True,capture_output=True,check=True)
    actual=np.array([[float(v) for v in row.split()] for row in result.stdout.splitlines()])
    torque,diagnostics=spring.evaluate(torch.tensor(q,dtype=torch.float64))
    expected=np.column_stack((torque.numpy(),diagnostics["beta_deg"].numpy()))
    error=np.abs(actual-expected).max(axis=0)
    assert error[:4].max()<1e-10 and error[4:].max()<1e-10
    assert (np.abs(actual[:,:4])>0.).all() and np.abs(actual[:,:4]).max()<40.
    headers=[args.rmcs/"rmcs_ws/src/rmcs_core/src/controller/chassis/wheel_leg"/name
             for name in ("gas_spring_compensation.hpp","gas_spring_model.hpp")]
    files=[ROOT/"src/wheeled_tasks/chassis/gas_spring_compensation.py",*headers,Path(__file__)]
    report=dict(status="passed",samples=len(q),scope="C++ and Torch over complete configured active-difference soft limits, randomized common rotations",
        model_sha256=setting["model_sha256"],active_difference_rad=[float(delta[0]),float(delta[-1])],
        max_abs_torque_difference_nm=float(error[:4].max()),max_abs_beta_difference_deg=float(error[4:].max()),
        peak_compensation_nm=float(np.abs(actual[:,:4]).max()),code_files_sha256={str(path):digest(path) for path in files})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

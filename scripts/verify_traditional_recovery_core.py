#!/usr/bin/env python3
"""Offline cross-language compensation and ABI check. No simulator or motor I/O."""
from __future__ import annotations
import argparse
import ctypes as ct
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from simulate_traditional_self_righting import Controller,digest,save
from wheeled_tasks.chassis.gas_spring_compensation import GasSpringCompensation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rmcs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads((ROOT/'model/wheel_leg_v2.0.2/friction_v1/training/mechanism.json').read_text())
    geometry=json.loads((ROOT/'contracts/full_vehicle_recovery_geometry_v1.json').read_text())
    model=GasSpringCompensation(cfg['v5_control']['gas_spring_compensation'],root=ROOT,device='cpu',dtype=torch.float64)
    folder=args.rmcs/'rmcs_ws/src/rmcs_core/src/controller/wheelleg'
    rng=np.random.default_rng(261031)
    q=rng.uniform(-12.,12.,(2048,4))
    for side,table in enumerate(model.sides):
        delta=rng.uniform(float(table['delta_rad'][0]),float(table['delta_rad'][-1]),len(q))
        if side==0:q[:,1]=q[:,0]+delta
        else:q[:,3]=q[:,2]-delta
    expected=model.evaluate(torch.tensor(q))[0].numpy()
    settings=[1.5,3.,5.,12.,75.,105.,geometry['stand_knee_deg'],geometry['stand_orbit_deg'],
              0.,180.,35.,5.,2.,0.,30.,40.,3.14,0.,.6,50.,30.,1.]
    with tempfile.TemporaryDirectory(prefix='recovery-core-')as tmp:
        library=Path(tmp)/'core.so'
        subprocess.run(['g++','-std=c++23','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
            str(folder/'self_righting_abi.cpp'),'-o',str(library)],check=True)
        controller=Controller(library,geometry['rows'],settings)
        try:
            actual=np.array([controller.compensation(row)for row in q])
            error=float(np.max(np.abs(actual-expected)))
            if error>1e-8:raise AssertionError(f'C++/Torch compensation differs by {error} Nm')
            bad=np.array([float('nan'),0.,0.,0.]);out=np.zeros(4)
            rejected=controller.lib.wheelleg_recovery_compensation(Controller.ptr(bad),Controller.ptr(out))!=0
            if not rejected:raise AssertionError('Nonfinite encoder feedback accepted')
            data=np.r_[q[0],np.zeros(4),actual[0],[0.,0.,-1.],np.zeros(5)]
            data[4]=3.14001
            result=controller.step(data,.001,True)
            if result[15]!=3 or np.any(result[8:14]):raise AssertionError('Overspeed did not latch zero outputs')
            controller.step(data,.001,False)
        finally:controller.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    receipt=dict(schema='traditional-recovery-core-offline/v1',passed=True,samples=len(q),
        compensation_max_abs_error_nm=error,nonfinite_encoder_rejected=rejected,
        measured_overspeed_fault_checked=True,abi_version=6,ppo_updates=0,hardware_motion=False,
        sources={str(p):digest(p)for p in [Path(__file__),*(folder/n for n in
            ('self_righting_core.hpp','self_righting_abi.cpp','recovery_geometry_v202.hpp'))]})
    save(args.output,receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()

"""Split recorded wheel vibration into forward-balance and yaw components."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter

ROOT=Path(__file__).resolve().parents[2]
REPORTS=Path(__file__).parent
RUN=ROOT/"runs/unified_robust_20261022/constant_wheel_static086"
sys.path[:0]=[str(ROOT/"scripts")]
from analyze_robust_getup import hf_rms
from evaluate_robust_getup_batch import quiet_filter_variants


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    results={}
    writer=SummaryWriter(str(RUN),filename_suffix=".wheel_direction089")
    for name,alpha in quiet_filter_variants([1.,.32]).items():
        directory=RUN/name
        report=json.loads((directory/"report.json").read_text())
        assert sha(directory/"trajectory.npz")==report["trajectory_sha256"]
        with np.load(directory/"trajectory.npz") as data:
            cases={}
            for case in ["stand_low","stand","stand_high"]:
                lanes=[r["lane"] for r in report["lanes"] if r["name"]==case]
                values={}
                for source,key,scale,min_hz in [("target20_95","actions",10.,20.),
                                               ("actual3_95","motor_velocity",1.,3.)]:
                    lane_values=[]
                    for lane in lanes:
                        time=data["time"][:,lane]
                        keep=(time>=3.)&(time<11.)&data["alive"][:,lane]
                        dt=float(np.median(np.diff(time[keep])))
                        pair=data[key][keep,lane][:,[2,5]]*scale
                        forward=(pair[:,:1]-pair[:,1:])*.5
                        yaw=(pair[:,:1]+pair[:,1:])*.5
                        full,balance,turn=[hf_rms(x,dt,min_hz) for x in [pair,forward,yaw]]
                        residual=abs(full**2-balance**2-turn**2)
                        assert residual<1e-8
                        lane_values.append((full**2,balance**2,turn**2,residual))
                    total,balance,turn,residual=np.mean(lane_values,axis=0)
                    values[source]=dict(full_rms_rad_s=float(np.sqrt(total)),
                        forward_balance_rms_rad_s=float(np.sqrt(balance)),yaw_rms_rad_s=float(np.sqrt(turn)),
                        yaw_energy_fraction=float(turn/total),energy_identity_residual=float(residual))
                    for metric,value in values[source].items():
                        writer.add_scalar(f"Diagnosis/WheelDirection089/{name}/{case}/{source}/{metric}",value,505)
                cases[case]=values
            results[name]=dict(alpha=alpha,cases=cases,trajectory_sha256=report["trajectory_sha256"])
    writer.close()
    result=dict(schema="wheel089-direction-decomposition-v1",window_seconds=[3.,11.],
        wheel_control_indices=[2,5],forward_balance="(left-right)/2",yaw="(left+right)/2",
        recordings=results,new_macro_updates=0,
        scope="Readonly modal energy decomposition, not a causal yaw-only filter test or randomized coverage")
    (REPORTS/"wheel_direction089.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()

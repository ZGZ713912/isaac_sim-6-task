#!/usr/bin/env python3
"""Standing displacement and measured vibration, separate from upright survival."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from train_robust_getup import write_json,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from wheeled_algo.standing_acceptance import excursion_limit
    protocol=json.loads((ROOT/"reports/unified_robust_20261021/acceptance_protocol004.json").read_text())
    report=json.loads((args.run/"evaluation/report.json").read_text())
    summary=json.loads((args.run/"evaluation/gap_summary.json").read_text())
    progress=json.loads((args.run/"progress.json").read_text())
    assert progress["status"]=="completed"
    checkpoint=Path(progress["checkpoint"])
    assert digest(checkpoint)==progress["checkpoint_sha256"]
    learning_clock=torch.load(checkpoint,map_location="cpu",weights_only=False)["infos"]["completed_updates"]
    trace_path=args.run/"evaluation/trajectory.npz"
    assert digest(trace_path)==report["trajectory_sha256"]
    trace=np.load(trace_path)
    rows=[]
    writer=SummaryWriter(str(args.run/"evaluation/metrics"),filename_suffix=".standing-control-quality")
    for lane,old in zip(report["lanes"],summary["rows"],strict=True):
        case=lane["behavior_case"]
        if not case.startswith("stand"):
            continue
        i=lane["lane"]
        active=trace["alive"][:,i]&~trace["passive"][:,i]
        tail=active&(trace["time"][:,i]>=report["seconds"]-1.)
        limit=excursion_limit(protocol,case)
        row=dict(name=lane["name"],case=case,survived=lane["survived"],
            excursion_limit_m=limit,max_excursion_m=old["max_position_excursion_m"],
            tail_stable_fraction=old["tail_stable_fraction"],
            position_within_limit=old["max_position_excursion_m"]<=limit,
            pose_and_position_pass=(lane["survived"] and old["tail_stable_fraction"]>=.9
                                    and old["max_position_excursion_m"]<=limit),
            leg_speed_peak_rad_s=old["max_leg_speed_rad_s"])
        if tail.sum()>=190:
            q=np.unwrap(trace["motor_position"][tail,i][:,[0,1,3,4]],axis=0)
            spectrum=np.fft.rfft(q-q.mean(0),axis=0)
            freq=np.fft.rfftfreq(len(q),.005)
            for low in (3.,20.):
                filtered=spectrum.copy()
                filtered[(freq<low)|(freq>95.)]=0.
                row[f"tail1s_leg_motion_{int(low)}_95hz_rms_rad"]=float(
                    np.sqrt(np.mean(np.fft.irfft(filtered,n=len(q),axis=0)**2)))
            row["tail1s_leg_speed_rms_rad_s"]=float(np.sqrt(np.mean(trace["motor_velocity"][tail,i][:,[0,1,3,4]]**2)))
            row["tail1s_leg_effort_rms_nm"]=float(np.sqrt(np.mean(trace["motor_effort"][tail,i][:,[0,1,3,4]]**2)))
            row["tail1s_saturation_fraction"]=float(trace["leg_saturation_fraction"][tail,i].mean())
        for key,value in row.items():
            if isinstance(value,(int,float)):
                writer.add_scalar("GapStanding/"+row["name"]+"/"+key,value,learning_clock)
        rows.append(row)
    writer.close()
    out=dict(schema="gap-standing-control-quality-v1",run=str(args.run),seconds=report["seconds"],
        learning_clock=learning_clock,
        trajectory_sha256=digest(trace_path),rows=rows,episodes=len(rows),
        pose_and_position_pass=sum(r["pose_and_position_pass"] for r in rows),
        pose_gate="gravity within15degrees, body angular speedXY<.7rad/s, height>.19m, two wheel contacts and four leg speeds<3rad/s",
        scope="Position/pose preliminary checks only. Measured vibration is reported separately; short duration cannot establish long-term recentering.",
        formal_acceptance=False,full_training_admitted=False,hardware_ready=False)
    write_json(args.output,out)
    print(json.dumps({k:v for k,v in out.items() if k!="rows"},ensure_ascii=False))


if __name__=="__main__":
    main()

"""Recorded recovery kinematics, numerical sensitivity and training-noise scope."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT=Path(__file__).resolve().parents[2]
REPORTS=Path(__file__).parent
RUNS=ROOT/"runs/unified_robust_20261022"
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts"),str(REPORTS)]
from export_robust_getup import load_mean
from iteration045_runner import write
from wheeled_algo.robust_getup import SensorConditionedMean,tilt_gate


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(directory):
    report=json.loads((directory/"report.json").read_text())
    assert sha(directory/"trajectory.npz")==report["trajectory_sha256"]
    return report,np.load(directory/"trajectory.npz")


def main():
    torch.set_num_threads(1)
    root=RUNS/"recovery_common_step088"
    old,old_data=read(RUNS/"dynamic_recovery_target079/robust")
    rows={};matching={}
    writer=SummaryWriter(str(root),filename_suffix=".recovery091")
    for directory in sorted(p for p in root.iterdir() if (p/"report.json").exists() and p.name!="all"):
        report,data=read(directory);cases={}
        for case in ["moving_push_left","moving_push_right","inverted_asymmetric"]:
            row=next(r for r in report["lanes"] if r["name"]==case);lane=row["lane"]
            time=data["time"][:,lane];end=row["termination_time_s"] or 12.
            keep=data["alive"][:,lane]&(time>=max(0.,end-.2))
            dq=data["motor_velocity"][keep,lane][:,[0,1,3,4]].reshape(-1,2,2)
            values=dict(terminal_time_s=row["termination_time_s"],
                preterminal200ms_common_speed_peak_rad_s=float(np.abs(dq.mean(-1)).max()),
                preterminal200ms_knee_half_speed_peak_rad_s=float(np.abs(np.diff(dq,axis=-1)*.5).max()),
                preterminal200ms_leg_speed_peak_rad_s=float(np.abs(dq).max()),
                preterminal200ms_roll_speed_peak_rad_s=float(np.abs(data["omega"][keep,lane,0]).max()),
                contact_impulse_peak_ns=row["max_contact_impulse_ns"])
            cases[case]=values
            for key,value in values.items():
                if isinstance(value,(float,int)):
                    writer.add_scalar(f"Diagnosis/Recovery091/{directory.name}/{case}/{key}",value,569)
            if directory.name=="robust":
                old_lane=next(r["lane"] for r in old["lanes"] if r["name"]==case)
                differences={}
                for key in ["observations","actions","motor_position","motor_velocity","local_position"]:
                    x,y=old_data[key][:1000,old_lane],data[key][:1000,lane]
                    delta=np.abs(x-y);per_frame=delta.max(-1);indices=np.flatnonzero(per_frame>0.)
                    entry=dict(first_different_input_frame=int(indices[0]) if len(indices) else None,
                        first_frame_max_difference=float(per_frame[0]),prepush_max_difference=float(per_frame.max()))
                    if key=="motor_position":
                        leg_delta=(x-y)[:,[0,1,3,4]]
                        entry["prepush_leg_circular_difference_rad"]=float(np.abs(np.arctan2(np.sin(leg_delta),np.cos(leg_delta))).max())
                    differences[key]=entry
                matching[case]=differences
        rows[directory.name]=dict(cases=cases,trajectory_sha256=report["trajectory_sha256"])
    checkpoint=RUNS/"pilot033_dynamic_recovery079/model_final.pt"
    state=torch.load(checkpoint,map_location="cpu",weights_only=False);mean,info=load_mean(checkpoint)
    task=state["infos"]["settings"]
    stochastic=SensorConditionedMean(mean,task["quiet_exploration_fraction"],task["quiet_wheel_exploration_fraction"])
    with torch.no_grad():stochastic.log_std.copy_(state["actor_state_dict"]["mlp.log_std"])
    source=RUNS/"dynamic_recovery_full079/robust"
    _,data=read(source)
    obs=torch.from_numpy(data["observations"][::11].reshape(-1,35).copy())
    with torch.inference_mode():
        outputs=stochastic(obs);gate=tilt_gate(obs,mean.stance_aware_recovery,mean.recovery_shape_aware)
    assert torch.equal(outputs[:,0],mean(obs))
    keep=gate[:,0]>=.999
    std=outputs[keep,1,:4].exp().reshape(-1,2,2)*.25
    # Predecoder physical noise; hard clipping/paired projection can reduce it.
    common_std=.5*std.square().sum(-1).sqrt()
    knee_std=std.square().sum(-1).sqrt()
    noise=dict(full_recovery_recorded_inputs=int(keep.sum()),
        per_leg_predecoder_std_rad=std.mean((0,1)).tolist(),
        common_angle_predecoder_std_rad=common_std.mean(0).tolist(),
        knee_difference_predecoder_std_rad=knee_std.mean(0).tolist(),
        maximum_mean_common_increment_rad=.3*.25,bounded_mean_knee_increment_rad=.15,
        alternative_leg_fraction=.25,
        alternative_common_angle_predecoder_std_rad=(common_std.mean(0)*.25).tolist(),
        alternative_knee_difference_predecoder_std_rad=(knee_std.mean(0)*.25).tolist(),
        mean_actions_exact=True,scope="Sampling scale only; not actual motion or proof that smaller noise improves learning")
    for key,value in noise.items():
        if isinstance(value,(float,int)):writer.add_scalar("Diagnosis/Recovery091/sampling/"+key,value,569)
    writer.close()
    value=dict(schema="recovery091-recorded-diagnosis-v1",status="completed",kinematics=rows,
        same079_checkpoint_across_nominal_batch_sizes=matching,training_sampling=noise,
        checkpoint_sha256=sha(checkpoint),new_macro_updates=0,robust_consumed=1384,full_training_admitted=False,
        limits=["Cross-run first actions differ by rounding despite equal first observations; unique source not proven",
            "Within088 variants, prepush1000frames are exact; all smaller common steps are rejected",
            "Common/knee speed and body roll are correlations, not contact-body attribution",
            "A lower recovery-noise fraction requires a bounded zero-PPO native exposure test before further PPO"])
    write(REPORTS/"recovery091.json",value);print(json.dumps(value),flush=True)


if __name__=="__main__":main()

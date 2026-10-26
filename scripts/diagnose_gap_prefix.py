#!/usr/bin/env python3
"""Controlled prefix comparisons for entry authority or torque-delay diagnosis."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from compare_gap_robustness import read
from train_robust_getup import write_json, digest


def metrics(trace, lane, steps):
    import numpy as np
    alive = trace["alive"][:steps, lane] & ~trace["passive"][:steps, lane]
    indices = np.flatnonzero(alive)
    row = dict(alive_at_end=bool(trace["alive"][steps-1,lane] & ~trace["terminated"][steps-1,lane]),
        ever_recovered=bool(trace["recovered"][:steps,lane].any()))
    if not len(indices):
        return row
    first = indices[0]
    initial = alive & (np.arange(steps) <= first+40)
    tail = alive & (np.arange(steps) >= steps-200)
    position = trace["local_position"][:steps,lane,:2]
    row.update(max_leg_speed_rad_s=float(trace["leg_speed_peak_rad_s"][:steps,lane][alive].max()),
        initial_200ms_effort_peak_nm=float(trace["leg_effort_peak_nm"][:steps,lane][initial].max()),
        saturation_fraction=float(trace["leg_saturation_fraction"][:steps,lane][alive].mean()),
        stable_fraction=float(trace["stable"][:steps,lane][alive].mean()),
        max_excursion_m=float(np.linalg.norm(position[alive]-position[first],axis=-1).max()),
        vx_mae_m_s=float(np.abs(trace["velocity"][:steps,lane,0][alive]-trace["commands"][:steps,lane,0][alive]).mean()),
        yaw_mae_rad_s=float(np.abs(trace["omega"][:steps,lane,2][alive]-trace["commands"][:steps,lane,1][alive]).mean()),
        final_second_stable_fraction=float(trace["stable"][:steps,lane][tail].mean()) if tail.any() else 0.)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before",type=Path,required=True)
    parser.add_argument("--after",type=Path,required=True)
    parser.add_argument("--change",choices=("handoff","target-limit","torque-delay","standing-targets"),required=True)
    parser.add_argument("--seconds",type=float,default=4.)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--before-checkpoint",type=Path)
    parser.add_argument("--after-checkpoint",type=Path)
    args=parser.parse_args()
    import numpy as np
    pb,rb,sb,tb=read(args.before)
    pa,ra,sa,ta=read(args.after)
    assert args.seconds <= min(pb["seconds"],pa["seconds"])
    for key in ("seed","severity","nominal_dynamics","actuator_model","actuator_timing_profile"):
        assert pb.get(key)==pa.get(key),key
    assert [r["name"] for r in rb["lanes"]]==[r["name"] for r in ra["lanes"]]
    assert np.array_equal(tb["observations"][0],ta["observations"][0])
    passive=tb["passive"][0]
    assert np.array_equal(passive,ta["passive"][0])
    for key in ("motor_position","motor_velocity"):
        assert np.array_equal(tb[key][0,passive],ta[key][0,passive]),key
    for b,a in zip(rb["lanes"],ra["lanes"],strict=True):
        assert b["material_coefficients"]==a["material_coefficients"]
        left,right=dict(b["first_reset_domain"]),dict(a["first_reset_domain"])
        if args.change=="torque-delay":
            left.pop("torque_delay_ms")
            assert right.pop("torque_delay_ms")==0
        assert left==right,b["name"]
    if args.change in ("torque-delay","standing-targets"):
        assert pb["checkpoint_sha256"]==pa["checkpoint_sha256"]
        assert not pb.get("evaluation_overrides",{})
        expected=({"torque_transport_delay_ms":0} if args.change=="torque-delay"
                  else {"hold_standing_leg_targets":"first_measured"})
        assert pa["evaluation_overrides"]==expected
    else:
        assert pb.get("evaluation_overrides",{})==pa.get("evaluation_overrides",{})=={}
        cb=json.loads((args.before/"task_contract.json").read_text())["settings"]
        ca=json.loads((args.after/"task_contract.json").read_text())["settings"]
        changed_setting = ("gap_entry_handoff_version" if args.change=="handoff"
                           else "gap_upright_target_limit_version")
        ca.pop(changed_setting,None)
        cb.pop(changed_setting,None)
        assert cb==ca
        import torch
        assert args.before_checkpoint is not None and args.after_checkpoint is not None
        for path,progress in ((args.before_checkpoint,pb),(args.after_checkpoint,pa)):
            assert digest(path)==progress["checkpoint_sha256"]
        wb,wa=[torch.load(p,map_location="cpu",weights_only=False) for p in (args.before_checkpoint,args.after_checkpoint)]
        assert not wb["infos"]["settings"].get(changed_setting)
        expected_version = "orbit-v2" if args.change=="handoff" else "error-v1"
        assert wa["infos"]["settings"][changed_setting]==expected_version
        assert wb["actor_state_dict"].keys()==wa["actor_state_dict"].keys()
        assert all(torch.equal(v,wa["actor_state_dict"][k]) for k,v in wb["actor_state_dict"].items())
    starts=[json.loads((p/"startup.json").read_text()) for p in (args.before,args.after)]
    assert starts[0]["dynamics_randomization"]["sampled_properties_sha256"]==starts[1]["dynamics_randomization"]["sampled_properties_sha256"]
    precontrol_files = [p/"evaluation_precontrol_state.npz" for p in (args.before,args.after)]
    # New diagnostics save actual root/q/dq before the first control action.
    # Historical comparisons explicitly retain their weaker pairing boundary.
    precontrol_exact = all(p.exists() for p in precontrol_files)
    if precontrol_exact:
        physical = [np.load(p) for p in precontrol_files]
        assert physical[0].files == physical[1].files
        for key in physical[0].files:
            assert np.array_equal(physical[0][key],physical[1][key]),key
    steps=round(args.seconds/.005)
    rows=[dict(name=b["name"],behavior_case=b["behavior_case"],
        before=metrics(tb,i,steps),after=metrics(ta,i,steps)) for i,b in enumerate(rb["lanes"])]
    from wheeled_algo.standing_acceptance import excursion_limit
    protocol=json.loads((ROOT/"reports/unified_robust_20261021/acceptance_protocol004.json").read_text())
    for row in rows:
        if row["behavior_case"].startswith("stand"):
            for side in ("before","after"):
                part=row[side]
                part["pose_and_position_pass"]=(part["alive_at_end"] and part.get("stable_fraction",0.)>=.9
                    and part.get("max_excursion_m",float("inf"))<=excursion_limit(protocol,row["behavior_case"]))
    groups={}
    for name,condition in dict(entry=lambda r:r["behavior_case"].startswith("real_start"),
        standing=lambda r:r["behavior_case"].startswith("stand"),
        fallen=lambda r:r["behavior_case"].startswith(("left_symmetric","right_symmetric","inverted_symmetric")),
        motion=lambda r:r["behavior_case"].startswith(("forward","backward","arc"))).items():
        selected=[r for r in rows if condition(r)]
        group={}
        for side in ("before","after"):
            parts=[r[side] for r in selected]
            stats=dict(episodes=len(parts),alive_at_end=sum(r["alive_at_end"] for r in parts),
                ever_recovered=sum(r["ever_recovered"] for r in parts),
                final_second_stable90=sum(r.get("final_second_stable_fraction",0.)>=.9 for r in parts),
                whole_window_stable90=sum(r.get("stable_fraction",0.)>=.9 for r in parts),
                pose_and_position_pass=sum(r.get("pose_and_position_pass",False) for r in parts))
            for key in ("max_leg_speed_rad_s","initial_200ms_effort_peak_nm","saturation_fraction",
                        "max_excursion_m","vx_mae_m_s","yaw_mae_rad_s"):
                values=[r[key] for r in parts if key in r]
                stats[key]=dict(mean=float(np.mean(values)),max=float(max(values))) if values else None
            group[side]=stats
        groups[name]=group
    out=dict(schema="real-gap-controlled-prefix-v1",change=args.change,seconds=args.seconds,
        before=str(args.before),after=str(args.after),paired_actor_inputs=True,
        passive_physical_state_exact=True,other_sensor_actuator_and_rigid_parameters_exact=True,
        active_precontrol_physical_state_exact=precontrol_exact,
        pairing_limit=(None if precontrol_exact else
            "Historical reference did not save the active physical state before its first control step"),
        groups=groups,rows=rows,optimizer_updates=0,qualification=None,full_training_admitted=False,hardware_ready=False)
    write_json(args.output,out)
    import torch
    from torch.utils.tensorboard import SummaryWriter
    checkpoint=Path(pa["checkpoint"])
    assert digest(checkpoint)==pa["checkpoint_sha256"]
    learning_clock=torch.load(checkpoint,map_location="cpu",weights_only=False)["infos"]["completed_updates"]
    writer=SummaryWriter(str(args.after/"evaluation/metrics"),filename_suffix=".controlled-prefix")
    for group,content in groups.items():
        for side,stats in content.items():
            for key,value in stats.items():
                fields=value.items() if isinstance(value,dict) else [(None,value)]
                for field,number in fields:
                    if isinstance(number,(int,float)):
                        tag=f"GapControlledPrefix/{args.change}/{args.seconds:g}s/{group}/{side}/{key}"
                        writer.add_scalar(tag+("/"+field if field else ""),number,learning_clock)
    writer.close()
    print(json.dumps(groups,ensure_ascii=False))


if __name__=="__main__":
    main()

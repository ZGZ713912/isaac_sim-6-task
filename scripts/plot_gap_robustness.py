#!/usr/bin/env python3
"""Static comparison from recorded TensorBoard scalars, not a second curve source."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before",type=Path,required=True)
    parser.add_argument("--after",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--before-label",default="Before")
    parser.add_argument("--after-label",default="After")
    args=parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulators=[]
    for path in (args.before,args.after):
        acc=EventAccumulator(str(path/"evaluation/metrics"),size_guidance={"scalars":0})
        acc.Reload()
        accumulators.append(acc)
    summary=json.loads((args.after/"evaluation/gap_summary.json").read_text())
    entry=[r for r in summary["rows"] if r["behavior_case"].startswith("real_start")]
    standing=[r for r in summary["rows"] if r["behavior_case"].startswith("stand")]
    fig,axes=plt.subplots(2,2,figsize=(15,12),layout="constrained")
    configurations=[(entry,"first_200ms_leg_effort_peak_nm",1.,"Entry: first200ms leg effort peak [Nm]"),
        (entry,"max_leg_speed_rad_s",1.,"Entry: maximum physical leg speed [rad/s]"),
        (standing,"max_position_excursion_m",100.,"Standing: maximum excursion [cm]"),
        (standing,"tail_leg_motion_3_95hz_rms_rad",1000.,"Standing: final4s leg motion3-95Hz RMS [mrad]")]
    for ax,(rows,key,scale,title) in zip(axes.flat,configurations):
        positions=np.arange(len(rows))
        for acc,shift,label,color in zip(accumulators,(-.18,.18),(args.before_label,args.after_label),("#949ba4","#1d8bba")):
            available=set(acc.Tags()["scalars"])
            values=[]
            for row in rows:
                tag="GapEvaluation/"+row["name"]+"/"+key
                values.append(acc.Scalars(tag)[-1].value*scale if tag in available else float("nan"))
            ax.barh(positions+shift,values,height=.34,label=label,color=color)
        labels=[r["name"].replace("real_start_","start").replace("stand_high","high").replace("stand_low","low").replace("stand__","mid__").replace("__"," / ") for r in rows]
        ax.set_yticks(positions,labels,fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x",alpha=.2)
        if key in ("max_position_excursion_m","tail_leg_motion_3_95hz_rms_rad"):
            ax.set_xscale("symlog",linthresh=5. if key=="max_position_excursion_m" else 1.)
        ax.legend(fontsize=8)
    fig.suptitle("Matched IMU and dynamics stress test; finite simulation only",fontsize=14)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare equal-time trajectories separately from the longer observation window."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/"tools")]
from wheeled_tasks.chassis.fallen_pose_bank import digest


def histories(path,ids):
    bank = json.loads(path.read_text())
    if bank["status"] != "completed":
        raise ValueError("Wait for completed physics")
    result = {}
    for chunk in bank["trace_files"]:
        wanted = [j for j in range(chunk["count"]) if bank["starts"][chunk["first_record"]+j]["pair_id"] in ids]
        if not wanted:
            continue
        source = path.parent/chunk["path"]
        if digest(source) != chunk["sha256"]:
            raise ValueError("Trace changed")
        trace = np.load(source)["trace"]
        for j in wanted:
            i = chunk["first_record"]+j
            s = bank["starts"][i]
            a = trace[:,j]
            gravity = Rotation.from_quat(a[:,3:7]).inv().apply(np.tile([0.,0.,-1.],(len(a),1)))
            result[(s["pair_id"],s["leg_friction_enabled"])] = dict(gravity=gravity,trace=a,
                columns=bank["trace_columns"],record=bank["records"][i],start=s,
                time_s=(np.arange(len(a))+1)*.01)
    return bank,result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coarse",type=Path,required=True)
    p.add_argument("--fine",type=Path,nargs="+",required=True)
    p.add_argument("--half-ms",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    fine = {}
    for path in args.fine:
        b = json.loads(path.read_text())
        _,h = histories(path,{s["pair_id"] for s in b["starts"]})
        if set(fine)&set(h):
            raise ValueError("Duplicate fine cases")
        fine.update(h)
    ids = {key[0] for key in fine}
    coarse_bank,coarse = histories(args.coarse,ids)
    half_bank,half = histories(args.half_ms,ids)
    def degrees(a,b):
        return float(np.rad2deg(np.arccos(np.clip(a@b,-1.,1.))))
    rows = []
    for case in sorted(ids):
        a,b = fine[(case,False)],fine[(case,True)]
        row = dict(pair_id=case,side=a["start"]["side"],roll_magnitude_deg=a["start"]["roll_magnitude_deg"],
            knee_pair_deg=a["start"]["knee_pair_deg"],orbital_pair_deg=a["start"]["orbital_pair_deg"],
            coarse_pair_difference_12s_deg=degrees(coarse[(case,False)]["gravity"][1199],coarse[(case,True)]["gravity"][1199]),
            fine_pair_difference_12s_deg=degrees(a["gravity"][1199],b["gravity"][1199]),
            fine_pair_difference_24s_deg=degrees(a["gravity"][-1],b["gravity"][-1]))
        for label,h in (("off",a),("moving",b)):
            flat = h["gravity"][:,2] < -math.cos(math.radians(15.))
            last_nonflat = np.flatnonzero(~flat)
            first = int(last_nonflat[-1]+1) if len(last_nonflat) else 0
            row[label+"_fine_time_entered_15deg_belly_until_end_s"] = float(h["time_s"][first]) if first < len(flat)-100 else None
            row[label+"_coarse_vs_fine_at_12s_deg"] = degrees(coarse[(case,label == "moving")]["gravity"][1199],h["gravity"][1199])
        if (case,False) in half:
            c,d = half[(case,False)],half[(case,True)]
            row["half_ms_pair_difference_12s_deg"] = degrees(c["gravity"][-1],d["gravity"][-1])
            row["half_ms_identical_world_origin"] = c["record"]["physics_origin_world_m"] == d["record"]["physics_origin_world_m"]
            row["moving_fine_vs_half_ms_at_12s_deg"] = degrees(b["gravity"][1199],d["gravity"][-1])
        rows.append(row)
    output = dict(schema="side-fall-equal-time-step-comparison-v1",sample_dt_s=.01,
        source_banks={str(path.resolve()):digest(path) for path in [args.coarse,*args.fine,args.half_ms]},
        script_sha256=digest(__file__),rows=rows,
        scope="12s comparisons isolate observation horizon. Half-ms controls reuse exactly the same physical origin for both arms;individual paths can remain contact-sensitive.")
    (args.output/"comparison.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
    spec = json.loads((ROOT/coarse_bank["asset_directory"]/"model_spec.json").read_text())
    zero = spec["passive_knee_zero_inner_rad"]
    cases = [61,237]
    fig,axes = plt.subplots(2,2,figsize=(12.,8.),sharex=True)
    for col,case in enumerate(cases):
        joint = "R_jonit2" if case == 61 else "L_joint2"
        curves = [(fine[(case,False)],"无腿摩擦 · 1 ms","#637484","-"),
            (coarse[(case,True)],"新动摩擦 · 5 ms","#d48521","--"),
            (fine[(case,True)],"新动摩擦 · 1 ms","#2476b9","-"),
            (half[(case,True)],"新动摩擦 · 0.5 ms，同坐标对照","#15976d",":")]
        for h,label,color,style in curves:
            axes[0,col].plot(h["time_s"],np.rad2deg(np.arccos(np.clip(-h["gravity"][:,2],-1.,1.))),label=label,color=color,ls=style,lw=1.5)
            q = h["trace"][:,h["columns"].index("q/"+joint)]
            knee = np.rad2deg(zero[joint]-q if joint.startswith("R") else q+zero[joint])
            axes[1,col].plot(h["time_s"],knee,color=color,ls=style,lw=1.5)
        start = fine[(case,False)]["start"]
        axes[0,col].set_title(f"#{case} · {'左' if start['side'] == 'left' else '右'}侧起点 {start['roll_magnitude_deg']:g}° · 腿夹角 75/75°",fontproperties=font)
        axes[0,col].set_ylim(-3,95);axes[1,col].set_ylim(72,112)
        axes[1,col].set_xlabel("时间 (s)",fontproperties=font)
        axes[1,col].set_ylabel(("右" if joint.startswith("R") else "左")+"腿实际夹角 (°)",fontproperties=font)
        for ax in axes[:,col]:
            ax.set_xlim(0,24);ax.grid(alpha=.2)
    axes[0,0].set_ylabel("车身偏离正立的角度 (°)",fontproperties=font)
    axes[0,0].legend(prop=font,fontsize=8,loc="upper right")
    fig.suptitle("支撑停留与腿角变化：粗步长不能替代细步长复核",fontproperties=font,fontsize=17)
    fig.tight_layout(rect=[0,.04,1,.94])
    fig.text(.5,.01,"0.5 ms 对照观察至 12 s，1 ms 至 24 s。右例原滚落途中有 CAD 相交，终态另行轻放复验见报告。",ha="center",fontproperties=font,fontsize=10)
    fig.savefig(args.output/"support_time_comparison.png",dpi=160)
    print(json.dumps(output,ensure_ascii=False))


if __name__ == "__main__":
    main()

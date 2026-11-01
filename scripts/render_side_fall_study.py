#!/usr/bin/env python3
"""Static scientific figures from the sealed paired native study."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/"tools")]
from summarize_fallen_pose_bank import meshes, LABELS
from build_full_vehicle_training import geometry
from wheeled_tasks.chassis.fallen_pose_bank import digest
from wheeled_tasks.chassis.side_fall_study import KNEE_PAIRS

FONT = FontProperties(fname="/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gallery-pairs",type=int,nargs="+",default=[])
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    bank = json.loads(Path(comparison["source_bank"]).read_text())
    if digest(comparison["source_bank"]) != comparison["source_bank_sha256"]:
        raise ValueError("Native bank changed")
    args.output.mkdir(parents=True,exist_ok=False)
    pairs = comparison["pairs"]
    inputs = {tuple(p["knee_pair_deg"]) for p in pairs}
    knees = list(KNEE_PAIRS) if inputs <= set(KNEE_PAIRS) else sorted(inputs)
    rolls = sorted({p["roll_magnitude_deg"] for p in pairs})
    orbits = sorted({tuple(p["orbital_pair_deg"]) for p in pairs})
    fig, axes = plt.subplots(2,len(rolls),figsize=(3.5*len(rolls),7.),squeeze=False)
    for row,side in enumerate(("left","right")):
        for col,roll in enumerate(rolls):
            ax = axes[row,col]
            ax.set_facecolor("#d4d8dd")
            values = np.full((len(knees),len(orbits)),np.nan)
            subset = [p for p in pairs if p["side"] == side and p["roll_magnitude_deg"] == roll]
            for p in subset:
                y = knees.index(tuple(p["knee_pair_deg"]))
                x = orbits.index(tuple(p["orbital_pair_deg"]))
                if p["both_screened"]:
                    values[y,x] = p["body_angle_difference_deg"]
                else:
                    ax.text(x,y,"×",ha="center",va="center",fontsize=9,color="#7b2434")
                if p["both_screened"] and p["moving"]["sustained_leg_support"]:
                    ax.plot(x,y,".",color="black",ms=4)
                if p["both_screened"] and p["both_direction_stable"]:
                    ax.plot(x,y,"o",mfc="none",mec="#405777",ms=8,mew=.7)
            im = ax.imshow(values,cmap="YlOrRd",vmin=0,vmax=180,aspect="auto",interpolation="nearest")
            ax.set_xticks(range(len(orbits)),[f"{l:g}/{r:g}" for l,r in orbits],rotation=90,fontsize=7)
            ax.set_yticks(range(len(knees)),[f"{l:.1f}/{r:.1f}" for l,r in knees],fontsize=8)
            ax.set_title(f"{'左' if side == 'left' else '右'}侧起点 {roll:g}°",fontproperties=FONT,fontsize=11)
            if col == 0:
                ax.set_ylabel("初始腿夹角 L/R (°)",fontproperties=FONT,fontsize=10)
    fig.subplots_adjust(left=.055,right=.98,top=.86,bottom=.21,hspace=.55,wspace=.32)
    fig.suptitle(f"新腿动摩擦对侧躺终态的影响 · {comparison['duration_s']:g} s / {1000*comparison['dt_s']:g} ms",fontproperties=FONT,fontsize=17,y=.97)
    fig.text(.5,.90,"同一起点比较：颜色为两终态的车身重力方向夹角；摆角 0° 为轮心在髋下方，正值朝车头",ha="center",fontproperties=FONT,fontsize=11)
    color_ax = fig.add_axes([.33,.085,.34,.025])
    cb = fig.colorbar(im,cax=color_ax,orientation="horizontal",ticks=[0,45,90,135,180])
    cb.set_label("Body gravity-direction difference (deg)",fontsize=9)
    fig.text(.5,.025,"横轴：初始整腿摆角 L/R (°)    灰：未采纳 / 未覆盖    ×：终态机械或 CAD 筛查未过\n黑点：加摩擦后腿/轮持续承担至少 5% 重量    空心圈：两组车身方向均通过末 2 s 窗口；不代表整车完全静止",ha="center",fontproperties=FONT,fontsize=9)
    fig.savefig(args.output/"side_fall_effect_map.png",dpi=160)
    plt.close(fig)
    selected = [p for p in pairs if p["pair_id"] in args.gallery_pairs]
    if not selected:
        return
    bundle = ROOT/bank["asset_directory"]
    spec = json.loads((bundle/"model_spec.json").read_text())
    parts = meshes(bundle,spec,visual=True)
    fig = plt.figure(figsize=(15.,4.5*len(selected)+1.3),facecolor="#f7f8fa")
    for row,pair in enumerate(selected):
        start = bank["starts"][pair["off"]["record_index"]]
        root = np.eye(4)
        root[:3,:3] = Rotation.from_quat(start["quaternion_xyzw"]).as_matrix()
        root[:3,3] = start["root_position_m"]
        poses = {n:list(t[:3,3])+Rotation.from_matrix(t[:3,:3]).as_quat().tolist()
            for n,t in geometry.fk(spec,start["full_joint_positions"],root).items()}
        panels = [(poses,start["quaternion_xyzw"],start["root_position_m"][2],"相同初态",None)]
        for arm,label in (("off","无腿电机摩擦"),("moving","新腿电机动摩擦")):
            m = pair[arm]
            pose = bank["records"][m["record_index"]]["final_pose"]
            state = "整车低速窗口通过" if m["full_window"] else "车身及腿低速窗口通过" if m["chassis_window"] else "车身方向稳定，部件仍动" if m["direction_stable"] else "仍在变化"
            tilt = math.degrees(math.acos(float(np.clip(-m["gravity_body"][2],-1.,1.))))
            mode = "轮腿斜撑" if m["sustained_leg_support"] and 15. < tilt < 75. else LABELS[m["posture_class"]]
            title = label+"\n"+mode+f" ({tilt:.1f}°) · "+state
            panels.append((pose["native_body_poses_world"],pose["root_quaternion_xyzw"],pose["root_position_m"][2],title,m))
        for col,(poses,quat,height,title,m) in enumerate(panels):
            ax = fig.add_subplot(len(selected),3,row*3+col+1,projection="3d",computed_zorder=False)
            ax.set_facecolor("#f7f8fa")
            rotation = Rotation.from_quat(quat)
            heading = rotation.apply([1.,0.,0.])
            yaw = math.atan2(heading[1],heading[0]) if np.linalg.norm(heading[:2]) > .15 else 0.
            norm = Rotation.from_euler("z",-yaw).as_matrix()
            for name,vertices,faces,local in parts:
                pose = poses[name]
                world = np.eye(4);world[:3,:3] = Rotation.from_quat(pose[3:]).as_matrix();world[:3,3] = pose[:3]
                t = world@local
                v = (vertices@t[:3,:3].T+t[:3,3])@norm.T
                color = "#27323d" if name in ("L_link3","R_link3") else "#eea235" if name.startswith("M_") else "#8b99a6" if name == "base_link" else "#437fc1" if name.startswith("L") else "#d3655b"
                ax.add_collection3d(Poly3DCollection(v[faces],facecolors=color,linewidth=0.,shade=True,zorder=3))
            square = [[-.7,-.7,0],[.7,-.7,0],[.7,.7,0],[-.7,.7,0]]
            ax.add_collection3d(Poly3DCollection([square],facecolors="#dce2e8",alpha=.5,zorder=1))
            direction = norm@rotation.apply([.23,0,0])
            ax.quiver(0,0,height,*direction,color="#1a976d",linewidth=2.,arrow_length_ratio=.18,zorder=5)
            ax.set(xlim=(-.65,.65),ylim=(-.65,.65),zlim=(-.03,.78))
            ax.set_box_aspect((1.,1.,.63),zoom=1.12)
            ax.set_proj_type("ortho");ax.view_init(elev=20,azim=-52);ax.set_axis_off()
            ax.set_title(title,fontproperties=FONT,fontsize=11,pad=2)
            if m:
                text = f"腿/轮承重 {100*m['leg_and_wheel_support']['mean_weight_fraction']:.1f}%  腿夹角 {m['knee_pair_deg'][0]:.1f}/{m['knee_pair_deg'][1]:.1f}°"
            else:
                k,o = start["knee_pair_deg"],start["orbital_pair_deg"]
                text = f"#{pair['pair_id']} · {start['side']} · roll {start['roll_magnitude_deg']:.1f}°\n腿夹角 {k[0]:.1f}/{k[1]:.1f}°  摆角 {o[0]:.1f}/{o[1]:.1f}°"
            ax.text2D(.05,.02,text,transform=ax.transAxes,fontproperties=FONT,fontsize=9,color="#344454")
    fig.suptitle(f"侧躺配对试验 · CAD 外形与实际终态 · {comparison['duration_s']:g} s / {1000*comparison['dt_s']:g} ms",fontproperties=FONT,fontsize=17,y=.98)
    fig.subplots_adjust(left=.01,right=.99,top=.80 if len(selected)==1 else .90,
        bottom=.11 if len(selected)==1 else .065,hspace=.15,wspace=.01)
    fig.text(.5,.015,"蓝/红：左右腿   深色：轮   灰：底盘外壳   橙：自由云台   绿：车头方向\n八电机失能，非轮摩擦 0.7；新模型是动摩擦先验，未辨识静摩擦。",ha="center",fontproperties=FONT,fontsize=10)
    fig.savefig(args.output/"side_fall_paired_cad.png",dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()

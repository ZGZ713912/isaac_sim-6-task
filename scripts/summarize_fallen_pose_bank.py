#!/usr/bin/env python3
"""Inspect measured passive endpoints; export medoids and a collision-pose gallery."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from wheeled_tasks.chassis.fallen_pose_bank import WINDOW, digest, posture_class

LABELS = dict(belly_down="正向趴地", wheel_supported="轮支撑", front_down="车头向下",
    rear_down="车尾向下", left_side="左侧着地", right_side="右侧着地", roof_down="翻顶", diagonal="斜向支撑")
STATE = dict(full="整车低速收敛", chassis="车身与腿落稳，云台仍动", direction="车身方向稳定，仍有部件运动")


def classify(record):
    """Never turn a mechanical rejection or a quiet base into a fully static robot."""
    if record["status"] not in ("settled", "not_settled"):
        return None
    p = record["final_pose"]
    if p.get("cad_contact_audit", {}).get("review_required"):
        return None
    if record["status"] == "settled" or p.get("window_rest", {}).get("passed"):
        return "full"
    if p.get("chassis_window_rest", {}).get("passed"):
        return "chassis"
    w = p.get("window_rest", {})
    if (w.get("seconds", 0.) >= WINDOW["seconds"]
            and w.get("root_translation_range_m", math.inf) <= WINDOW["root_translation_range_m"]
            and w.get("gravity_range_deg", math.inf) <= WINDOW["gravity_range_deg"]
            and .8 <= w.get("mean_support_weight_fraction", 0.) <= 1.2):
        return "direction"
    return None


def group_modes(records, angle_deg=12., height_m=.035):
    """Complete-link gravity clusters with bounded height; joints stay in raw data.

    The number of modes comes from measured configurations. Direction names are
    attached afterwards; there is no requested/preset number of modes.
    """
    groups = []
    # Prefer fully resting real endpoints as the first cluster representatives.
    rank = dict(full=0, chassis=1, direction=2)
    admitted = [(r, classify(r)) for r in records if classify(r)]
    admitted.sort(key=lambda item: (rank[item[1]], item[0]["index"]))
    cos = math.cos(math.radians(angle_deg))
    for record, state in admitted:
        p = record["final_pose"]
        gravity, height = np.array(p["gravity_body"]), p["root_position_m"][2]
        wheel_only = posture_class(p) == "wheel_supported"
        for group in groups:
            values = [records[i]["final_pose"] for i in group["record_indices"]]
            if (wheel_only == group["wheel_only"]
                    and all(gravity@np.array(v["gravity_body"]) >= cos for v in values)
                    and max([height]+[v["root_position_m"][2] for v in values])
                        -min([height]+[v["root_position_m"][2] for v in values]) <= height_m):
                group["record_indices"].append(record["index"])
                group["state_counts"][state] = group["state_counts"].get(state, 0)+1
                break
        else:
            groups.append(dict(record_indices=[record["index"]], representative_record=record["index"],
                wheel_only=wheel_only, state_counts={state:1}))
    for group in groups:
        # Medoid among the best evidence level, measured endpoint only.
        available = [i for i in group["record_indices"] if classify(records[i]) == min(group["state_counts"], key=rank.get)]
        gravities = np.array([records[i]["final_pose"]["gravity_body"] for i in group["record_indices"]])
        selected = min(available, key=lambda i: np.arccos(np.clip(gravities@records[i]["final_pose"]["gravity_body"], -1., 1.)).sum())
        p = records[selected]["final_pose"]
        group.update(representative_record=selected, posture_class=posture_class(p), evidence=classify(records[selected]),
            representative_pose=p, support_combinations=dict(Counter("+".join(records[i]["final_pose"]["support_bodies"])
                for i in group["record_indices"])))
    return sorted(groups, key=lambda g: (-len(g["record_indices"]), rank[g["evidence"]], g["representative_record"]))


def direction_overview(records):
    result = []
    rank = dict(full=0,chassis=1,direction=2)
    for name in ("belly_down","front_down","rear_down","left_side","right_side","roof_down","diagonal","wheel_supported"):
        members = [r for r in records if classify(r) and posture_class(r["final_pose"]) == name]
        if not members:
            continue
        level = min((classify(r) for r in members),key=rank.get)
        gravity = np.array([r["final_pose"]["gravity_body"] for r in members])
        chosen = min((r for r in members if classify(r) == level),key=lambda r:
            np.arccos(np.clip(gravity@r["final_pose"]["gravity_body"],-1.,1.)).sum())
        result.append(dict(posture_class=name,evidence=level,record_indices=[r["index"] for r in members],
            representative_record=chosen["index"],representative_pose=chosen["final_pose"],
            state_counts=dict(Counter(classify(r) for r in members))))
    return result


def meshes(bundle, spec, *, visual=False):
    import trimesh
    result = []
    for b in spec["bodies"]:
        if visual:
            path = bundle/b["mesh"]
            if not path.is_file():
                path = bundle.parent/"isaac"/b["mesh"]
            mesh = trimesh.load(path,force="mesh",process=True)
            if len(mesh.faces) > 5000:
                mesh = mesh.simplify_quadric_decimation(face_count=5000)
            result.append((b["name"],np.array(mesh.vertices),np.array(mesh.faces),np.array(b["mesh_origin"])))
            continue
        for c in b["collisions"]:
            mesh = (trimesh.load(bundle/c["file"], force="mesh", process=False) if c["type"] == "mesh"
                else trimesh.creation.cylinder(radius=c["radius"], height=c["length"], sections=64))
            result.append((b["name"], np.array(mesh.vertices), np.array(mesh.faces), np.array(c["origin"])))
    return result


def gallery(groups, bundle, spec, output, *, limit=12, visual=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    font = FontProperties(fname="/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
    parts = meshes(bundle, spec,visual=visual)
    selected = groups[:limit]
    if not selected:
        return None
    cols = min(3, len(selected))
    rows = math.ceil(len(selected)/cols)
    fig = plt.figure(figsize=(5.1*cols, 4.5*rows+.9), facecolor="#f7f8fa")
    top, bottom = .94,.075
    cell_h, cell_w = (top-bottom)/rows,1./cols
    for k, group in enumerate(selected):
        p = group["representative_pose"]
        row,col = divmod(k,cols)
        cell_top = top-row*cell_h
        ax = fig.add_axes([col*cell_w+.01,cell_top-.86*cell_h,cell_w-.02,.69*cell_h],
            projection="3d",computed_zorder=False)
        ax.set_facecolor("#f7f8fa")
        root_rot = Rotation.from_quat(p["root_quaternion_xyzw"])
        direction = root_rot.apply([1., 0., 0.])
        fallback = np.linalg.norm(direction[:2]) < .15
        if fallback:
            direction = root_rot.apply([0., 1., 0.])
        yaw = math.atan2(direction[1], direction[0])-(math.pi/2 if fallback else 0.)
        normalize = Rotation.from_euler("z", -yaw).as_matrix()
        for name, vertices, faces, local in parts:
            native = p["native_body_poses_world"][name]
            world = np.eye(4)
            world[:3, :3] = Rotation.from_quat(native[3:]).as_matrix()
            world[:3, 3] = native[:3]
            transform = world@local
            transformed = (vertices@transform[:3, :3].T+transform[:3, 3])@normalize.T
            color = ("#27323d" if name in ("L_link3", "R_link3") else "#eea235" if name.startswith("M_")
                else "#8b99a6" if name == "base_link" else "#437fc1" if name.startswith("L") else "#d3655b")
            edges = {} if visual else dict(edgecolors="#26313d")
            collection = Poly3DCollection(transformed[faces], facecolors=color, **edges,
                linewidth=.12, alpha=1. if visual else .95, zorder=3,shade=visual)
            ax.add_collection3d(collection)
        square = np.array([[-.62,-.62,0],[.62,-.62,0],[.62,.62,0],[-.62,.62,0]])
        ax.add_collection3d(Poly3DCollection([square], facecolor="#dde3e8", alpha=.45, zorder=1))
        for u in np.linspace(-.6, .6, 7):
            ax.plot([u,u],[-.6,.6],[0,0],color="#c9d0d6",lw=.4,zorder=2)
            ax.plot([-.6,.6],[u,u],[0,0],color="#c9d0d6",lw=.4,zorder=2)
        x = normalize@root_rot.apply([.22,0,0])
        z = p["root_position_m"][2]
        ax.quiver(0,0,z,*x,color="#21a179",linewidth=2.,arrow_length_ratio=.18,zorder=5)
        ax.text(*np.array([0,0,z])+x, "车头",fontproperties=font,fontsize=8,color="#128061",zorder=6)
        ax.set(xlim=(-.5,.5),ylim=(-.5,.5),zlim=(-.03,.60))
        ax.set_box_aspect((1.,1.,.63),zoom=1.13)
        ax.set_proj_type("ortho")
        ax.view_init(elev=19, azim=-52)
        ax.set_axis_off()
        counts = group["state_counts"]
        levels = " / ".join(f"{dict(full='整车',chassis='车身及腿',direction='方向')[s]} {n}" for s,n in counts.items())
        title = f"{k+1}. {LABELS[group['posture_class']]} · {len(group['record_indices'])} 次\n{levels}\n5 ms：{STATE[group['evidence']]}"
        if group.get("refinement"):
            title += "\n1 ms："+group["refinement"]["display"]
        fig.text((col+.5)*cell_w,cell_top,title,ha="center",va="top",fontproperties=font,fontsize=12,color="#192734")
        fig.text(col*cell_w+.025,cell_top-.90*cell_h,f"记录 {group['representative_record']}   车身高度 {z*1000:.0f} mm\n"
            f"支撑：{', '.join(p['support_bodies'])}",va="top",fontproperties=font,fontsize=7.5,color="#415466")
    fig.suptitle("失能倒地候选 · 实际刚体位姿与"+("CAD 外形" if visual else "碰撞包络"),fontproperties=font,fontsize=17,y=.995)
    fig.text(.5,.01,"灰：底盘/外壳   橙：自由云台   蓝/红：左右腿   深色：驱动轮   绿箭头：车头方向\n"
        "八个电机失能，非轮摩擦 0.7；低速窗口通过不证明长期稳定，1 ms 复放的运动与自交另列。",ha="center",fontproperties=font,fontsize=10,color="#415466")
    path = output/("posture_gallery_cad.png" if visual else "posture_gallery.png")
    fig.savefig(path,dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--partial",action="store_true",help="Explicit preview of a still-running bank")
    parser.add_argument("--visual-meshes",action="store_true",help="Also render CAD meshes;requires fast_simplification")
    parser.add_argument("--contact-audit",type=Path,help="Exclude endpoints with new CAD contacts or CAD-ground penetration")
    parser.add_argument("--replay-comparison",type=Path,help="Attach measured1ms refinement results without overwriting source evidence")
    args = parser.parse_args()
    bank = json.loads(args.bank.read_text())
    if bank["status"] != "completed" and not args.partial:
        raise ValueError("Need a completed bank;use--partial only for a labelled preview")
    args.output.mkdir(parents=True,exist_ok=False)
    records = bank["records"]
    if args.contact_audit:
        audit = json.loads(args.contact_audit.read_text())
        if audit["source_bank_sha256"] != digest(args.bank):
            raise ValueError("Contact audit does not match this bank snapshot")
        by_index = {r["index"]:r for r in audit["records"]}
        for r in records:
            r["final_pose"]["cad_contact_audit"] = by_index[r["index"]]
    groups = group_modes(records)
    overview = direction_overview(records)
    replay_comparison = None
    if args.replay_comparison:
        replay_comparison = json.loads(args.replay_comparison.read_text())
        if replay_comparison["source_bank_sha256"] != digest(args.bank):
            raise ValueError("Replay comparison does not match this bank")
        by_source = {r["source_record"]:r for r in replay_comparison["records"]}
        for group in overview:
            if group["representative_record"] in by_source:
                group["refinement"] = by_source[group["representative_record"]]
    summary = dict(schema="passive-fallen-pose-summary-v1",source_bank=str(args.bank.resolve()),
        source_bank_sha256=digest(args.bank),source_status=bank["status"],partial=args.partial,
        created_utc=datetime.now(timezone.utc).isoformat(),source_script_sha256=digest(__file__),
        requested_placements=bank["requested_cases"],completed_placements=bank["completed_cases"],
        outcome_counts=dict(Counter(r["status"] for r in records)),
        evidence_counts=dict(Counter(classify(r) or "not_admitted" for r in records)),
        direction_counts=dict(Counter(posture_class(r["final_pose"]) for r in records if classify(r))),
        cluster_contract=dict(maximum_pairwise_gravity_angle_deg=12.,maximum_height_range_m=.035,
            ignores="whole-vehicle yaw,XY and within-mode leg/gimbal variation;raw configurations are retained"),
        modes=groups,direction_overview=overview,exhaustive=False,recovery_or_hardware_qualified=False)
    prefixes = sorted({min(n,len(records)) for n in (64,128,182,256,320,384,438,len(records)) if n > 0})
    summary["coverage_prefixes"] = [dict(placements=n,
        direction_classes=sorted({posture_class(r["final_pose"]) for r in records[:n] if classify(r)}),
        direction_class_count=len({posture_class(r["final_pose"]) for r in records[:n] if classify(r)}),
        fine_mode_count=len(group_modes(records[:n])),evidence_counts=dict(Counter(classify(r) or "not_admitted" for r in records[:n])))
        for n in prefixes]
    summary["coarse_vs_fine"] = "Seven descriptive gravity-direction buckets are not seven exact joint configurations;fine clusters depend on declared tolerances. Finite coverage is not an exhaustive proof."
    if args.contact_audit:
        summary.update(contact_audit=str(args.contact_audit.resolve()),contact_audit_sha256=digest(args.contact_audit))
    if replay_comparison:
        summary.update(replay_comparison=replay_comparison,replay_comparison_sha256=digest(args.replay_comparison),
            full_stability_qualified=False)
    spec = json.loads((ROOT/bank["asset_directory"]/"model_spec.json").read_text())
    image = gallery(overview,ROOT/bank["asset_directory"],spec,args.output)
    if args.visual_meshes:
        summary["cad_gallery"] = str(gallery(overview,ROOT/bank["asset_directory"],spec,args.output,visual=True))
    summary["gallery"] = str(image) if image else None
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("modes","direction_overview")},ensure_ascii=False))


if __name__ == "__main__":
    main()

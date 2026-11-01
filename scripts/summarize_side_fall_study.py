#!/usr/bin/env python3
"""Compare measured paired falls and distinguish ground bracing from retained leg shape."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/"tools")]
from wheeled_tasks.chassis.fallen_pose_bank import digest, native_clearance
from wheeled_tasks.chassis.side_fall_study import placement_fingerprint, orbital_angles


def angle(a,b):
    return float(np.rad2deg(np.arccos(np.clip(np.asarray(a)@np.asarray(b),-1.,1.))))


def trace_metrics(trace,columns,body_names,weight,side):
    last = trace[-200:]
    def contact(names):
        force = np.stack([last[:,columns.index("contact_z_n/"+n)] for n in names],axis=-1)
        total = np.maximum(force,0.).sum(-1)
        return dict(mean_weight_fraction=float(total.mean()/weight),fraction_over_2n=float((total>2.).mean()))
    legs = [n for n in body_names if n not in ("base_link","M_link1","M_link2")]
    wheels = ["L_link3","R_link3"]
    structures = [n for n in legs if n not in wheels]
    near = [n for n in legs if n.startswith("L" if side == "left" else "R")]
    result = dict(leg_and_wheel_support=contact(legs),wheel_support=contact(wheels),
        leg_structure_support=contact(structures),near_side_leg_support=contact(near))
    result["sustained_leg_support"] = (result["leg_and_wheel_support"]["mean_weight_fraction"] >= .05
        and result["leg_and_wheel_support"]["fraction_over_2n"] >= .8)
    result["active_leg_speed_rms_rad_s"] = float(np.sqrt(np.mean(np.stack([
        last[:,columns.index("dq/"+n)] for n in ("L_joint1","LL_joint1","R_joint1","RR_joint1")])**2)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--cad",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--partial",action="store_true",help="Explicit provisional analysis of completed batches only")
    args = parser.parse_args()
    bank = json.loads(args.bank.read_text())
    cad = json.loads(args.cad.read_text())
    if (bank["status"] != "completed" and not args.partial) or cad["source_bank_sha256"] != digest(args.bank):
        raise ValueError("Need a completed bank and its exact endpoint CAD audit")
    if not bank.get("fixed_duration") or bank["motor_effort_peak_nm"] > 1e-6:
        raise ValueError("Need a fixed-duration passive paired sweep")
    args.output.mkdir(parents=True,exist_ok=False)
    audits = {r["index"]:r for r in cad["records"]}
    bundle = ROOT/bank["asset_directory"]
    spec = json.loads((bundle/"model_spec.json").read_text())
    from build_full_vehicle_training import ResetGeometry
    geometry = ResetGeometry(spec,bundle)
    metrics = {}
    for chunk in bank["trace_files"]:
        path = args.bank.parent/chunk["path"]
        if digest(path) != chunk["sha256"]:
            raise ValueError("Trace changed")
        trace = np.load(path)["trace"]
        for j in range(chunk["count"]):
            index = chunk["first_record"]+j
            start = bank["starts"][index]
            record = bank["records"][index]
            pose = record["final_pose"]
            m = trace_metrics(trace[:,j],bank["trace_columns"],bank["native"]["body_names"],
                sum(bank["native"]["masses_kg"][0])*9.81,start["side"])
            m.update(record_index=index,status=record["status"],cad_review_required=audits[index]["review_required"],
                posture_class=pose["posture_class"],gravity_body=pose["gravity_body"],
                knee_pair_deg=pose["knee_deg"],orbital_pair_deg=orbital_angles(spec,pose["full_joint_positions"]),
                full_window=pose["window_rest"]["passed"],chassis_window=pose["chassis_window_rest"]["passed"],
                gravity_range_deg=pose["window_rest"].get("gravity_range_deg"),
                translation_range_m=pose["window_rest"].get("root_translation_range_m"),
                leg_dissipation_j=record["leg_loss_dissipation_j"])
            m["mechanically_screened"] = record["status"] in ("settled","not_settled") and not m["cad_review_required"]
            w = pose["window_rest"]
            m["direction_stable"] = (w.get("gravity_range_deg",math.inf) <= 1.
                and w.get("root_translation_range_m",math.inf) <= .005 and .8 <= w.get("mean_support_weight_fraction",0.) <= 1.2)
            clearances = {}
            for body in spec["bodies"]:
                if body["name"] not in ("base_link","M_link1","M_link2"):
                    n = body["name"]
                    clearances[n] = native_clearance(dict(bodies=[body]),geometry.meshes,[n],[pose["native_body_poses_world"][n]])
            m["minimum_leg_wheel_clearance_m"] = min(clearances.values())
            metrics[index] = m
    by_pair = {}
    for i,s in enumerate(bank["starts"]):
        if placement_fingerprint(s) != s["placement_sha256"]:
            raise ValueError("Initial state fingerprint mismatch")
        by_pair.setdefault(s["pair_id"],{})[s["leg_friction_enabled"]] = i
    pairs = []
    for pair_id, arms in by_pair.items():
        off, moving = arms[False],arms[True]
        s = bank["starts"][off]
        if s["placement_sha256"] != bank["starts"][moving]["placement_sha256"]:
            raise ValueError("Unmatched placement")
        a,b = metrics[off],metrics[moving]
        pairs.append(dict(pair_id=pair_id,side=s["side"],roll_magnitude_deg=s["roll_magnitude_deg"],
            knee_pair_deg=s["knee_pair_deg"],orbital_pair_deg=s["orbital_pair_deg"],
            initial_gimbal_yaw_rad=s["gimbal_yaw_rad"],
            initial_leg_clearance_m=min(v for n,v in s["initial_body_clearance_m"].items() if n not in ("base_link","M_link1","M_link2")),
            off=a,moving=b,body_angle_difference_deg=angle(a["gravity_body"],b["gravity_body"]),
            posture_changed=a["posture_class"] != b["posture_class"],
            both_screened=a["mechanically_screened"] and b["mechanically_screened"],
            both_direction_stable=a["direction_stable"] and b["direction_stable"]))
    valid = [p for p in pairs if p["both_screened"]]
    stable = [p for p in valid if p["both_direction_stable"]]
    def stats(rows):
        return dict(pairs=len(rows),posture_changed=sum(p["posture_changed"] for p in rows),
            angle_difference_over_15deg=sum(p["body_angle_difference_deg"]>15. for p in rows),
            off_sustained_leg_support=sum(p["off"]["sustained_leg_support"] for p in rows),
            moving_sustained_leg_support=sum(p["moving"]["sustained_leg_support"] for p in rows),
            off_full_window=sum(p["off"]["full_window"] for p in rows),
            moving_full_window=sum(p["moving"]["full_window"] for p in rows),
            off_chassis_window=sum(p["off"]["chassis_window"] for p in rows),
            moving_chassis_window=sum(p["moving"]["chassis_window"] for p in rows),
            off_postures=dict(Counter(p["off"]["posture_class"] for p in rows)),
            moving_postures=dict(Counter(p["moving"]["posture_class"] for p in rows)))
    selected = []
    def choose(rows,n=1):
        for p in rows:
            if p["pair_id"] not in selected:
                selected.append(p["pair_id"])
                n -= 1
                if n == 0:
                    break
    for side in ("left","right"):
        subset = [p for p in stable if p["side"] == side]
        choose(sorted(subset,key=lambda p:p["body_angle_difference_deg"],reverse=True),2)
        choose(sorted([p for p in valid if p["side"] == side and p["roll_magnitude_deg"] == 90.],
            key=lambda p:p["moving"]["minimum_leg_wheel_clearance_m"],reverse=True),1)
        choose(sorted([p for p in subset if p["moving"]["sustained_leg_support"]],
            key=lambda p:p["moving"]["leg_and_wheel_support"]["mean_weight_fraction"],reverse=True),2)
    summary = dict(schema="wheel-leg-side-fall-friction-comparison-v1",source_bank=str(args.bank.resolve()),
        source_status=bank["status"],partial=args.partial,
        source_bank_sha256=digest(args.bank),source_cad_sha256=digest(args.cad),script_sha256=digest(__file__),
        dt_s=bank["physics_dt_s"],duration_s=bank["actual_timeout_s"],all_pairs=stats(pairs),
        both_mechanically_screened=stats(valid),both_direction_stable_and_screened=stats(stable),
        by_roll={str(roll):stats([p for p in valid if p["roll_magnitude_deg"] == roll]) for roll in sorted({p["roll_magnitude_deg"] for p in pairs})},
        record_outcomes=dict(Counter(r["status"] for r in bank["records"])),endpoint_cad_rejections=cad["review_required_count"],
        refinement_pair_ids=selected,physical_static_friction_identified=False,
        support_definition="last2s mean leg+wheel ground normal >=5% vehicle weight,over2N in>=80% samples",
        scope="Finite fixed-Coulomb-prior experiment;endpoint CAD screen excludes new triangle intersections,not full-trajectory self-collision qualification.",
        pairs=pairs)
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    save("comparison.json",summary)
    payload = dict(schema="wheel-leg-side-fall-paired-starts-v1",model_spec_sha256=bank["model_spec_sha256"],
        asset_manifest_sha256=bank["asset_manifest_sha256"],starts=[s for s in bank["starts"] if s["pair_id"] in selected],
        selected_from_bank_sha256=digest(args.bank))
    save("refinement_starts.json",payload)
    fields = ["pair_id","side","roll_magnitude_deg","knee_pair_deg","orbital_pair_deg","both_screened","both_direction_stable","body_angle_difference_deg","posture_changed"]
    with (args.output/"pairs.csv").open("w") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields+["off_posture","moving_posture","off_leg_weight_fraction","moving_leg_weight_fraction"])
        writer.writeheader()
        for p in pairs:
            writer.writerow({**{k:p[k] for k in fields},"off_posture":p["off"]["posture_class"],"moving_posture":p["moving"]["posture_class"],
                "off_leg_weight_fraction":p["off"]["leg_and_wheel_support"]["mean_weight_fraction"],
                "moving_leg_weight_fraction":p["moving"]["leg_and_wheel_support"]["mean_weight_fraction"]})
    print(json.dumps({k:v for k,v in summary.items() if k not in ("pairs","scope")},ensure_ascii=False))


if __name__ == "__main__":
    main()

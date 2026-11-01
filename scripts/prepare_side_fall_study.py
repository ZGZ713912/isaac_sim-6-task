#!/usr/bin/env python3
"""Compile feasible side starts and screen their original CAD surfaces before physics."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts"), str(ROOT/"tools")]
from build_full_vehicle_training import geometry, ResetGeometry
from wheeled_tasks.chassis.fallen_pose_bank import digest
from wheeled_tasks.chassis.full_vehicle_training import authenticate
from wheeled_tasks.chassis.side_fall_study import SCHEMA, build_side_starts, paired, orbital_angles, placement_fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rolls", type=float, nargs="+", default=[60.,75.,90.,105.,120.])
    p.add_argument("--orbits", type=float, nargs="+", default=[-90.,0.,90.])
    p.add_argument("--seed", type=int, default=261031)
    p.add_argument("--endpoint-bank",type=Path)
    p.add_argument("--endpoint-record",type=int)
    p.add_argument("--clearance-m",type=float,default=.002)
    args = p.parse_args()
    if not 0. < args.clearance_m <= .01:
        p.error("--clearance-m must be positive and at most .01")
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4.json").read_text())
    cfg = json.loads((ROOT/plan["mechanism"]["file"]).read_text())
    authenticate(cfg, ROOT)
    bundle = ROOT/cfg["asset_directory"]
    spec = json.loads((bundle/"model_spec.json").read_text())
    if args.endpoint_bank:
        if args.endpoint_record is None:
            p.error("Endpoint replacement needs --endpoint-record")
        parent = json.loads(args.endpoint_bank.read_text())
        if (parent["status"] != "completed" or parent["model_spec_sha256"] != digest(bundle/"model_spec.json")
                or parent["asset_manifest_sha256"] != digest(bundle/"manifest.json")):
            p.error("Endpoint asset differs")
        record = parent["records"][args.endpoint_record]
        if record["status"] not in ("settled","not_settled"):
            p.error("Endpoint failed its mechanical screen")
        pose = record["final_pose"]
        original = pose["full_joint_positions"]
        q = geometry.solve_pose(spec,{n:original[n] for n in (*geometry.ACTIVE,"M_joint1","M_joint2")},original)
        builder = ResetGeometry(spec,bundle)
        rotation = Rotation.from_quat(pose["root_quaternion_xyzw"])
        bounds = builder.bounds(q,rotation.as_matrix())
        height = float(-min(v[2] for _,v in bounds)+args.clearance_m)
        zero = spec["passive_knee_zero_inner_rad"]
        knees = np.rad2deg([q["L_joint2"]+zero["L_joint2"],zero["R_jonit2"]-q["R_jonit2"]]).tolist()
        start = dict(name="endpoint_replacement",pair_id=100000+parent["starts"][args.endpoint_record]["pair_id"],
            side="left" if pose["gravity_body"][1]>0 else "right",roll_magnitude_deg=abs(pose["roll_deg"]),
            signed_roll_deg=pose["roll_deg"],shape_id=0,knee_pair_deg=knees,orbital_pair_deg=orbital_angles(spec,q),
            leg_shape="measured_endpoint",sampling="closed_chain_projected_endpoint_replacement",
            gimbal_yaw_rad=q["M_joint1"],quaternion_xyzw=rotation.as_quat().tolist(),spawn_family=pose["family"],
            root_position_m=[0.,0.,height],full_joint_positions=q,min_collision_clearance_m=args.clearance_m,
            initial_body_clearance_m={n:min(float(v[2]+height) for b,v in bounds if b==n) for n in {b for b,_ in bounds}},
            initial_loop_gap_m=float(np.linalg.norm(geometry.closure_error(spec,q))),
            source_bank_sha256=digest(args.endpoint_bank),source_record=args.endpoint_record,
            passive_joint_projection_peak_rad_or_m=max(abs(q[n]-original[n]) for n in q))
        start["placement_sha256"] = placement_fingerprint(start)
        starts = [start]
    else:
        if args.endpoint_record is not None:
            p.error("--endpoint-record needs --endpoint-bank")
        starts = build_side_starts(spec,bundle,rolls=args.rolls,orbits=args.orbits,seed=args.seed,clearance_m=args.clearance_m)
    header = dict(schema=SCHEMA,asset_directory=cfg["asset_directory"],
        model_spec_sha256=digest(bundle/"model_spec.json"),asset_manifest_sha256=digest(bundle/"manifest.json"),
        seed=args.seed, roll_magnitudes_deg=args.rolls, orbital_grid_deg=args.orbits,
        gimbal_yaw="one random yaw per leg configuration;matched across roll,side and friction arms",
        source_hashes={str(f.relative_to(ROOT)):digest(f) for f in (Path(__file__),ROOT/"src/wheeled_tasks/chassis/side_fall_study.py")})
    if args.endpoint_bank:
        header.update(endpoint_bank=str(args.endpoint_bank.resolve()),endpoint_bank_sha256=digest(args.endpoint_bank),
            endpoint_record=args.endpoint_record,roll_magnitudes_deg=[starts[0]["roll_magnitude_deg"]],orbital_grid_deg=None,
            gimbal_yaw="Measured passive endpoint yaw retained;parent placement randomized yaw")
    initial = dict(header,status="prepared",records=[])
    for index,s in enumerate(starts):
        frame = np.eye(4)
        frame[:3,:3] = Rotation.from_quat(s["quaternion_xyzw"]).as_matrix()
        frame[:3,3] = s["root_position_m"]
        frames = geometry.fk(spec,s["full_joint_positions"],frame)
        poses = {n: list(t[:3,3])+Rotation.from_matrix(t[:3,:3]).as_quat().tolist() for n,t in frames.items()}
        initial["records"].append(dict(index=index,final_pose=dict(native_body_poses_world=poses)))
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    save("all_starts.json",dict(header,starts=starts))
    save("initial_bank.json",initial)
    subprocess.run([sys.executable,str(ROOT/"scripts/audit_fallen_pose_contacts.py"),"--bank",str(args.output/"initial_bank.json"),
        "--output",str(args.output/"initial_cad_screen.json"),"--partial"],check=True)
    audit = json.loads((args.output/"initial_cad_screen.json").read_text())
    rejected = {r["index"] for r in audit["records"] if r["review_required"]}
    accepted = [s for i,s in enumerate(starts) if i not in rejected]
    save("placements.json",dict(header,starts=paired(accepted),requested_base_cases=len(starts),
        accepted_base_cases=len(accepted),rejected_initial_cad_indices=sorted(rejected),
        initial_cad_screen_sha256=digest(args.output/"initial_cad_screen.json")))
    print(json.dumps(dict(requested_base_cases=len(starts),accepted_base_cases=len(accepted),
        paired_lanes=2*len(accepted),output=str(args.output/"placements.json"))),flush=True)


if __name__ == "__main__":
    main()

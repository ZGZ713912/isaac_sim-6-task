#!/usr/bin/env python3
"""Read-only CAD surface-contact screen for passive measured endpoints.

This does not enable self-collision in physics or qualify CAD/hardware. New
non-adjacent surface intersections require review and are excluded from the
screened gallery. Joint-connected and nominal assembly contacts are listed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/"tools")]
from build_full_vehicle_training import geometry
from wheeled_tasks.chassis.fallen_pose_bank import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--partial",action="store_true")
    args = parser.parse_args()
    bank = json.loads(args.bank.read_text())
    if bank["status"] != "completed" and not args.partial:
        raise ValueError("Wait for completed collection or explicitly request a partial preview")
    if args.output.exists():
        raise ValueError("Preserve existing audits")
    bundle = ROOT/bank["asset_directory"]
    if digest(bundle/"model_spec.json") != bank["model_spec_sha256"]:
        raise ValueError("Geometry changed")
    spec = json.loads((bundle/"model_spec.json").read_text())
    adjacent = {tuple(sorted((j["parent"],j["child"]))) for j in spec["joints"]}
    adjacent |= {tuple(sorted((c["body0"],c["body1"]))) for c in spec["constraints"]}
    manager = trimesh.collision.CollisionManager()
    frames = geometry.fk(spec,spec["nominal_joint_pos"])
    hashes,vertices = {},{}
    beginning = time.monotonic()
    for b in spec["bodies"]:
        path = bundle/b["mesh"]
        if not path.is_file():
            path = bundle.parent/"isaac"/b["mesh"]
        hashes[str(path.relative_to(ROOT))] = digest(path)
        mesh = trimesh.load(path,force="mesh",process=False)
        mesh.apply_transform(np.array(b["mesh_origin"]))
        vertices[b["name"]] = np.array(mesh.vertices)
        manager.add_object(b["name"],mesh,frames[b["name"]])
    def pairs():
        return {tuple(sorted(p)) for p in manager.in_collision_internal(return_names=True)[1]}
    nominal = pairs()
    exempt = nominal|adjacent
    records = []
    for r in bank["records"]:
        p = r["final_pose"]
        minimum = float("inf")
        for n,v in p["native_body_poses_world"].items():
            frame = np.eye(4)
            frame[:3,:3] = Rotation.from_quat(v[3:]).as_matrix()
            frame[:3,3] = v[:3]
            manager.set_transform(n,frame)
            minimum = min(minimum,float((vertices[n]@frame[2,:3]+frame[2,3]).min()))
        contacts = pairs()
        unexpected = contacts-exempt
        records.append(dict(index=r["index"],all_cad_surface_contacts=sorted(contacts),
            new_nonadjacent_cad_surface_contacts=sorted(unexpected),cad_minimum_z_m=minimum,
            review_required=bool(unexpected) or minimum < -.003))
        if len(records)%64 == 0:
            print(json.dumps(dict(audited=len(records),review_required=sum(v["review_required"] for v in records))),flush=True)
    output = dict(schema="passive-fallen-cad-surface-screen-v1",source_bank=str(args.bank.resolve()),
        source_bank_sha256=digest(args.bank),source_status=bank["status"],source_script_sha256=digest(__file__),
        source_mesh_sha256=hashes,mesh_decimation=False,adjacent_joint_and_loop_pairs=sorted(adjacent),
        nominal_assembly_contacts=sorted(nominal),records=records,
        review_required_count=sum(v["review_required"] for v in records),
        new_contact_pair_counts=dict(Counter("+".join(p) for r in records for p in r["new_nonadjacent_cad_surface_contacts"])),
        scope="Triangle-surface screen;adjacent and nominal assembly pairs exempt. No volume containment,contact dynamics or hardware qualification.",
        elapsed_wall_s=time.monotonic()-beginning)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in output.items() if k not in ("records","source_mesh_sha256","adjacent_joint_and_loop_pairs","nominal_assembly_contacts")}))


if __name__ == "__main__":
    main()

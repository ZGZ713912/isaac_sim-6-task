#!/usr/bin/env python3
"""Sample CAD intersections along FK-reconstructed traces, retaining explicit scope."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/"tools")]
from build_full_vehicle_training import geometry
from wheeled_tasks.chassis.fallen_pose_bank import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pair-ids",type=int,nargs="+",required=True)
    parser.add_argument("--period-s",type=float,default=.1)
    args = parser.parse_args()
    bank = json.loads(args.bank.read_text())
    if bank["status"] != "completed" or args.period_s < .01:
        raise ValueError("Need a completed measured trace and sampling >=10ms")
    args.output.mkdir(parents=True,exist_ok=False)
    spec = json.loads((ROOT/bank["asset_directory"]/"model_spec.json").read_text())
    columns = bank["trace_columns"]
    qi = [columns.index("q/"+n) for n in bank["native"]["joint_names"]]
    stride = round(args.period_s/.01)
    samples = []
    for chunk in bank["trace_files"]:
        path = args.bank.parent/chunk["path"]
        if digest(path) != chunk["sha256"]:
            raise ValueError("Measured trace changed")
        trace = np.load(path)["trace"]
        for j in range(chunk["count"]):
            index = chunk["first_record"]+j
            start = bank["starts"][index]
            if start["pair_id"] not in args.pair_ids:
                continue
            times = sorted(set([0,len(trace)-1]+list(range(stride-1,len(trace),stride))))
            for step in times:
                values = trace[step,j]
                frame = np.eye(4)
                frame[:3,:3] = Rotation.from_quat(values[3:7]).as_matrix()
                frame[:3,3] = values[:3]
                q = dict(zip(bank["native"]["joint_names"],values[qi]))
                frames = geometry.fk(spec,q,frame)
                poses = {n:list(t[:3,3])+Rotation.from_matrix(t[:3,:3]).as_quat().tolist() for n,t in frames.items()}
                samples.append(dict(index=len(samples),native_record=index,pair_id=start["pair_id"],
                    leg_friction_enabled=start["leg_friction_enabled"],time_s=(step+1)*.01,
                    final_pose=dict(native_body_poses_world=poses)))
    synthetic = dict(status="fk_reconstructed_trajectory",asset_directory=bank["asset_directory"],
        model_spec_sha256=bank["model_spec_sha256"],records=samples)
    path = args.output/"reconstructed_samples.json"
    path.write_text(json.dumps(synthetic,indent=2,allow_nan=False)+"\n")
    subprocess.run([sys.executable,str(ROOT/"scripts/audit_fallen_pose_contacts.py"),"--bank",str(path),
        "--output",str(args.output/"cad_screen.json"),"--partial"],check=True)
    audit = json.loads((args.output/"cad_screen.json").read_text())
    summaries = []
    for index in sorted({s["native_record"] for s in samples}):
        rows = [(s,audit["records"][s["index"]]) for s in samples if s["native_record"] == index]
        bad = [(s,a) for s,a in rows if a["review_required"]]
        summaries.append(dict(native_record=index,pair_id=rows[0][0]["pair_id"],
            leg_friction_enabled=rows[0][0]["leg_friction_enabled"],samples=len(rows),
            flagged_samples=len(bad),first_flagged_time_s=bad[0][0]["time_s"] if bad else None,
            new_contact_pairs=dict(Counter("+".join(p) for _,a in bad for p in a["new_nonadjacent_cad_surface_contacts"])),
            endpoint_native_vs_fk_error_m=bank["records"][index]["final_pose"]["reconstructed_fk_position_error_m"]))
    output = dict(schema="side-fall-reconstructed-trajectory-screen-v1",source_bank=str(args.bank.resolve()),
        source_bank_sha256=digest(args.bank),source_trace_hashes=bank["trace_files"],script_sha256=digest(__file__),
        sampling_period_s=stride*.01,actual_native_body_poses=False,
        scope="FK reconstruction from measured native q/root. Finite triangle-surface samples;cannot exclude intersections between samples or volume containment. Native/FK endpoint error is reported separately.",
        records=summaries)
    (args.output/"summary.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(output,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()

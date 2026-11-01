#!/usr/bin/env python3
"""Build and measure the v2.0.2 reset branch used by the Wheelbipe adapter."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"tools")]
import numpy as np
import full_robot_v202 as geometry
from wheeled_tasks.wheelbipe_port.model import BUNDLE, TABLE, digest


def main():
    spec = json.loads((BUNDLE/"model_spec.json").read_text())
    names = [j["name"] for j in spec["joints"] if j["type"] != "fixed"]
    joints = {j["name"]:j for j in spec["joints"]}
    q = spec["nominal_joint_pos"].copy()
    zeros = spec["passive_knee_zero_inner_rad"]
    positions, side_rows = [], [dict(lengths_m=[], angles_rad=[]),dict(lengths_m=[], angles_rad=[])]
    for beta in np.linspace(40., 110., 281):
        q = geometry.solve_pose(spec, dict(L_joint1=-.42, R_joint1=.42, L_joint3=0., R_joint3=0.,
            L_joint2=math.radians(beta)-zeros["L_joint2"],
            R_jonit2=zeros["R_jonit2"]-math.radians(beta)), q)
        positions.append([q[n] for n in names])
        frames = geometry.fk(spec, q)
        for k, prefix in enumerate(("L", "R")):
            joint = joints[prefix+"_joint1"]
            offset = frames[prefix+"_link3"][:3,3]-np.array(joint["origin"])[:3,3]
            side_rows[k]["lengths_m"].append(float(np.hypot(offset[0],offset[2])))
            side_rows[k]["angles_rad"].append(float(math.atan2(-offset[0], -offset[2])))
            axis = np.array(joint["origin"])[:3,:3]@np.array(joint["axis"])
            side_rows[k]["axis_y_sign"] = float(np.sign(axis[1]))
            side_rows[k]["active_pair"] = [prefix+"_joint1",prefix*2+"_joint1"]
            side_rows[k]["joint_columns"] = [i for i,n in enumerate(names) if n.startswith(prefix)]
    if any(np.any(np.diff(row["lengths_m"])<=0.) for row in side_rows):
        raise ValueError("Reset leg lengths must be strictly increasing")
    max_gap = 0.
    # Interpolation plus the common-angle approximation must satisfy all six
    # constraints, including independently folded and rotated left/right legs.
    rows = np.array(positions)
    for i in range(len(rows)-1):
        for left_angle,right_angle in ((-2.35,2.35),(0.,0.),(2.35,-2.35)):
            sample = .5*(rows[i]+rows[i+1])
            for k, target in enumerate((left_angle,right_angle)):
                row = side_rows[k]
                angle = .5*(row["angles_rad"][i]+row["angles_rad"][i+1])
                delta = (target-angle)/row["axis_y_sign"]
                for n in row["active_pair"]:
                    sample[names.index(n)] += delta
            gap = np.linalg.norm(geometry.closure_error(spec,dict(zip(names,sample))).reshape(-1,3),axis=-1).max()
            max_gap = max(max_gap,float(gap))
    if max_gap > 0.0001:
        raise ValueError(f"Reset interpolation/common rotation gap exceeds 0.1 mm: {max_gap}")
    TABLE.parent.mkdir(parents=True,exist_ok=True)
    result = dict(schema="wheelbipe-v202-reset-lookup-v1",model_spec_sha256=digest(BUNDLE/"model_spec.json"),
        joint_names=names,positions=positions,sides=side_rows,knee_domain_deg=[40.,110.],
        max_audited_constraint_gap_m=max_gap,audit_cases=3*(len(rows)-1),
        state_writes="episode_reset_only",hardware_calibrated=False)
    TABLE.write_text(json.dumps(result,separators=(",",":"))+"\n")
    print(json.dumps(dict(path=str(TABLE),sha256=digest(TABLE),max_gap_m=max_gap,
        length_ranges_m=[[r["lengths_m"][0],r["lengths_m"][-1]] for r in side_rows]),indent=2))


if __name__ == "__main__":
    main()

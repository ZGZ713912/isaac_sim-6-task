#!/usr/bin/env python3
"""Revalidate RMCS lookup poses and append feasible inverted reset coverage."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_robust_getup_bank import Builder
from build_v6_height_lookup import digest


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Reset bank extension never overwrites existing files")
    profile = ROOT / "reports/robust_policy_repair_20261024/rmcs_reference/v7_recovery_profiles_107_v2.json"
    if digest(profile) != "b966a5f65246775bd185d27871c3f42783ec25ee73e8df2d6ee3efbc769b9400":
        raise ValueError("RMCS reference differs from the reviewed branch snapshot")
    reference = json.loads(profile.read_text())
    original = json.loads(args.source.read_text())
    result = deepcopy(original)
    bundle = ROOT / original["asset_directory"]
    builder = Builder(bundle, digest(bundle / "manifest.json"))
    if (original["asset_manifest_sha256"] != builder.manifest_sha256
            or original["model_spec_sha256"] != digest(bundle / "model_spec.json")
            or reference["model_spec_sha256"] != original["model_spec_sha256"]):
        raise ValueError("Bank and RMCS reference mechanism specs differ")
    validated = {}
    shapes = []
    for name in ("fold", "thrust", "support"):
        q = reference["targets"][name]["full_joint_positions"]
        if set(q) != set(builder.nominal):
            raise ValueError("RMCS reference joint set differs")
        validated[name] = builder.validate_q(q)
        if not np.allclose([q[n] for n in builder.controls], reference["targets"][name]["control6_rad"], atol=1e-12):
            raise ValueError("Reference six-motor order differs")
        shapes.append(("rmcs_" + name, q))
    for beta in (50., 95., 103.):
        q = builder.align_wheels(builder.knee_pose(beta))
        for orbit in (-1.6, -.8, .8, 1.6, math.pi):
            shapes.append((f"rmcs_inverted_knee{beta:g}_orbit{orbit:.3f}",
                           builder.common_rotate(q, orbit, -orbit)))
    for name, q in shapes:
        physical = builder.validate_q(q)
        for roll, pitch in ((180., 0.), (0., 180.), (150., 0.), (-150., 0.)):
            rotation = Rotation.from_euler("xyz", [roll, pitch, 0.], degrees=True)
            bounds = builder.collision_bounds(q, rotation.as_matrix())
            result["poses"].append(dict(
                name=f"{name}_r{roll:g}_p{pitch:g}", family="inverted", leg_shape=name,
                rmcs_reference=True, roll_deg=roll, pitch_deg=pitch,
                root_position_m=[0., 0., float(.008-min(row["lower"][2] for row in bounds))],
                root_quaternion_xyzw=rotation.as_quat().tolist(), full_joint_positions=q,
                min_collision_clearance_m=.008, **physical))
    if result["poses"][:len(original["poses"])] != original["poses"]:
        raise ValueError("Existing reset indices or states changed")
    result["extension"] = dict(
        source=str(args.source), source_sha256=digest(args.source),
        original_pose_count=len(original["poses"]), appended_pose_count=len(result["poses"])-len(original["poses"]),
        original_poses_unchanged=True, reference_file=str(profile.relative_to(ROOT)),
        reference_sha256=digest(profile), reference_branch="dev/wheel_leg_rl",
        reference_commit="0fcc9b45cd80e4dba69c03a1fbed2b848ab5092a",
        reference_targets_revalidated=validated, native_settle_validated=False,
        dynamic_recovery_validated=False, scripted_actuator_fallback=False, hardware_ready=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output), sha256=digest(args.output),
                          poses=len(result["poses"]), appended=result["extension"]["appended_pose_count"],
                          max_gap_m=max(p["gap_m"] for p in result["poses"]))))


if __name__ == "__main__":
    main()

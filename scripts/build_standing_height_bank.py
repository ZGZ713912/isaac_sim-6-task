#!/usr/bin/env python3
"""Append closed-chain feasible births for a gradual height curriculum.

Only geometry and reset poses are generated. No model, PD compensation,
state-feedback controller or motor trajectory is constructed.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_robust_getup_bank import Builder
from recovery_static_targets import RecoveryStaticTargetBuilder


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StandingGeometry(RecoveryStaticTargetBuilder):
    validate_q = Builder.validate_q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "reports/robust_policy_repair_20261024/pose_bank_height_matched073.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=65)
    args = parser.parse_args()
    if args.output.exists() or not 17 <= args.samples <= 257:
        parser.error("Use a new output file and17..257 samples")
    original = json.loads(args.source.read_text())
    bundle = ROOT / original["asset_directory"]
    if digest(bundle / "manifest.json") != original["asset_manifest_sha256"]:
        raise ValueError("Geometry asset changed")
    builder = StandingGeometry(bundle, original["asset_manifest_sha256"])
    bank = deepcopy(original)
    rows = []
    heights = sorted(set([.23, .305, .42] + [round(float(value), 9) for value in np.linspace(.23, .42, args.samples)]))
    for height in heights:
        knee = brentq(lambda beta: builder.balanced_pose(beta)["height_m"] - height, 40.001, 106.999, xtol=1e-8)
        pose = builder.balanced_pose(knee)
        validation = builder.validate_q(pose["q"])
        if abs(pose["height_m"] - height) > 1e-8 or pose["minimum_ground_clearance_m"] < -1e-8:
            raise ValueError("Infeasible standing reset")
        index = len(bank["poses"])
        bank["poses"].append(dict(name=f"scratch_height_{height:.9f}", family="upright",
            roll_deg=float(np.rad2deg(pose["root_roll_rad"])), pitch_deg=0.,
            root_position_m=[0., 0., float(pose["height_m"])],
            root_quaternion_xyzw=Rotation.from_matrix(pose["rotation"]).as_quat().tolist(),
            full_joint_positions={name: float(value) for name, value in pose["q"].items()},
            min_collision_clearance_m=float(pose["minimum_ground_clearance_m"]),
            leg_shape="scratch_height_matched_closed_chain", requested_height_m=height,
            diagnostic_reset_only=True, **validation))
        rows.append(dict(height_m=height, pose_index=index, gap_m=validation["gap_m"],
            knee_deg=validation["knee_deg"], minimum_ground_clearance_m=float(pose["minimum_ground_clearance_m"])))
    if bank["poses"][:len(original["poses"])] != original["poses"]:
        raise ValueError("Original pose indices changed")
    bank["standing_height_grid"] = dict(version="scratch-standing-height-grid-v1", rows=rows,
        scope="reset-only feasible geometry; no PD compensation or learned policy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bank, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    receipt = dict(schema="scratch-standing-height-bank-v1", source=str(args.source), source_sha256=digest(args.source),
        bank=str(args.output), bank_sha256=digest(args.output), original_poses_preserved=True,
        original_pose_count=len(original["poses"]), appended=len(rows), rows=rows, new_PPO=0,
        static_PD_compensation_used=False, hardware_ready=False,
        sources_sha256={str(path.relative_to(ROOT)): digest(path) for path in (Path(__file__).resolve(),
            ROOT / "scripts/recovery_static_targets.py", ROOT / "scripts/build_robust_getup_bank.py",
            ROOT / "scripts/build_v6_recovery_profiles.py", builder.solver_path, bundle / "tools/v6_mechanism.py",
            bundle / "model_spec.json", bundle / "manifest.json")})
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: receipt[key] for key in ("appended", "bank_sha256", "original_poses_preserved")}))


if __name__ == "__main__":
    main()

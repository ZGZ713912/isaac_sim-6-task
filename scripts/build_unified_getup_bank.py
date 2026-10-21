#!/usr/bin/env python3
"""Compile closed-chain-safe fallen poses against the CURRENT paired-107 asset."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_v6_recovery_profiles import RecoveryProfileBuilder, digest


class Builder(RecoveryProfileBuilder):
    def validate_q(self, q):
        knees = self.knees(q)
        if any(not 40. - 1e-5 <= beta <= 107. + 1e-5 for beta in knees):
            raise ValueError(f"Paired-107 passive knee outside physical limits: {knees}")
        for name, binding in self.spec["spring_binding"].items():
            limit = self.joints[name]["limit"]
            compression = binding["compression_at_q_zero_m"] - q[name]
            if not (float(limit["lower"]) - 1e-8 <= q[name] <= float(limit["upper"]) + 1e-8
                    and -1e-8 <= compression <= binding["stroke_m"] + 1e-8):
                raise ValueError("Spring outside physical travel")
        gap = np.linalg.norm(self.closure_error(self.spec, q).reshape(-1, 3), axis=1).max()
        if gap > 1e-8:
            raise ValueError("Open linkage in reset bank")
        return dict(knee_deg=knees, gap_m=float(gap))


def build(output):
    bundle = ROOT / "model/纯底盘_v7_flat_base107/urdf"
    builder = Builder(bundle, digest(bundle / "manifest.json"))
    # Fixed feasible mechanism configurations, including independent left/right
    # leg lengths and opposing common rotations. Never randomize four joints.
    variants = []
    for beta in ((50, 50), (65, 65), (85, 85), (50, 85), (85, 50)):
        q = builder.align_wheels(builder.knee_pose(*beta))
        variants.append((f"knee{beta[0]}_{beta[1]}", q))
    base = variants[1][1]
    for left, right in ((.9, -.9), (-.9, .9), (math.pi, math.pi), (1.57, -1.57)):
        variants.append((f"orbit{left:.2f}_{right:.2f}", builder.common_rotate(base, left, right)))
    orientations = [("upright", 0, 0), ("front", 0, 35), ("front", 0, 65),
                    ("front", 0, 90), ("back", 0, -35), ("back", 0, -65),
                    ("back", 0, -90), ("left", 90, 0), ("right", -90, 0),
                    ("inverted", 180, 0), ("inverted", 0, 180),
                    ("compound", 65, 65), ("compound", -65, -65)]
    poses = []
    for name, q in variants:
        physical = builder.validate_q(q)
        for family, roll, pitch in orientations:
            rotation = Rotation.from_euler("xyz", [roll, pitch, 0], degrees=True)
            bounds = builder.collision_bounds(q, rotation.as_matrix())
            min_z = min(row["lower"][2] for row in bounds)
            poses.append(dict(name=f"{family}_r{roll}_p{pitch}_{name}", family=family,
                roll_deg=roll, pitch_deg=pitch, leg_shape=name,
                root_position_m=[0., 0., float(-min_z + .008)],
                root_quaternion_xyzw=rotation.as_quat().tolist(),
                full_joint_positions=q, min_collision_clearance_m=.008, **physical))
    result = dict(schema="wheel-leg-unified-reset-bank-v1", asset_directory=str(bundle.relative_to(ROOT)),
        asset_manifest_sha256=builder.manifest_sha256, model_spec_sha256=digest(bundle / "model_spec.json"),
        tree_joint_names=builder.manifest["tree_joint_names"], poses=poses,
        state_writes="episode_reset_only", motor_effort_during_settle="zero_all_six_axes",
        passive_gas_springs="retained", coverage="spawn_and_post_settle_classified_separately",
        self_collision_scope="existing_asset_filtered_pairs; no new self-collision qualification",
        geometry_scope="closed chain, knee and spring travel, terrain clearance; native settling still required")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(poses=len(poses), max_gap_m=max(p["gap_m"] for p in poses), output=str(output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)

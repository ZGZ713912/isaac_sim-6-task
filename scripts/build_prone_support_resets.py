#!/usr/bin/env python3
"""Build normally-up, belly-grounded placements with independent leg states.

These are reset parameters, not trajectories, equilibria or learned policies.
Native admission remains a separate prerequisite to training.
"""
from copy import deepcopy
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_multi_pose_recovery_bank import CadScreen, leg_configuration, orbital_angles
from build_full_vehicle_training import ResetGeometry, geometry
from wheeled_tasks.chassis.full_robot_v202 import digest


def build(output, receipt, seed=261106):
    if output.exists() or receipt.exists():
        raise ValueError("Preserve existing reset parameters and receipts")
    plan = json.loads((ROOT / "contracts/wheel_leg_rollover_chassis_slopes_v3.json").read_text())
    mechanism = json.loads((ROOT / plan["mechanism"]["file"]).read_text())
    bundle = ROOT / mechanism["asset_directory"]
    spec = json.loads((bundle / "model_spec.json").read_text())
    collision = ROOT / plan["chassis_collision"]["geometry"]["file"]
    exact_base = np.array(json.loads(collision.read_text())["source_extremes"])
    builder, screen = ResetGeometry(spec, bundle), CadScreen(spec, bundle)
    rng = np.random.default_rng(seed)
    rows, rejected = [], []
    for cls, signs in zip(("front_front", "rear_rear", "front_rear", "rear_front"),
                          ((1, 1), (-1, -1), (1, -1), (-1, 1))):
        count = 0
        for attempt in range(800):
            if count == 10:
                break
            knees = rng.uniform(65., 106., 2).tolist()
            orbits = (rng.uniform(105., 165., 2) * signs).tolist()
            try:
                q = leg_configuration(builder, knees, orbits)
                collisions, cad_min = screen.check(q, np.eye(3))
                if collisions:
                    raise ValueError("CAD surface intersections beyond original assembly exemptions")
                # Replace only the old conservative chassis collision support.
                bounds = [(name, v) for name, v in builder.bounds(q, np.eye(3)) if name != "base_link"]
                bounds.append(("base_link", exact_base.min(0)))
                minimum = min(v[2] for _, v in bounds)
                base_min = float(exact_base[:, 2].min())
                if minimum < base_min - 1e-6:
                    raise ValueError("Leg geometry would lift the body off the belly at birth")
                height = -minimum + .002
                if cad_min + height < -.0001:
                    raise ValueError("Original CAD surface would penetrate the flat floor")
                gap = float(np.linalg.norm(geometry.closure_error(spec, q).reshape(-1, 3), axis=1).max())
                if gap > 1e-7:
                    raise ValueError("Open linkage")
                rows.append(dict(name=f"prone_support_{cls}_{count:02d}", family="upright",
                    roll_deg=0., pitch_deg=0., root_position_m=[0., 0., height],
                    root_quaternion_xyzw=[0., 0., 0., 1.], full_joint_positions=q,
                    min_collision_clearance_m=.002, knee_deg=knees, gap_m=gap,
                    static_equilibrium_qualified=False,
                    recovery=dict(family="belly_down", leg_class=cls,
                        orbital_pair_deg=orbital_angles(spec, q), split="train" if count < 8 else "holdout",
                        dynamic=False, cad_surface_screen_passed=True, native_passive_admitted=False)))
                count += 1
            except ValueError as error:
                rejected.append(dict(leg_class=cls, attempt=attempt, reason=str(error)))
        if count != 10:
            raise ValueError(f"Incomplete belly-grounded reset cell: {cls}, {count}")
    result = dict(schema="wheel-leg-prone-support-resets-v1", seed=seed,
        scope="identity body orientation, zero velocities, fixed gimbal; only independent left/right legs vary",
        model_spec_sha256=digest(bundle / "model_spec.json"),
        collision_geometry_sha256=digest(collision), clearance_m=.002,
        self_collision_scope="original CAD surface intersection screen; adjacent/nominal assembly pairs exempt; no containment or stress certification",
        native_admitted=False, poses=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    proof = dict(schema="wheel-leg-prone-support-reset-build-v1", reset_file_sha256=digest(output),
        builder_sha256=digest(Path(__file__)), train=32, holdout=8, rejected=rejected,
        maximum_closure_m=max(p["gap_m"] for p in rows),
        root_height_range_m=[min(p["root_position_m"][2] for p in rows), max(p["root_position_m"][2] for p in rows)],
        source_mesh_sha256=screen.hashes, new_PPO=0, native_admitted=False)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.receipt)))

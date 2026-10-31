#!/usr/bin/env python3
"""Compile closed-chain, original-CAD-screened recovery placements for v4.

Uses python-fcl only at build/audit time, never in the learner. Placements are
near-ground initial states, not claims of static equilibria. No simulator/PPO.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from build_full_vehicle_training import ResetGeometry, geometry
from wheeled_tasks.chassis.full_robot_v202 import digest
from wheeled_tasks.chassis.multi_pose_recovery import FAMILIES, LEG_CLASSES, leg_class


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def orbital_angles(spec, q):
    frames = geometry.fk(spec, q)
    joints = {j["name"]: j for j in spec["joints"]}
    result = []
    for side in ("L", "R"):
        v = frames[side+"_link3"][:3, 3]-np.asarray(joints[side+"_joint1"]["origin"])[:3, 3]
        result.append(math.degrees(math.atan2(v[0], -v[2])))
    return result


def leg_configuration(builder, knees, orbits):
    q, spec = dict(builder.spec["nominal_joint_pos"]), builder.spec
    zero = spec["passive_knee_zero_inner_rad"]
    initial = [math.degrees(q["L_joint2"]+zero["L_joint2"]),
               math.degrees(zero["R_jonit2"]-q["R_jonit2"])]
    for a in np.linspace(0., 1., 20)[1:]:
        q = builder.solve(dict(L_joint1=q["L_joint1"], R_joint1=q["R_joint1"], L_joint3=0., R_joint3=0.,
            L_joint2=math.radians(initial[0]+a*(knees[0]-initial[0]))-zero["L_joint2"],
            R_jonit2=zero["R_jonit2"]-math.radians(initial[1]+a*(knees[1]-initial[1]))), q)
    current = orbital_angles(spec, q)
    delta = [math.radians((a-b+180.)%360.-180.)/builder.axis_sign(s)
             for a,b,s in zip(current, orbits, ("L", "R"))]
    steps = max(1, math.ceil(max(map(abs, delta))/.25))
    for _ in range(steps):
        q = builder.rotate(q, *(d/steps for d in delta))
    for _ in range(12):
        errors = [(a-b+180.)%360.-180. for a,b in zip(orbital_angles(spec, q), orbits)]
        if max(map(abs, errors)) < 1e-6:
            break
        q = builder.rotate(q, *(math.radians(e)/builder.axis_sign(s) for e,s in zip(errors, ("L", "R"))))
    if max(abs((a-b+180.)%360.-180.) for a,b in zip(orbital_angles(spec, q), orbits)) > 1e-5:
        raise ValueError("Orbital angle continuation did not converge")
    for j in spec["joints"]:
        if "lower" in j["limit"] and not float(j["limit"]["lower"])-1e-7 <= q[j["name"]] <= float(j["limit"]["upper"])+1e-7:
            raise ValueError("Sample violates a native joint limit")
    for name, binding in spec["spring_binding"].items():
        compression = binding["compression_at_q_zero_m"]-q[name]
        if not 0. <= compression <= binding["stroke_m"]:
            raise ValueError("Sample violates physical gas-spring stroke")
    return q


def tangent_velocity(spec, q, rng):
    """Project sampled active velocities onto the measured closed-chain tangent."""
    names = [j["name"] for j in spec["joints"]]
    residual, eps = geometry.closure_error(spec, q), 1e-6
    jacobian = np.column_stack([(geometry.closure_error(spec, {**q, n:q[n]+eps})-residual)/eps for n in names])
    active = [names.index(n) for n in geometry.ACTIVE]
    passive = [i for i,n in enumerate(names) if i not in active and n not in ("M_joint1", "M_joint2")]
    velocity = np.zeros(len(names))
    velocity[active] = rng.uniform(-.8, .8, len(active))
    velocity[passive] = np.linalg.lstsq(jacobian[:, passive], -jacobian[:, active]@velocity[active], rcond=None)[0]
    velocity *= min(1., 1.5/max(1.5, np.abs(velocity).max()))
    error = float(np.linalg.norm(jacobian@velocity))
    if error > 1e-5:
        raise ValueError("Dynamic start velocity is not on the closed-chain tangent")
    return dict(zip(names, velocity.tolist())), error


class CadScreen:
    def __init__(self, spec, bundle):
        self.spec = spec
        self.manager = trimesh.collision.CollisionManager()
        self.vertices, self.hashes = {}, {}
        frames = geometry.fk(spec, spec["nominal_joint_pos"])
        for body in spec["bodies"]:
            path = (bundle/body["mesh"]).resolve()
            if not path.is_file():
                path = (bundle.parent/"isaac"/body["mesh"]).resolve()
            mesh = trimesh.load(path, force="mesh", process=False)
            self.hashes[str(path.relative_to(ROOT))] = digest(path)
            mesh.apply_transform(np.array(body["mesh_origin"]))
            self.vertices[body["name"]] = np.array(mesh.vertices)
            self.manager.add_object(body["name"], mesh, frames[body["name"]])
        adjacent = {tuple(sorted((j["parent"], j["child"]))) for j in spec["joints"]}
        adjacent |= {tuple(sorted((c["body0"], c["body1"]))) for c in spec["constraints"]}
        self.nominal = self.pairs()
        self.exempt = adjacent|self.nominal

    def pairs(self):
        return {tuple(sorted(p)) for p in self.manager.in_collision_internal(return_names=True)[1]}

    def check(self, q, rotation):
        frames = geometry.fk(self.spec, q)
        minimum = float("inf")
        for name, vertices in self.vertices.items():
            self.manager.set_transform(name, frames[name])
            world = vertices@(rotation@frames[name][:3, :3]).T+rotation@frames[name][:3, 3]
            minimum = min(minimum, float(world[:, 2].min()))
        return sorted(self.pairs()-self.exempt), minimum


def build(output, *, samples=6, holdout=2, seed=261031):
    # Always derive from the preserved four-state bank. The current v4 plan
    # already selects our generated output and must not be used as the input.
    plan = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4_standard_20261031.json").read_text())
    mechanism = json.loads((ROOT/plan["mechanism"]["file"]).read_text())
    bundle = ROOT/mechanism["asset_directory"]
    spec = json.loads((bundle/"model_spec.json").read_text())
    bank = json.loads((ROOT/plan["pose_bank"]["file"]).read_text())
    builder, screen = ResetGeometry(spec, bundle), CadScreen(spec, bundle)
    rng = np.random.default_rng(seed)
    rejected = []
    rotations = dict(belly_down=(0.,0.,0.), left_side=(-90.,0.,0.), right_side=(90.,0.,0.),
        roof_down=(180.,0.,0.), front_down=(0.,90.,0.), rear_down=(0.,-90.,0.), diagonal=(65.,35.,0.))
    legacy = dict(belly_down="upright", left_side="left", right_side="right", roof_down="inverted",
                  front_down="front", rear_down="back", diagonal="compound")
    for family in FAMILIES:
        for cls_i, cls in enumerate(LEG_CLASSES):
            count = 0
            for attempt in range((samples+holdout)*80):
                if count == samples+holdout:
                    break
                signs = ((1,1),(-1,-1),(1,-1),(-1,1))[cls_i]
                # Full front/rear half-planes, with dense samples near the fore/aft boundary.
                magnitude = rng.uniform(8., 172., 2)
                if count % 4 == 0:
                    magnitude[count % 2] = rng.uniform(.2, 8.)
                orbits = (magnitude*np.array(signs)).tolist()
                knees = rng.uniform(45., 106., 2).tolist()
                try:
                    q = leg_configuration(builder, knees, orbits)
                    q["M_joint1"] = float(rng.uniform(-math.pi, math.pi))
                    q["M_joint2"] = 0.
                    euler = np.array(rotations[family])+rng.uniform(-4., 4., 3)
                    if family == "diagonal" and count % 2:
                        euler[0] *= -1.
                    rotation = Rotation.from_euler("xyz", euler, degrees=True)
                    collisions, cad_minimum = screen.check(q, rotation.as_matrix())
                    if collisions:
                        rejected.append(dict(family=family, leg_class=cls, contacts=collisions))
                        continue
                    minimum = min(v[2] for _,v in builder.bounds(q, rotation.as_matrix()))
                    height = -min(minimum, cad_minimum)+.002
                    gap = float(np.linalg.norm(geometry.closure_error(spec,q).reshape(-1,3), axis=1).max())
                    if gap > 1e-7:
                        raise ValueError("Sample has an open linkage")
                    zero = spec["passive_knee_zero_inner_rad"]
                    actual_knees = [math.degrees(q["L_joint2"]+zero["L_joint2"]),
                                   math.degrees(zero["R_jonit2"]-q["R_jonit2"])]
                    pose = dict(name=f"recovery_{family}_{cls}_{count:02d}", family=legacy[family],
                        roll_deg=float(euler[0]), pitch_deg=float(euler[1]), root_position_m=[0.,0.,float(height)],
                        root_quaternion_xyzw=rotation.as_quat().tolist(), full_joint_positions=q,
                        min_collision_clearance_m=float(minimum+height), knee_deg=actual_knees, gap_m=gap,
                        admitted_as_training_reset=True, static_equilibrium_qualified=False,
                        recovery=dict(family=family, leg_class=cls, orbital_pair_deg=orbital_angles(spec,q),
                            split="train" if count < samples else "holdout", dynamic=False,
                            cad_surface_screen_passed=True, cad_minimum_ground_clearance_m=float(cad_minimum+height)))
                    additions = [pose]
                    # Dynamic suffix births remain on the same position/velocity manifold.
                    if family == "belly_down" and count < samples:
                        dynamic = deepcopy(pose)
                        dynamic["name"] += "_dynamic"
                        dynamic["full_joint_velocities"], tangent_error = tangent_velocity(spec,q,rng)
                        dynamic["root_linear_velocity_body_m_s"] = rng.uniform(-.08,.08,3).tolist()
                        dynamic["root_angular_velocity_body_rad_s"] = rng.uniform(-.8,.8,3).tolist()
                        dynamic["recovery"].update(dynamic=True, closure_velocity_error_m_s=tangent_error)
                        additions.append(dynamic)
                    # Admit the whole sample atomically: a rejected dynamic
                    # tangent must not leave a duplicate static reset behind.
                    bank["poses"].extend(additions)
                    count += 1
                except ValueError as error:
                    rejected.append(dict(family=family, leg_class=cls, reason=str(error)))
            if count != samples+holdout:
                raise ValueError(f"Cannot compile a complete feasible reset cell: {family}/{cls}, {count}")
            print(json.dumps(dict(family=family, leg_class=cls, accepted=count)), flush=True)
    bank["geometry_scope"] = "Original standing/standard births plus full-vehicle near-ground multi-pose placements; no old pose/checkpoint transfer"
    bank["multi_pose_recovery"] = dict(schema="wheel-leg-multi-pose-bank-v1", seed=seed,
        samples_per_cell=samples, holdout_per_cell=holdout, self_collision_response=False,
        admission="closed_chain_joint_stroke_ground_and_original_cad_surface_screen",
        static_equilibrium_qualified=False, native_policy_behavior_qualified=False,
        cad_scope="Undecimated triangle-surface screen; adjacent and nominal assembly pairs exempt; no containment/stress qualification",
        nominal_assembly_contacts=sorted(screen.nominal), source_mesh_sha256=screen.hashes,
        builder_sha256=digest(__file__), source_pose_bank_sha256=plan["pose_bank"]["sha256"])
    save(output, bank)
    receipt = dict(schema="wheel-leg-multi-pose-bank-build-v1", bank_sha256=digest(output),
        poses=len(bank["poses"]), rejected=rejected, source_mesh_sha256=screen.hashes,
        builder_sha256=digest(__file__), new_PPO=0, static_equilibrium_qualified=False)
    save(output.with_suffix(".build.json"), receipt)
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"model/recovery_v202_20261031/pose_bank.json")
    parser.add_argument("--samples-per-cell", type=int, default=6)
    parser.add_argument("--holdout-per-cell", type=int, default=2)
    parser.add_argument("--seed", type=int, default=261031)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve the authenticated bank before rebuilding")
    if min(args.samples_per_cell,args.holdout_per_cell) < 1:
        parser.error("Both training and held-out reset cells need at least one sample")
    bank = build(args.output, samples=args.samples_per_cell, holdout=args.holdout_per_cell, seed=args.seed)
    print(json.dumps(dict(output=str(args.output), poses=len(bank["poses"]), sha256=digest(args.output))))


if __name__ == "__main__":
    main()

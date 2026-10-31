#!/usr/bin/env python3
"""Place independent leg combinations using original CAD and screen initial contacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from simulate_traditional_self_righting import ROOT, digest, save
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from build_full_vehicle_training import ResetGeometry, geometry
from wheeled_tasks.chassis.side_fall_study import leg_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=261031)
    parser.add_argument('--yaw-attempts', type=int, default=16)
    parser.add_argument('--knee', type=float, default=75.)
    args = parser.parse_args()
    if args.output.exists() or not 40. <= args.knee <= 110. or args.yaw_attempts < 1:
        raise ValueError('Need a new output, valid knee angle and positive attempt count')
    args.output.mkdir(parents=True)
    bundle = ROOT/'model/wheel_leg_v2.0.2/friction_v1/training'
    spec = json.loads((bundle/'model_spec.json').read_text())
    manifest = json.loads((bundle/'manifest.json').read_text())
    manager = trimesh.collision.CollisionManager()
    builder = ResetGeometry(spec, bundle)
    nominal = geometry.fk(spec, spec['nominal_joint_pos'])
    vertices, hashes = {}, {}
    for body in spec['bodies']:
        path = bundle/body['mesh']
        if not path.exists():
            path = bundle.parent/'isaac'/body['mesh']
        hashes[str(path.relative_to(ROOT))] = digest(path)
        mesh = trimesh.load(path, force='mesh', process=False)
        mesh.apply_transform(np.array(body['mesh_origin']))
        vertices[body['name']] = np.asarray(mesh.vertices)
        manager.add_object(body['name'], mesh, nominal[body['name']])
    adjacent = {tuple(sorted((j['parent'], j['child']))) for j in spec['joints']}
    adjacent |= {tuple(sorted((c['body0'], c['body1']))) for c in spec['constraints']}
    nominal_pairs = {tuple(sorted(p)) for p in manager.in_collision_internal(return_names=True)[1]}
    rng = np.random.default_rng(args.seed)
    combinations = dict(ff=(90.,90.), rr=(-90.,-90.), fr=(90.,-90.), rf=(-90.,90.))
    orientations = dict(prone=(0.,0.,0.), inverted=(180.,0.,0.),
        left=(-90.,0.,0.), right=(90.,0.,0.))
    records, rejected, missing = [], [], []
    for family, angles in orientations.items():
        rotation = Rotation.from_euler('xyz', angles, degrees=True)
        world = np.eye(4)
        world[:3, :3] = rotation.as_matrix()
        for combination, orbits in combinations.items():
            name = family+'_'+combination
            accepted = False
            for attempt in range(args.yaw_attempts):
                q = leg_configuration(builder, (args.knee, args.knee), orbits)
                q['M_joint1'] = float(rng.uniform(-np.pi, np.pi))
                frames = geometry.fk(spec, q)
                minimum = float('inf')
                for body, frame in frames.items():
                    transform = world@frame
                    manager.set_transform(body, transform)
                    minimum = min(minimum, float((vertices[body]@transform[2, :3]+transform[2, 3]).min()))
                pairs = {tuple(sorted(p)) for p in manager.in_collision_internal(return_names=True)[1]}
                unexpected = sorted(pairs-adjacent-nominal_pairs)
                if unexpected:
                    rejected.append(dict(name=name, attempt=attempt, yaw_rad=q['M_joint1'], new_pairs=unexpected))
                    continue
                records.append(dict(name=name, family=family, legs=combination,
                    root_position_m=[0.,0.,-minimum+.001], root_quaternion_xyzw=rotation.as_quat().tolist(),
                    full_joint_positions=q,
                    full_joint_velocities=dict.fromkeys(manifest['tree_joint_names'], 0.),
                    root_com_velocity_world=[0.]*6, static_equilibrium_proven=False,
                    initial_condition='CAD-grounded geometric placement with screened random yaw; not a measured passive equilibrium',
                    initial_cad_minimum_z_m=.001, new_nonadjacent_cad_surface_contacts=[],
                    yaw_attempt=attempt))
                accepted = True
                break
            if not accepted:
                missing.append(name)
    result = dict(schema='traditional-recovery-geometric-starts/v1', status='completed',
        source_sha256=digest(__file__), model_spec_sha256=digest(bundle/'model_spec.json'),
        source_mesh_sha256=hashes, seed=args.seed, knee_deg=args.knee, poses=records,
        rejected_yaw_candidates=rejected, uncovered_combinations=missing,
        adjacent_pairs=sorted(adjacent), nominal_assembly_contacts=sorted(nominal_pairs),
        scope='Original CAD triangle-surface screen at placement only. No volume containment, passive stability, '
              'trajectory collision or self-righting qualification. Random yaw is conditioned on this finite screen.')
    save(args.output/'starts.json', result)
    print(json.dumps(dict(starts=len(records), rejected=len(rejected), uncovered=missing)))


if __name__ == '__main__':
    main()

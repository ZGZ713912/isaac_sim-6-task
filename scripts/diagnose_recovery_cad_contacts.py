#!/usr/bin/env python3
"""Quantify flagged CAD surface contacts at recorded native snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import fcl
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from simulate_traditional_self_righting import ROOT, digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-contacts', type=int, default=32)
    parser.add_argument('--max-snapshots', type=int, default=0,
        help='Evenly select at most this many flagged snapshots; zero diagnoses every flagged snapshot')
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.max_contacts <= 1024 or args.max_snapshots<0:
        raise ValueError('Select a new output and 1..1024 contacts')
    screen_path = args.review/'cad_screen.json'
    samples_path = args.review/'samples.json'
    mapping_path = args.review/'sample_mapping.json'
    screen, samples, mapping = [json.loads(p.read_text()) for p in
                                (screen_path, samples_path, mapping_path)]
    samples, mapping = samples['records'], mapping['samples']
    bank = json.loads(samples_path.read_text())
    bundle = ROOT/bank['asset_directory']
    if digest(bundle/'model_spec.json') != bank['model_spec_sha256']:
        raise ValueError('Model differs from recorded snapshots')
    for name, expected in screen['source_mesh_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('CAD differs from the source screen: '+name)
    spec = json.loads((bundle/'model_spec.json').read_text())
    models = {}
    for body in spec['bodies']:
        path = bundle/body['mesh']
        if not path.exists():
            path = bundle.parent/'isaac'/body['mesh']
        mesh = trimesh.load(path, force='mesh', process=False)
        mesh.apply_transform(np.array(body['mesh_origin']))
        model = fcl.BVHModel()
        model.beginModel(len(mesh.vertices), len(mesh.faces))
        model.addSubModel(np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32))
        model.endModel()
        models[body['name']] = model
    selected=[i for i,row in enumerate(screen['records']) if row['new_nonadjacent_cad_surface_contacts']]
    total_flagged=len(selected)
    if args.max_snapshots and len(selected)>args.max_snapshots:
        selected=[selected[i]for i in np.linspace(0,len(selected)-1,args.max_snapshots,dtype=int)]
    result = dict(schema='traditional-recovery-contact-diagnostic/v1', status='running',
        source_sha256=digest(__file__),
        inputs={str(p): digest(p) for p in (screen_path, samples_path, mapping_path)},
        max_contacts_per_pair=args.max_contacts, records=[],
        total_flagged_self_contact_snapshots=total_flagged,selected_snapshot_indices=selected,
        scope='Limited FCL triangle-contact depths and sampled separating translations at finite snapshots. '
              'Depth is not solid-volume penetration. Translation is an upper bound along the tested directions, '
              'not a minimum-distance, containment, between-sample or mechanical-damage certification.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, result)
    for index in selected:
        row=screen['records'][index]
        poses = samples[index]['final_pose']['native_body_poses_world']
        for a, b in row['new_nonadjacent_cad_surface_contacts']:
            frames = {}
            for name in (a, b):
                pose = poses[name]
                frame = np.eye(4)
                frame[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
                frame[:3, 3] = pose[:3]
                frames[name] = frame
            transform = np.linalg.solve(frames[a], frames[b])
            first = fcl.CollisionObject(models[a])
            second = fcl.CollisionObject(models[b], fcl.Transform(transform[:3, :3], transform[:3, 3]))
            contacts = fcl.CollisionResult()
            fcl.collide(first, second,
                fcl.CollisionRequest(num_max_contacts=args.max_contacts, enable_contact=True), contacts)
            if not contacts.is_collision:
                raise ValueError('Flagged pair could not be reproduced')
            directions = [*np.eye(3), *-np.eye(3)]
            for contact in contacts.contacts:
                normal = np.asarray(contact.normal)
                length = np.linalg.norm(normal)
                if length > 1e-9:
                    directions.extend((normal/length, -normal/length))
            directions = np.unique(np.round(directions, 5), axis=0)
            def collides(direction, distance):
                second.setTransform(fcl.Transform(transform[:3, :3], transform[:3, 3]+direction*distance))
                return bool(fcl.collide(first, second, fcl.CollisionRequest(), fcl.CollisionResult()))
            best = None
            for direction in directions:
                if collides(direction, .03):
                    continue
                low, high = 0., .03
                for _ in range(12):
                    mid = (low+high)*.5
                    if collides(direction, mid):
                        low = mid
                    else:
                        high = mid
                if best is None or high < best['distance_m']:
                    best = dict(distance_m=high, direction_in_first_body=direction.tolist())
            record = dict(**mapping[index], pair=[a, b], returned_contact_count=len(contacts.contacts),
                largest_returned_fcl_depth_m=max(float(c.penetration_depth) for c in contacts.contacts),
                sampled_separating_translation=best)
            result['records'].append(record)
            save(args.output, result)
            print(json.dumps(record), flush=True)
    result['status'] = 'completed'
    save(args.output, result)


if __name__ == '__main__':
    main()

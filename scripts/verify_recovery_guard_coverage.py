#!/usr/bin/env python3
"""Check original gimbal CAD against queried cooked guard convex hulls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
import trimesh

from simulate_traditional_self_righting import ROOT, digest, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous coverage checks; select a new output')
    receipt_path=args.run/'receipt.json'
    cooked_path=args.run/'cooked_collision_geometry.json'
    receipt=json.loads(receipt_path.read_text())
    if receipt['collision_geometry']!='cad-gimbal-guards':
        raise ValueError('This check requires the temporary gimbal guard fixture')
    bundle=ROOT/receipt['asset_directory']
    if digest(bundle/'model_spec.json')!=receipt['model_spec_sha256']:
        raise ValueError('Vehicle model differs from the native receipt')
    spec=json.loads((bundle/'model_spec.json').read_text())
    rows=[]
    for item in json.loads(cooked_path.read_text())['meshes']:
        name=item['path'].split('/')[-2]
        if name not in ('M_link1','M_link2'):
            continue
        if len(item['convexes'])!=1:
            raise ValueError('Need a single cooked convex per gimbal body')
        body=next(b for b in spec['bodies']if b['name']==name)
        path=bundle/body['mesh']
        if not path.exists():
            path=bundle.parent/'isaac'/body['mesh']
        expected=receipt['temporary_collision_source_sha256'].get(str(path))
        if expected is None:
            expected=receipt['temporary_collision_source_sha256'].get(str(path.relative_to(ROOT)))
        if digest(path)!=expected:
            raise ValueError('CAD differs from the native collision source')
        mesh=trimesh.load(path,force='mesh',process=False)
        frame=np.asarray(body['mesh_origin'])
        vertices=np.asarray(mesh.vertices)@frame[:3,:3].T+frame[:3,3]
        hull=ConvexHull(np.asarray(item['convexes'][0]['vertices']))
        distance=(vertices@hull.equations[:,:3].T+hull.equations[:,3]).max()
        rows.append(dict(body=name,source_vertex_count=len(vertices),
            source_mesh_sha256=digest(path),maximum_signed_plane_distance_m=float(distance),
            minimum_plane_margin_m=float(-distance),all_vertices_contained=bool(distance<=1e-8)))
    if {r['body']for r in rows}!={'M_link1','M_link2'}:
        raise ValueError('Need queried cooked geometry for both gimbal bodies')
    result=dict(schema='traditional-recovery-cooked-guard-coverage/v2',source_sha256=digest(__file__),
        inputs={str(p):digest(p)for p in (receipt_path,cooked_path)},records=rows,
        all_vertices_contained=all(r['all_vertices_contained']for r in rows),
        scope='Every original CAD vertex, hence every triangle, lies within one queried cooked convex '
              'in the same local frame. This does not prove solver nonpenetration or trajectory admission.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save(args.output,result)
    print(json.dumps(result))
    if not result['all_vertices_contained']:
        raise SystemExit(1)


if __name__=='__main__':
    main()

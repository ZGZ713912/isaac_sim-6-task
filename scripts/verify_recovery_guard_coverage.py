#!/usr/bin/env python3
"""Check original gimbal CAD against queried cooked guard convex hulls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree
import trimesh

from simulate_traditional_self_righting import ROOT, digest, save


def component_vertices(mesh,original):
    edges=mesh.edges_unique
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),
                    shape=(len(mesh.vertices),len(mesh.vertices)))
    count,labels=connected_components(graph,directed=False)
    distances,matches=cKDTree(np.asarray(mesh.vertices)).query(np.asarray(original.vertices))
    if distances.max()>1e-7:
        raise ValueError('CAD processing changed original vertices beyond the connectivity tolerance')
    original_labels=labels[matches]
    face_labels=original_labels[np.asarray(original.faces)]
    if (face_labels!=face_labels[:,0,None]).any():
        raise ValueError('An original CAD triangle spans multiple cooked component indices')
    return [np.asarray(original.vertices)[original_labels==index]for index in range(count)]


def compare_vertices(vertices,cooked_vertices):
    hull=ConvexHull(cooked_vertices)
    distance=(vertices@hull.equations[:,:3].T+hull.equations[:,3]).max()
    row=dict(source_vertex_count=len(vertices),cooked_convex_vertex_count=len(cooked_vertices),
        maximum_signed_plane_distance_m=float(distance),minimum_plane_margin_m=float(-distance),
        all_vertices_contained=bool(distance<=1e-8))
    if np.linalg.matrix_rank(vertices-vertices.mean(0),tol=1e-10)==3:
        center=vertices.mean(0);scale=max(float(np.ptp(vertices,axis=0).max()),1e-9)
        normalized=(vertices-center)/scale
        original_hull=ConvexHull(normalized)
        original_convex=trimesh.Trimesh(vertices=normalized,faces=original_hull.simplices,process=False)
        _,outward_distance,_=trimesh.proximity.closest_point_naive(original_convex,(cooked_vertices-center)/scale)
        outward_distance*=scale
        original_hull=ConvexHull(vertices)
        inside=(cooked_vertices@original_hull.equations[:,:3].T+original_hull.equations[:,3]).max(1)<=0.
        outward_distance[inside]=0.
        row['original_convex_vertex_count']=len(original_hull.vertices)
        if np.isfinite(outward_distance).all():
            row['maximum_outward_distance_from_original_cad_convex_hull_m']=float(outward_distance.max())
        else:
            row['outward_distance_scope']='Numerically degenerate hull triangles; outward distance not evaluated'
    else:
        row['outward_distance_scope']='Degenerate source component; outward hull distance not evaluated'
    return row


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
    if receipt['status']!='completed':
        raise ValueError('Need a final completed receipt so the input hash is stable')
    component_mode=receipt['collision_geometry']=='cad-gimbal-component-guards'
    if receipt['collision_geometry'] not in ('cad-gimbal-guards','cad-gimbal-tight-guards','cad-gimbal-component-guards'):
        raise ValueError('This check requires the temporary gimbal guard fixture')
    bundle=ROOT/receipt['asset_directory']
    if digest(bundle/'model_spec.json')!=receipt['model_spec_sha256']:
        raise ValueError('Vehicle model differs from the native receipt')
    spec=json.loads((bundle/'model_spec.json').read_text())
    rows=[]
    meshes=json.loads(cooked_path.read_text())['meshes']
    for name in ('M_link1','M_link2'):
        pieces=[item for item in meshes if item['path'].split('/')[-2]==name]
        if not pieces or any(len(item['convexes'])!=1 for item in pieces):
            raise ValueError('Need one queried cooked convex per authored gimbal mesh')
        body=next(b for b in spec['bodies']if b['name']==name)
        path=bundle/body['mesh']
        if not path.exists():
            path=bundle.parent/'isaac'/body['mesh']
        expected=receipt['temporary_collision_source_sha256'].get(str(path))
        if expected is None:
            expected=receipt['temporary_collision_source_sha256'].get(str(path.relative_to(ROOT)))
        if digest(path)!=expected:
            raise ValueError('CAD differs from the native collision source')
        original=trimesh.load(path,force='mesh',process=False)
        mesh=trimesh.load(path,force='mesh',process=True)if component_mode else original
        frame=np.asarray(body['mesh_origin'])
        source_pieces=component_vertices(mesh,original)if component_mode else[np.asarray(original.vertices)]
        cooked={int(item['path'].split('RecoveryCadCollision')[-1]):item for item in pieces}
        if len(cooked)!=len(pieces)or set(cooked)!=set(range(len(source_pieces))):
            raise ValueError('Queried native component indices differ from original CAD connectivity')
        for index,source in enumerate(source_pieces):
            vertices=source@frame[:3,:3].T+frame[:3,3]
            cooked_vertices=np.asarray(cooked[index]['convexes'][0]['vertices'])
            rows.append(dict(body=name,component=index,source_mesh_sha256=digest(path),
                             **compare_vertices(vertices,cooked_vertices)))
    if {r['body']for r in rows}!={'M_link1','M_link2'}:
        raise ValueError('Need queried cooked geometry for both gimbal bodies')
    result=dict(schema='traditional-recovery-cooked-guard-coverage/v3',source_sha256=digest(__file__),
        inputs={str(p):digest(p)for p in (receipt_path,cooked_path)},records=rows,
        all_vertices_contained=all(r['all_vertices_contained']for r in rows),
        component_correspondence_checked=component_mode,
        scope='Check whether every original CAD vertex lies within its corresponding queried cooked convex '
              'in the same local frame. Every triangle belongs to one connected component and is therefore '
              'contained in that same convex when the check passes. Measure outward distance from the '
              'original component convex hull where it has three-dimensional extent. '
              'This does not bound filling of CAD concavities or prove solver nonpenetration or trajectory admission.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save(args.output,result)
    print(json.dumps(result))
    if not result['all_vertices_contained']:
        raise SystemExit(1)


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""CAD support/COM screen for exact side lying; no dynamic qualification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import math

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tools')]
import numpy as np
from scipy.spatial import ConvexHull
from build_full_vehicle_training import ResetGeometry,geometry
from wheeled_tasks.chassis.side_fall_study import leg_configuration


def analyze(output):
    plan=json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4.json').read_text())
    cfg=json.loads((ROOT/plan['mechanism']['file']).read_text())
    bundle=ROOT/cfg['asset_directory'];spec=json.loads((bundle/'model_spec.json').read_text())
    builder=ResetGeometry(spec,bundle)
    bodies={b['name']:b for b in spec['bodies']}
    q0=leg_configuration(builder,(75.,75.),(0.,0.))
    def moments(q,names):
        frames=geometry.fk(spec,q)
        return sum(bodies[n]['mass']*geometry.point(frames[n],bodies[n]['com']) for n in names)
    groups={side:[b['name'] for b in spec['bodies'] if b['name'].startswith(side)]for side in ['L','R','M']}
    base_moment=moments(q0,['base_link']);total=sum(b['mass']for b in spec['bodies'])
    left=[];right=[];gap=0.
    for beta in np.linspace(40.01,109.99,29):
        zero=leg_configuration(builder,(beta,beta),(0.,0.))
        for angle in np.linspace(-180.,180.,145)[:-1]:
            q=zero.copy()
            for name in ('L_joint1','LL_joint1'):q[name]-=math.radians(angle)
            for name in ('R_joint1','RR_joint1'):q[name]+=math.radians(angle)
            gap=max(gap,float(np.linalg.norm(geometry.closure_error(spec,q).reshape(-1,3),axis=1).max()))
            for side,rows in [('L',left),('R',right)]:rows.append(dict(beta=beta,orbit=angle,moment=moments(q,groups[side]).tolist()))
    gimbal=[]
    for yaw in np.linspace(-180.,180.,145)[:-1]:
        for pitch in np.linspace(-30.,35.,14):
            q=q0.copy();q['M_joint1']=math.radians(yaw);q['M_joint2']=math.radians(pitch)
            gimbal.append(dict(yaw=yaw,pitch=pitch,moment=moments(q,groups['M']).tolist()))
    cloud=[]
    for collision in bodies['base_link']['collisions']:
        if collision['type']!='mesh':raise ValueError('Expected base CAD decomposition meshes')
        t=np.array(collision['origin']);cloud.append(builder.meshes[collision['file']]@t[:3,:3].T+t[:3,3])
    cloud=np.concatenate(cloud)
    results=[]
    for side,sign in [('left',1.),('right',-1.)]:
        z=sign*cloud[:,1];minimum=z.min()
        # X/-Z are the support coordinates for a +90-degree roll. Mirroring
        # the second horizontal coordinate preserves all signed distances.
        contacts=cloud[z<minimum+.0001][:,[0,2]]*np.array([1.,-sign])
        hull=ConvexHull(contacts);edges=[]
        for equation in hull.equations:
            n=equation[:2];offset=equation[2]
            vector=np.array([n[0],0.,-sign*n[1]])
            selected=[];score=float(base_moment@vector)
            for rows in (left,right,gimbal):
                index=int(np.argmax(np.array([r['moment']for r in rows])@vector))
                selected.append(rows[index]);score+=float(np.array(rows[index]['moment'])@vector)
            edges.append(dict(outward_max_m=score/total+offset,
                edge=equation.tolist(),leg_left=selected[0],leg_right=selected[1],gimbal=selected[2]))
        maximum=max(e['outward_max_m']for e in edges)
        results.append(dict(side=side,base_ground_height_m=-float(minimum),support_polygon_m=contacts[hull.vertices].tolist(),
            sampled_com_can_cross_support=bool(maximum>0.),maximum_outward_m=maximum,edges=edges))
    receipt=dict(schema='side-recovery-support-screen/v1',source='full v202 friction_v1 model_spec and collision decomposition',
        independent_leg_configurations=len(left),gimbal_configurations=len(gimbal),kinematic_loop_gap_peak_m=gap,
        limitation='Finite configuration grid and rigid flat base face. Gimbal extrema are an overapproximation of passive attainable motion. No proof of dynamic impossibility or feasibility.',
        results=results)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps([dict(side=r['side'],outward_max_m=r['maximum_outward_m'])for r in results]))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    analyze(p.parse_args().output)

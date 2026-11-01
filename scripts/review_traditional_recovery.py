#!/usr/bin/env python3
"""Render measured body poses and optionally screen CAD surfaces along recovery."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from simulate_traditional_self_righting import digest,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',required=True)
    parser.add_argument('--audit-period',type=float,default=.1)
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--full-window',action='store_true',help='Audit beyond first completion through the entire observation window')
    args=parser.parse_args()
    receipt=json.loads((args.run/'receipt.json').read_text())
    if receipt['status']!='completed' or digest(args.run/'trace.npz')!=receipt['trace_sha256']:
        raise ValueError('Need a completed, unchanged native trace')
    args.output.mkdir(parents=True,exist_ok=False)
    sources={}
    for source in (Path(__file__),ROOT/'scripts/audit_fallen_pose_contacts.py'):
        target=args.output/'sources'/source.name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);sources[str(source.resolve())]=digest(target)
    data=np.load(args.run/'trace.npz');times=data['body_time_s'];poses=data['body_poses'];trace=data['values']
    settings=json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4.json').read_text())
    mechanism=json.loads((ROOT/settings['mechanism']['file']).read_text())
    bundle=ROOT/mechanism['asset_directory'];spec=json.loads((bundle/'model_spec.json').read_text())
    if digest(bundle/'model_spec.json')!=receipt['model_spec_sha256']:raise ValueError('Model differs')
    names=receipt['native_body_names'];rows=receipt['records'];indices=[]
    for name in args.cases:indices.append(next(i for i,r in enumerate(rows)if r['name']==name))
    parts=[]
    for body in spec['bodies']:
        path=bundle/body['mesh']
        if not path.exists():path=bundle.parent/'isaac'/body['mesh']
        mesh=trimesh.load(path,force='mesh',process=True)
        if len(mesh.faces)>1800:mesh=mesh.simplify_quadric_decimation(face_count=1800)
        parts.append((names.index(body['name']),body['name'],np.array(mesh.vertices),np.array(mesh.faces),np.array(body['mesh_origin'])))
    figure_height=4.2*len(indices)+.65
    fig=plt.figure(figsize=(17,figure_height),facecolor='#f8fafc')
    fig.subplots_adjust(left=.01,right=.99,top=1.-.8/figure_height,bottom=.025,wspace=.01,hspace=.22)
    sampled=[];mapping=[];figure_end_times={}
    settle=float(receipt.get('requested_passive_settle_seconds',0.))
    for row_index,i in enumerate(indices):
        record=rows[i]
        duration=record['completion_seconds']
        if duration is None:duration=record.get('first_fault_seconds')
        end=float(times[-1])if duration is None else min(float(times[-1]),settle+float(duration))
        figure_end_times[record['name']]=end
        chosen=[int(np.abs(times-t).argmin())for t in np.linspace(times[0],end,5)]
        for col,k in enumerate(chosen):
            ax=fig.add_subplot(len(indices),5,row_index*5+col+1,projection='3d',computed_zorder=False)
            center=poses[k,i,names.index('base_link'),:3].copy();center[2]=0.
            for bi,name,vertices,faces,local in parts:
                body=poses[k,i,bi];frame=np.eye(4)
                frame[:3,:3]=Rotation.from_quat(body[3:]).as_matrix();frame[:3,3]=body[:3]-center
                transform=frame@local;v=vertices@transform[:3,:3].T+transform[:3,3]
                color='#25313d'if name in ('L_link3','R_link3')else'#edaa35'if name.startswith('M_')else'#899aaa'if name=='base_link'else'#357db9'if name.startswith('L')else'#d06557'
                ax.add_collection3d(Poly3DCollection(v[faces],facecolors=color,linewidths=0.,shade=True,zorder=3))
            square=[[-.65,-.65,0],[.65,-.65,0],[.65,.65,0],[-.65,.65,0]]
            ax.add_collection3d(Poly3DCollection([square],facecolors='#dce4ec',alpha=.5,zorder=1))
            ax.set(xlim=(-.57,.57),ylim=(-.57,.57),zlim=(-.03,.78));ax.set_box_aspect((1,1,.71))
            ax.set_proj_type('ortho');ax.view_init(elev=22,azim=-55);ax.set_axis_off()
            ax.set_title(f"t = {times[k]:.2f} s",fontsize=10)
            if col==0:
                fig.text(.018,ax.get_position().y1+.2/figure_height,
                    textwrap.fill(record['name'],width=110,break_on_hyphens=False),
                    fontsize=9.5,fontweight='bold',va='bottom')
        if args.audit:
            if args.full_window:end=float(times[-1])
            stride=max(1,round(args.audit_period/float(np.median(np.diff(times)))))
            steps=sorted(set([0,int(np.abs(times-end).argmin()),*range(0,int(np.searchsorted(times,end)),stride)]))
            for k in steps:
                index=len(sampled)
                sampled.append(dict(index=index,final_pose=dict(native_body_poses_world=dict(zip(names,poses[k,i].tolist())))))
                mapping.append(dict(index=index,case=record['name'],time_s=float(times[k])))
    fig.suptitle('Measured native body poses | blue/red: legs, orange: passive gimbal | time since placement',fontsize=11,y=.985)
    fig.savefig(args.output/'measured_poses.png',dpi=150);plt.close(fig)
    n=len(receipt['native_joint_names']);leg_ids=[receipt['native_joint_names'].index(j)for j in ('L_joint1','LL_joint1','R_joint1','RR_joint1')]
    fig,axes=plt.subplots(len(indices),3,figsize=(14,2.7*len(indices)),squeeze=False)
    time=(np.arange(len(trace))+1)*float(data['dt_s'])
    for r,i in enumerate(indices):
        x=trace[:,i];g0=7+2*n
        axes[r,0].plot(time,np.rad2deg(np.arccos(np.clip(-x[:,g0+2],-1,1))));axes[r,0].set_ylabel(rows[i]['name']+'\nTilt (deg)')
        axes[r,1].plot(time,x[:,2]);axes[r,1].axhline(.305,color='#888',ls='--');axes[r,1].set_ylabel('Height (m)')
        axes[r,2].plot(time,x[:,7+n+np.array(leg_ids)]);axes[r,2].axhline(3.14,color='red',ls='--');axes[r,2].axhline(-3.14,color='red',ls='--');axes[r,2].set_ylabel('Four shaft speeds (rad/s)')
        for ax in axes[r]:
            if settle>0.:ax.axvspan(0.,settle,color='#b8c4cf',alpha=.35)
            ax.grid(alpha=.2);ax.set_xlabel('Time since placement (s)')
    fig.tight_layout();fig.savefig(args.output/'measured_trajectory.png',dpi=150);plt.close(fig)
    summary=dict(schema='traditional-recovery-native-review/v1',native_run=str(args.run.resolve()),
        native_trace_sha256=receipt['trace_sha256'],actual_native_body_poses=True,cases=args.cases,
        source_sha256=digest(__file__),sources=sources,plot_time_origin='episode placement',
        passive_settle_seconds=settle,figure_end_times_since_placement_s=figure_end_times,
        scope='Review figures; rendering does not qualify collisions or hardware')
    if args.audit:
        save(args.output/'sample_mapping.json',dict(samples=mapping))
        save(args.output/'samples.json',dict(status='completed',asset_directory=mechanism['asset_directory'],
            model_spec_sha256=receipt['model_spec_sha256'],records=sampled))
        subprocess.run([sys.executable,str(ROOT/'scripts/audit_fallen_pose_contacts.py'),'--bank',str(args.output/'samples.json'),
            '--output',str(args.output/'cad_screen.json')],check=True)
        audit=json.loads((args.output/'cad_screen.json').read_text());screen=[]
        for name in args.cases:
            chosen=[audit['records'][r['index']]for r in mapping if r['case']==name]
            screen.append(dict(case=name,samples=len(chosen),flagged=sum(a['review_required']for a in chosen),
                new_pairs=dict(Counter('+'.join(p)for a in chosen for p in a['new_nonadjacent_cad_surface_contacts']))))
        summary.update(audit_period_s=max(args.audit_period,float(np.median(np.diff(times)))),cad_screen=screen,
            full_window=args.full_window,
            cad_scope='Finite triangle-surface samples '+('through the full observation window'if args.full_window else'until first recovery completion or fault')+'; adjacent and nominal assembly contacts exempt; no containment or between-sample proof')
    save(args.output/'review.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()

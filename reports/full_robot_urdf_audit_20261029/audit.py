"""Read-only geometry/inertia audit. Run with the isaacsim6 Python environment.

Only writes audit results next to this script; does not start a physics engine.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REF = Path('/home/noir/Documents/workspace/example/wheeled-legged_RL/assets/robots/wheelbipeV14_2_1')
FULL = ROOT / 'model/wheel_legv2.0.0/urdf/轮腿urdf.SLDASM.urdf'
ALIGN = Rotation.from_euler('z', 90, degrees=True).as_matrix()


def vec(s, default='0 0 0'):
    return np.fromstring(s or default, sep=' ')


def matrix(origin):
    t = np.eye(4)
    if origin is not None:
        t[:3, :3] = Rotation.from_euler('xyz', vec(origin.get('rpy'))).as_matrix()
        t[:3, 3] = vec(origin.get('xyz'))
    return t


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def point(t, p):
    return t[:3, :3] @ np.asarray(p) + t[:3, 3]


def parse(path):
    r = ET.parse(path).getroot()
    ls = {l.get('name'): l for l in r.findall('link')}
    children = {j.find('child').get('link') for j in r.findall('joint')}
    roots = sorted(set(ls) - children)
    frames = {name: np.eye(4) for name in roots}
    pending = list(r.findall('joint'))
    joints = []
    while pending:
        old = len(pending)
        for j in pending[:]:
            parent, child = j.find('parent').get('link'), j.find('child').get('link')
            if parent not in frames:
                continue
            t = matrix(j.find('origin'))
            frames[child] = frames[parent] @ t
            a = vec(j.find('axis').get('xyz')) if j.find('axis') is not None else np.array([1., 0, 0])
            world_a = frames[child][:3, :3] @ a
            joints.append(dict(name=j.get('name'), type=j.get('type'), parent=parent, child=child,
                               xyz=t[:3, 3].tolist(), rpy=vec(j.find('origin').get('rpy')).tolist(),
                               axis=a.tolist(), axis_norm=float(np.linalg.norm(a)),
                               world_axis=world_a.tolist(), world_origin=frames[child][:3, 3].tolist(),
                               limit=j.find('limit').attrib if j.find('limit') is not None else None))
            pending.remove(j)
        if old == len(pending):
            raise ValueError(f'Unresolvable tree in {path}')
    links = []
    for n, l in ls.items():
        inertial = l.find('inertial')
        mass = float(inertial.find('mass').get('value'))
        a = {k: float(v) for k, v in inertial.find('inertia').attrib.items()}
        inertia = np.array([[a['ixx'], a['ixy'], a['ixz']],
                            [a['ixy'], a['iyy'], a['iyz']],
                            [a['ixz'], a['iyz'], a['izz']]])
        eig = np.linalg.eigvalsh(inertia)
        com = matrix(inertial.find('origin'))[:3, 3]
        links.append(dict(name=n, mass_kg=mass, com_local=com.tolist(), com_world=point(frames[n], com).tolist(),
                          inertia=inertia.tolist(), inertia_eigenvalues=eig.tolist(),
                          positive_definite=bool(eig[0] > 0),
                          triangle_valid=bool(eig[0]+eig[1] >= eig[2]-1e-12)))
    total = sum(l['mass_kg'] for l in links)
    com = sum(l['mass_kg'] * np.array(l['com_world']) for l in links) / total
    data = dict(path=str(path), sha256=sha(path), robot_name=r.get('name'), roots=roots,
                link_count=len(links), joint_count=len(joints), joint_types=dict(Counter(j['type'] for j in joints)),
                total_mass_kg=total, zero_pose_com=com.tolist(),
                invalid_inertia_links=[l['name'] for l in links if not l['positive_definite'] or not l['triangle_valid']],
                links=links, joints=joints)
    return data, frames, ls


def mesh_path(path, link):
    filename = link.find('visual/geometry/mesh').get('filename')
    if filename.startswith('package://'):
        return path.parent.parent / filename.split('/', 3)[3]
    return path.parent / filename


def full_mesh_audit(frames, links):
    paths = {n: mesh_path(FULL, l) for n, l in links.items()}
    meshes = {n: trimesh.load_mesh(p, process=True) for n, p in paths.items()}
    ref = meshes['RR_link4']
    target = trimesh.transform_points(ref.vertices, frames['RR_link4'])
    tree = cKDTree(target)
    records = {}
    for n, mesh in meshes.items():
        parts = list(mesh.split(only_watertight=False))
        matches = []
        for part in parts:
            if len(part.faces) != len(ref.faces):
                continue
            world = trimesh.transform_points(part.vertices, frames[n])
            err = float(tree.query(world)[0].max())
            if err < 2e-4 and abs(abs(part.volume / ref.volume) - 1) < 1e-4:
                matches.append(dict(faces=len(part.faces), max_vertex_distance_m=err))
        records[n] = dict(path=str(paths[n]), exists=paths[n].exists(), sha256=sha(paths[n]),
                          faces=len(mesh.faces), vertices=len(mesh.vertices), bounds_m=mesh.bounds.tolist(),
                          watertight=bool(mesh.is_watertight), components=len(parts), volume_m3=float(mesh.volume),
                          component_face_counts=sorted([len(p.faces) for p in parts], reverse=True),
                          rr_link4_world_matches=matches)
        if n in ('L_link3', 'R_link3'):
            records[n]['wheel_radius_xy_m'] = float(np.linalg.norm(mesh.vertices[:, :2], axis=1).max())
        print(f'mesh {n}: {len(mesh.faces)} faces, {len(parts)} components, RR4 matches {len(matches)}', flush=True)
    return records, meshes


def gf_list(v):
    if v is None:
        return None
    return [float(x) for x in v]


def limit_value(v):
    return v if v is None or np.isfinite(v) else ('+inf' if v > 0 else '-inf')


def usd_audit():
    stage = Usd.Stage.Open(str(REF / 'wheelbipeV14_2.usd'))
    cache = UsdGeom.XformCache()
    bodies, joints = [], []
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(p)
            bodies.append(dict(name=p.GetName(), path=str(p.GetPath()), mass_kg=m.GetMassAttr().Get(),
                               com=gf_list(m.GetCenterOfMassAttr().Get()), inertia=gf_list(m.GetDiagonalInertiaAttr().Get()),
                               world_origin=gf_list(cache.GetLocalToWorldTransform(p).ExtractTranslation())))
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            local0, local1 = j.GetLocalPos0Attr().Get(), j.GetLocalPos1Attr().Get()
            w0 = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(b0[0])).Transform(Gf.Vec3d(local0))
            w1 = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(b1[0])).Transform(Gf.Vec3d(local1))
            joints.append(dict(name=p.GetName(), path=str(p.GetPath()), type=p.GetTypeName(),
                               body0=[str(x) for x in b0], body1=[str(x) for x in b1],
                               local_pos0=gf_list(local0), local_pos1=gf_list(local1),
                               world_pos0=gf_list(w0), world_pos1=gf_list(w1),
                               anchor_gap_m=float(np.linalg.norm(np.array(w0)-np.array(w1))),
                               axis=p.GetAttribute('physics:axis').Get(),
                               lower=limit_value(p.GetAttribute('physics:lowerLimit').Get()), upper=limit_value(p.GetAttribute('physics:upperLimit').Get()),
                               excluded=bool(j.GetExcludeFromArticulationAttr().Get())))
    return dict(path=str(REF / 'wheelbipeV14_2.usd'),
                layers=[dict(path=l.realPath, sha256=sha(Path(l.realPath))) for l in stage.GetUsedLayers() if l.realPath],
                up_axis=str(UsdGeom.GetStageUpAxis(stage)), meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
                bodies=bodies, joints=joints, total_mass_kg=sum(b['mass_kg'] or 0 for b in bodies),
                joint_types=dict(Counter(j['type'] for j in joints)),
                loop_joints=[j for j in joints if '/loop_joints/' in j['path']])


def comparison(models, all_frames):
    full = models['full']
    v5 = models['v5_source']
    raw_changes = []
    full_j = {j['name']: j for j in full['joints']}
    for old in v5['joints']:
        new = full_j[old['name']]
        raw_changes.append(dict(name=old['name'], type_before=old['type'], type_after=new['type'],
                                xyz_delta_m=(np.array(new['xyz'])-old['xyz']).tolist(),
                                axis_before=old['axis'], axis_after=new['axis'],
                                orientation_delta_deg=float(np.degrees(Rotation.from_matrix(
                                    Rotation.from_euler('xyz', new['rpy']).as_matrix() @
                                    Rotation.from_euler('xyz', old['rpy']).as_matrix().T).magnitude()))))
    v6=all_frames['v6']
    full_f=all_frames['full']
    position_differences={n:dict(full_aligned=(ALIGN@full_f[n][:3,3]).tolist(),
                                v6=v6[n][:3,3].tolist(),
                                delta_mm=float(np.linalg.norm(ALIGN@full_f[n][:3,3]-v6[n][:3,3])*1000))
                          for n in v6}
    # Map measured V6 pins back through the recorded local Rz(+90) conversion.
    # This is a diagnostic transplant of old pins, not a new full-vehicle closure certification.
    constraints=json.loads((ROOT/'model/纯底盘_v6_232mm/urdf/constraints.json').read_text())['constraints']
    pin_residuals=[]
    for c in constraints:
        if 'spring' in c['name']:
            continue
        p0=point(full_f[c['body0']],ALIGN.T@c['local_pos0_m'])
        p1=point(full_f[c['body1']],ALIGN.T@c['local_pos1_m'])
        pin_residuals.append(dict(name=c['name'], gap_m=float(np.linalg.norm(p0-p1)),
                                  source='V6 measured pins mapped to raw full link frames'))
    a=np.array(full_j['L_joint1']['world_axis']); b=np.array(full_j['RR_joint1']['world_axis'])
    axis_error=float(np.degrees(np.arccos(np.clip(abs(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),0,1))))
    link_map={l['name']:l for l in full['links']}
    chassis_mass=sum(l['mass_kg'] for l in full['links'] if not l['name'].startswith('M_'))
    axis_dots={}
    for j in full['joints']:
        old=next((o for o in models['v6']['joints'] if o['name']==j['name']),None)
        if old is not None:
            v1=ALIGN@np.array(j['world_axis']);v2=np.array(old['world_axis'])
            axis_dots[j['name']]=float(v1@v2/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    return dict(raw_v5_joint_changes=raw_changes, aligned_full_vs_v6_origins=position_differences,
                aligned_full_vs_v6_axis_dots=axis_dots,
                diagnostic_transplanted_loop_pin_residuals=pin_residuals,
                rr_joint1_unoriented_axis_error_vs_left_hip_deg=axis_error,
                full_chassis_only_mass_kg=chassis_mass,
                full_gimbal_mass_kg=link_map['M_link1']['mass_kg']+link_map['M_link2']['mass_kg'],
                root_alignment='control=Rz(+90deg)*full; geometric comparison only, not a calibration',
                full_zero_pose_com_aligned=(ALIGN@full['zero_pose_com']).tolist())


def plot(models, frames, meshes, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig, axes=plt.subplots(1,3,figsize=(16,6.2))
    fig.subplots_adjust(left=.055,right=.985,bottom=.16,top=.84,wspace=.25)
    colors={'leg':'#2765b0','transmission':'#dc8c18','spring':'#3b9769','gimbal':'#a255a7'}
    for i,key in enumerate(['full','v7']):
        ax=axes[i]; data=models[key]; f=frames[key]; a=ALIGN if key=='full' else np.eye(3)
        if key=='full':
            for n in ['base_link','M_link1','M_link2','L_link1','L_link2','L_link3','LL_link1','LL_link2','LL_link3','LL_link4']:
                mesh=meshes[n]; vs=(a@trimesh.transform_points(mesh.vertices,f[n]).T).T
                # Deterministic subsampling makes the silhouette readable without changing data.
                step=max(1,len(mesh.faces)//6000)
                triangles=vs[mesh.faces[::step]][:,:,[0,2]]
                ax.add_collection(PolyCollection(triangles,facecolors='#99a4b0',edgecolors='none',alpha=.045))
        for j in data['joints']:
            n=j['child']
            if n.startswith(('R','RR')):
                continue
            parent=j['parent'];p=a@f[parent][:3,3];q=a@f[n][:3,3]
            kind='gimbal' if n.startswith('M') else 'transmission' if n.startswith('LL_') else 'spring' if n.startswith('LLL_') else 'leg'
            ax.plot([p[0],q[0]],[p[2],q[2]],'-o',color=colors[kind],lw=2,ms=4)
        if key=='v7':
            cs=json.loads((ROOT/'model/纯底盘_v7_flat_base107/urdf/constraints.json').read_text())['constraints']
            for c in cs:
                if not c['name'].startswith('L_'): continue
                p=point(f[c['body0']],c['local_pos0_m']);q=point(f[c['body1']],c['local_pos1_m'])
                ax.plot([p[0],q[0]],[p[2],q[2]],'r--',lw=3)
                ax.scatter([p[0]],[p[2]],facecolors='none',edgecolors='red',s=80,zorder=5)
        ax.set_title('Full vehicle v2.0.0: URDF tree' if key=='full' else 'Chassis v7: tree + external closures')
    ax=axes[2]
    origins={b['name']:np.array(b['world_origin']) for b in reference['bodies']}
    for j in reference['joints']:
        n=j['body1'][0].split('/')[-1];pname=j['body0'][0].split('/')[-1]
        if not (n.startswith('left_') or n.startswith('gimbal')) or 'guide' in n: continue
        if '/loop_joints/' in j['path']:
            p=np.array(j['world_pos0']);q=np.array(j['world_pos1']);ax.plot([p[0],q[0]],[p[2],q[2]],'r--',lw=3)
            ax.scatter([p[0]],[p[2]],facecolors='none',edgecolors='red',s=80,zorder=5); continue
        p,q=origins[pname],origins[n]
        kind='gimbal' if 'gimbal' in n else 'transmission' if 'front' in n else 'spring' if 'spring' in n else 'leg'
        ax.plot([p[0],q[0]],[p[2],q[2]],'-o',color=colors[kind],lw=2,ms=4)
    ax.set_title('wheelbipeV14_2_1: authored USD closures')
    for ax in axes:
        ax.set_aspect('equal');ax.set_xlim(-.3,.3);ax.set_ylim(-.25,.4)
        ax.set_xlabel('forward X (m)');ax.set_ylabel('up Z (m)');ax.grid(alpha=.2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color=color,lw=2,label=name) for name,color in colors.items()]
               +[Line2D([0],[0],color='red',marker='o',markerfacecolor='none',linestyle='--',label='closure anchor')],
               loc='lower center',bbox_to_anchor=(.5,.01),ncols=5)
    fig.suptitle('Left leg + gimbal, exported zero poses (different zero conventions; no dynamics)',fontsize=14,y=.97)
    fig.savefig(OUT/'geometry_comparison.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    paths={'full':FULL, 'v5_source':ROOT/'model/纯底盘_v5/source/urdf/urdf_v5.0.urdf',
           'v6_source':ROOT/'model/纯底盘_v6_232mm/source/urdf/urdf_v5.0.urdf'}
    for folder in sorted((ROOT/'model').glob('纯底盘*')):
        key={'纯底盘_v6_232mm':'v6','纯底盘_v7_flat_base107':'v7'}.get(folder.name,folder.name)
        paths[key]=folder/'urdf/robot.urdf'
    models,frames={},{}
    for key,path in paths.items():
        if path.exists():
            models[key],frames[key],_=parse(path)
    _,_,links=parse(FULL)
    records,meshes=full_mesh_audit(frames['full'],links)
    reference=usd_audit()
    data=dict(scope='Static XML, mesh, FK, inertia and composed USD inspection; zero physics/PPO/hardware steps',
              models=models, full_meshes=records, reference_usd=reference,
              comparison=comparison(models,frames))
    (OUT/'audit.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    plot(models,frames,meshes,reference)
    print('Wrote',OUT/'audit.json',flush=True)

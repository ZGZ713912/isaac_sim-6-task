"""CPU-only geometry oracle: declared mechanism FK versus imported frames."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
p.add_argument('--output',type=Path,required=True)
p.add_argument('--snapshot',type=Path)
a=p.parse_args()
sys.path.insert(0,str(a.root/'src'))
wp.init(); wp.set_device('cpu')
from wheeled_tasks.chassis.kamino_closed_chain_model import import_robot, _transform
directory=a.root/'model/wheel_leg_v2.0.2/friction_v1/training'
spec=json.loads((directory/'model_spec.json').read_text())
manifest=json.loads((directory/'manifest.json').read_text())
builder,corrections,audit=import_robot(directory)
names=[n.rsplit('/',1)[-1] for n in builder.body_label]
rng=np.random.default_rng(106)
position_errors=[]; rotation_errors=[]
for sample in range(16):
    joints={n:q+rng.uniform(-.05,.05) for n,q in manifest['nominal_joint_pos'].items()}
    root_pose=np.eye(4)
    root_pose[:3,:3]=Rotation.from_euler('xyz',rng.uniform(-1.,1.,3)).as_matrix()
    root_pose[:3,3]=rng.uniform(-.3,.3,3)
    declared={'base_link':root_pose}
    for joint in spec['joints']:
        if joint['type']=='fixed': motion=np.eye(4)
        else:
            motion=np.eye(4); axis=np.asarray(joint['axis'])
            if joint['type']=='prismatic': motion[:3,3]=axis*joints[joint['name']]
            else: motion[:3,:3]=Rotation.from_rotvec(axis*joints[joint['name']]).as_matrix()
        declared[joint['child']]=declared[joint['parent']]@np.asarray(joint['origin'])@motion
    imported={names.index('base_link'):root_pose}
    for j,typ in enumerate(builder.joint_type):
        if int(typ) not in (0,1): continue
        motion=np.eye(4)
        axis=np.asarray(builder.joint_axis[builder.joint_qd_start[j]])
        name=builder.joint_label[j].rsplit('/',1)[-1]
        if int(typ)==0: motion[:3,3]=axis*joints[name]
        else: motion[:3,:3]=Rotation.from_rotvec(axis*joints[name]).as_matrix()
        parent,child=builder.joint_parent[j],builder.joint_child[j]
        imported[child]=imported[parent]@_transform(builder.joint_X_p[j])@motion@np.linalg.inv(_transform(builder.joint_X_c[j]))
    for i,name in enumerate(names):
        restored=imported[i].copy()
        restored[:3,:3]=restored[:3,:3]@corrections[i].T
        position_errors.append(float(np.max(np.abs(restored[:3,3]-declared[name][:3,3]))))
        rotation_errors.append(float(np.max(np.abs(restored[:3,:3]-declared[name][:3,:3]))))
result=dict(schema='kamino-frame-oracle-v1',device='cpu',poses=16,bodies_per_pose=len(names),
    max_body_position_error_m=max(position_errors),max_rotation_matrix_error=max(rotation_errors),
    frame_preservation=audit,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
result['passed']=max(position_errors)<1e-5 and max(rotation_errors)<1e-5
if a.snapshot is not None:
    snapshot=json.loads((a.snapshot/'snapshot.json').read_text())
    state=np.load(a.snapshot/'initial.npz')
    native_positions=[]; native_rotations=[]
    for env in range(len(state['joint'])):
        joints=dict(zip(snapshot['joint_names'],state['joint'][env]))
        base=state['base'][env]
        root_pose=np.eye(4);root_pose[:3,:3]=Rotation.from_quat(base[3:]).as_matrix();root_pose[:3,3]=base[:3]
        declared={'base_link':root_pose}
        for joint in spec['joints']:
            motion=np.eye(4);axis=np.asarray(joint.get('axis',[0.,0.,1.]))
            if joint['type']=='prismatic': motion[:3,3]=axis*joints[joint['name']]
            elif joint['type']!='fixed': motion[:3,:3]=Rotation.from_rotvec(axis*joints[joint['name']]).as_matrix()
            declared[joint['child']]=declared[joint['parent']]@np.asarray(joint['origin'])@motion
        for b,name in enumerate(snapshot['body_names']):
            body=state['body'][env,b]
            native_positions.append(float(np.max(np.abs(declared[name][:3,3]-body[:3]))))
            native_rotations.append(float(np.max(np.abs(declared[name][:3,:3]-Rotation.from_quat(body[3:]).as_matrix()))))
    result['native_snapshot']=dict(worlds=len(state['joint']),body_position_max_error_m=max(native_positions),
        body_rotation_matrix_max_error=max(native_rotations))
    result['passed']=result['passed'] and max(native_positions)<2e-5 and max(native_rotations)<2e-5
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
if not result['passed']: raise SystemExit(1)

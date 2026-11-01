#!/usr/bin/env python3
"""Freeze policy-independent passive admission and fresh train/holdout splits."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source,evidence,output):
    source,evidence,output=source.resolve(),evidence.resolve(),output.resolve()
    bank=json.loads(source.read_text());probe=json.loads(evidence.read_text())
    if (probe['status']!='completed' or probe['new_PPO']!=0 or probe['checkpoint_reads']
            or probe['bank_sha256']!=digest(source) or probe['height_shift_m']!=0.
            or probe['seconds']<.15 or probe['physics_hz']!=200. or probe['policy_hz']!=100.):
        raise ValueError('Need complete original-timing passive evidence with no actor or weights')
    by_pose={}
    for row in probe['cases']:by_pose.setdefault(row['pose_index'],[]).append(row)
    allowed={i for i,rows in by_pose.items() if {r['heading_rad'] for r in rows}=={0.,1.5707963267948966}
        and all(r['passed'] and r['closure_peak_m']<=.0028 and r['passive_motor_peak_nm']==0 for r in rows)}
    groups={}
    for i,p in enumerate(bank['poses']):
        if i in allowed:
            r=p['recovery'];groups.setdefault((r['family'],r['leg_class']),[]).append(i)
    if len(groups)!=28:raise ValueError('Every body-family/leg-class cell must remain available')
    # No policy was trained on this successor bank. Preserve safe old holdouts
    # first; allocate missing static holdouts before this fresh run begins.
    heldout=set()
    for key,ids in groups.items():
        static=[i for i in ids if not bank['poses'][i]['recovery']['dynamic']]
        static.sort(key=lambda i:(bank['poses'][i]['recovery']['split']!='holdout',i))
        if len(static)<3:raise ValueError(f'Need >=2 static holdouts and training coverage: {key}')
        heldout.update(static[:2])
    signature=lambda p:json.dumps([p['root_position_m'],p['root_quaternion_xyzw'],p['full_joint_positions']],sort_keys=True)
    hold_signatures={signature(bank['poses'][i]) for i in heldout}
    # A velocity variant of a held-out geometry must not leak into training.
    allowed={i for i in allowed if i in heldout or signature(bank['poses'][i]) not in hold_signatures}
    result=deepcopy(bank);result['poses']=[];rows=[];excluded=[]
    for i,p in enumerate(bank['poses']):
        if p.get('recovery') and i not in allowed:
            excluded.append(dict(source_pose_index=i,name=p['name'],reason='native_passive_gap_or_holdout_geometry_duplicate'))
            continue
        p=deepcopy(p)
        if p.get('recovery'):
            p['recovery']['split']='holdout' if i in heldout else 'train'
            p['recovery']['native_passive_admitted']=True
            p['recovery']['source_pose_index']=i
            rows.extend(dict(r,pose_index=len(result['poses']),source_pose_index=i) for r in by_pose[i])
        result['poses'].append(p)
    declaration=dict(schema='native-passive-reset-admission-v1',new_PPO=0,checkpoint_reads=[],
        source_bank_sha256=digest(source),source_evidence_sha256=digest(evidence),
        physics_hz=200,policy_hz=100,seconds=.15,guard_m=.003,admission_margin_m=.0002,
        cases=rows,excluded=excluded,all_pass=True,
        split_policy='partition_frozen_before_fresh_PPO; static_and_dynamic_geometry_disjoint',
        scope='nominal_passive_feasibility_only_not_recovery_or_hardware_qualification')
    output.parent.mkdir(parents=True,exist_ok=True)
    admission=output.parent/'native_admission.json'
    admission.write_text(json.dumps(declaration,indent=2)+'\n')
    result['multi_pose_recovery']['native_reset_admission']=dict(file=str(admission.relative_to(ROOT)),sha256=digest(admission))
    result['multi_pose_recovery']['admission']='original_CAD_geometry_plus_native_passive_two_heading_200Hz'
    result['multi_pose_recovery']['source_pose_bank_sha256']=digest(source)
    output.write_text(json.dumps(result,indent=2)+'\n')
    receipt=dict(schema='native-admitted-recovery-bank-build-v1',bank_sha256=digest(output),
        builder_sha256=digest(Path(__file__)),admission_sha256=digest(admission),
        poses=len(result['poses']),excluded=excluded,new_PPO=0,behavior_qualified=False)
    output.with_suffix('.build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(poses=len(result['poses']),heldout=len(heldout),excluded=len(excluded))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'model/recovery_v202_20261031/pose_bank.json')
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; preserve prior bank')
    build(a.source,a.evidence,a.output)

#!/usr/bin/env python3
"""Select the unchanged standard prone posture, using zero-torque native evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from build_multi_pose_recovery_bank import orbital_angles
from wheeled_tasks.chassis.multi_pose_recovery import LEG_CLASSES,leg_class


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def build(evidence_path,output):
    original=ROOT/'model/recovery_v202_20261031/pose_bank.json'
    source=json.loads(original.read_text());native=json.loads(evidence_path.read_text())
    if (native.get('status') != 'completed' or native.get('new_PPO') != 0
            or native.get('checkpoint_reads') != [] or native.get('height_shift_m') != 0.
            or native.get('bank_sha256') != digest(original) or native.get('seconds',0) < .15
            or native.get('physics_hz') != 200 or native.get('policy_hz') != 100):
        raise ValueError('Need unchanged, original-rate passive placement evidence')
    bank=deepcopy(source);bank['poses']=deepcopy(source['poses'][:4]);pose=bank['poses'][3]
    if pose['name'] != 'floor_knee103_103_orbit1.6_r0_p0':raise ValueError('Standard pose changed')
    rows=[r for r in native['cases'] if r['pose_index']==3]
    if len(rows)!=2 or {r['heading_rad'] for r in rows}!={0.,__import__('math').pi/2} or any(
            not r['passed'] or r['terminated'] or r['mechanical'] or r['closure_peak_m']>.0028
            or r['passive_motor_peak_nm']!=0. for r in rows):
        raise ValueError('The standard posture has not passed both passive native trials')
    spec=json.loads((ROOT/'model/wheel_leg_v2.0.2/friction_v1/training/model_spec.json').read_text())
    orbits=orbital_angles(spec,pose['full_joint_positions']);cls=LEG_CLASSES[leg_class(orbits)]
    pose['admitted_as_training_reset']=True
    pose['recovery']=dict(family='belly_down',leg_class=cls,orbital_pair_deg=orbits,split='standard',
        dynamic=False,cad_surface_screen_passed=True,native_passive_admitted=True,
        scope='single_original_standard_prone_posture; no unseen-pose qualification')
    proof=dict(schema='native-passive-reset-admission-v1',new_PPO=0,checkpoint_reads=[],all_pass=True,
        seconds=native['seconds'],guard_m=.003,physics_hz=200,policy_hz=100,
        source_evidence_sha256=digest(evidence_path),source_bank_sha256=digest(original),
        physical_pose_unchanged=True,cases=rows,hardware_ready=False)
    save(output/'native_admission.json',proof)
    bank['multi_pose_recovery'].update(reset_scope='standard_prone_only',
        native_reset_admission=dict(file=str((output/'native_admission.json').relative_to(ROOT)),
            sha256=digest(output/'native_admission.json')),
        admission='one unchanged standard belly posture passed200Hz zero-torque native settling',
        excluded_complex_pose_count=len(source['poses'])-4)
    save(output/'pose_bank.json',bank)
    save(output/'pose_bank.build.json',dict(schema='standard-prone-native-bank-build-v1',
        builder_sha256=digest(Path(__file__)),source_bank_sha256=digest(original),
        source_evidence_sha256=digest(evidence_path),bank_sha256=digest(output/'pose_bank.json'),
        reset_scope='standard_prone_only',recovery_pose_count=1,other_pose_count=3,
        required_leg_class=cls,geometry_changes=False,new_PPO=0,behavior_qualified=False))
    return cls


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(build(a.evidence.resolve(),a.output.resolve()))

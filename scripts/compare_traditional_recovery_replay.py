#!/usr/bin/env python3
"""Compare complete recorded recovery states across source or fixture revisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simulate_traditional_self_righting import digest,save


def same_bits(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(
        np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8))


def load_run(path):
    receipt_path=path/'receipt.json';trace_path=path/'trace.npz'
    receipt=json.loads(receipt_path.read_text())
    if receipt['status']!='completed' or digest(trace_path)!=receipt['trace_sha256']:
        raise ValueError('Need final completed receipts and unchanged native traces')
    snapshots={name:bool((path/'sources'/Path(name).name).exists()
        and digest(path/'sources'/Path(name).name)==sha)for name,sha in receipt['sources'].items()}
    if not all(snapshots.values()):
        raise ValueError('A native source snapshot is missing or changed')
    return receipt,np.load(trace_path),dict(receipt_sha256=digest(receipt_path),
        trace_sha256=digest(trace_path),source_snapshots_valid=snapshots)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',type=Path,required=True)
    parser.add_argument('--new',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous replay comparisons; select a new output')
    old,a,old_proof=load_run(args.old);new,b,new_proof=load_run(args.new)
    for key in ('model_spec_sha256','asset_manifest_sha256','native_body_names','native_joint_names','trace_layout'):
        if old[key]!=new[key]:raise ValueError('Different replay contract: '+key)
    times_equal=same_bits(a['body_time_s'],b['body_time_s'])and same_bits(a['dt_s'],b['dt_s'])
    rows=[]
    for name in args.cases:
        i=next(i for i,r in enumerate(old['records'])if r['name']==name)
        j=next(i for i,r in enumerate(new['records'])if r['name']==name)
        row=dict(name=name,body_time_bitwise_equal=bool(times_equal))
        for key,label in [('body_poses','body_poses'),('values','joint_control_trace')]:
            x=a[key][:,i];y=b[key][:,j]
            same_shape=x.shape==y.shape
            row[label+'_bitwise_equal']=bool(same_bits(x,y))
            row[label+'_max_abs_difference']=float(np.abs(x-y).max())if same_shape else None
        row['equal']=all(row[k]for k in ('body_time_bitwise_equal','body_poses_bitwise_equal','joint_control_trace_bitwise_equal'))
        rows.append(row)
    result=dict(schema='traditional-recovery-source-replay/v1',source_sha256=digest(__file__),
        old_native_run=str(args.old),new_native_run=str(args.new),old=old_proof,new=new_proof,records=rows,
        all_requested_replays_equal=all(row['equal']for row in rows),
        scope='Whole recorded primal body-pose and controller replay. A finite CAD screen can transfer only '
              'at identical poses, times and CAD geometry. No broader contact, controller or hardware qualification.')
    args.output.parent.mkdir(parents=True,exist_ok=True);save(args.output,result)
    print(json.dumps(dict(equal=result['all_requested_replays_equal'],records=rows)))
    if not result['all_requested_replays_equal']:raise SystemExit(1)


if __name__=='__main__':main()

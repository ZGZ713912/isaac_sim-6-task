#!/usr/bin/env python3
"""Archive recovery receipts and source snapshots without deleting native evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from simulate_traditional_self_righting import ROOT,digest,save


def compress_copy(source,target):
    data=source.read_bytes();target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('wb')as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0)as archive:
            archive.write(data)
    restored=gzip.decompress(target.read_bytes())
    if hashlib.sha256(restored).hexdigest()!=digest(source):
        raise ValueError('Compressed evidence verification failed: '+str(source))
    return dict(source=str(source),source_sha256=digest(source),source_bytes=len(data),
        archived=str(target),archived_sha256=digest(target),gzip_restore_verified=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--first-native',type=int,default=39)
    parser.add_argument('--log-directory',type=Path,default=Path('/tmp'))
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous archives; select a new directory')
    args.output.mkdir(parents=True)
    shutil.copy2(__file__,args.output/Path(__file__).name)
    runs=[]
    for receipt_path in sorted(args.root.glob('native*/receipt.json')):
        run=receipt_path.parent
        if int(run.name.removeprefix('native'))<args.first_native:continue
        receipt=json.loads(receipt_path.read_text())
        if receipt['status']=='running'and not(run/'closure.json').exists():
            raise ValueError('Native run has no final receipt or closure: '+str(run))
        metadata=[]
        paths=set(run.glob('*.json'))|set(run.glob('*.jsonl'))
        for path in sorted(paths):
            metadata.append(compress_copy(path,args.output/run.name/(path.name+'.gz')))
        log=args.log_directory/('traditional_recovery_'+run.name+'.log')
        if log.exists():metadata.append(compress_copy(log,args.output/run.name/(log.name+'.gz')))
        snapshots=[]
        for name,expected in receipt['sources'].items():
            source=run/'sources'/Path(name).name
            exists=source.is_file();valid=exists and digest(source)==expected
            row=dict(original_path=name,source_sha256=expected,snapshot_exists=exists,hash_valid=valid)
            if valid:
                target=args.output/'source_objects'/expected/Path(name).name
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():shutil.copy2(source,target)
                if digest(target)!=expected:raise ValueError('Archived source hash differs')
                row['archived']=str(target)
            snapshots.append(row)
        external=[]
        for path in sorted([*run.glob('*.npz'),*run.glob('*.so'),*run.glob('events/*')]):
            if not path.is_file():continue
            sha=digest(path)
            if path.name=='trace.npz'and receipt.get('trace_sha256')and sha!=receipt['trace_sha256']:
                raise ValueError('Native trace changed before archival: '+str(path))
            external.append(dict(path=str(path),sha256=sha,bytes=path.stat().st_size,
                retained_in_original_location=True,copied_into_archive=False))
        runs.append(dict(run=run.name,status=receipt['status'],metadata=metadata,sources=snapshots,
            external_artifacts=external))
        if len(runs)%10==0:print(json.dumps(dict(archived_runs=len(runs))),flush=True)
    reviews=[]
    for directory in sorted(args.root.glob('review*')):
        if not directory.is_dir():continue
        metadata=[compress_copy(path,args.output/directory.name/(path.name+'.gz'))
            for path in sorted(directory.glob('*.json'))]
        snapshots=[]
        for source in sorted((directory/'sources').glob('*')):
            if not source.is_file():continue
            sha=digest(source);target=args.output/'source_objects'/sha/source.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(source,target)
            if digest(target)!=sha:raise ValueError('Archived review source hash differs')
            snapshots.append(dict(source=str(source),sha256=sha,archived=str(target)))
        reviews.append(dict(review=directory.name,final_review_recorded=(directory/'review.json').exists(),
            metadata=metadata,sources=snapshots,
            figures=[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size,
                retained_in_original_location=True)for p in sorted(directory.glob('*.png'))]))
    for source in sorted((args.root/'analysis_source_objects').glob('*/*')):
        if not source.is_file():continue
        sha=digest(source)
        if sha!=source.parent.name:raise ValueError('Analysis source object hash differs')
        target=args.output/'source_objects'/sha/source.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(source,target)
        if digest(target)!=sha:raise ValueError('Archived analysis source hash differs')
    result=dict(schema='traditional-recovery-evidence-archive/v1',
        source_sha256=digest(args.output/Path(__file__).name),root=str(args.root),runs=runs,
        reviews=reviews,
        originals_deleted=0,originals_modified=0,
        scope='Verified gzip copies of metadata and content-addressed source snapshots. Large traces, '
              'native shared libraries and TensorBoard events remain at their original paths with SHA-256 '
              'bindings. This archive is not a self-contained simulator installation or behavioral qualification.')
    save(args.output/'manifest.json',result)
    print(json.dumps(dict(archived_runs=len(runs),manifest=str(args.output/'manifest.json'))))


if __name__=='__main__':main()

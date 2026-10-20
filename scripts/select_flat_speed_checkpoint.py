#!/usr/bin/env python3
"""Apply the frozen bidirectional speed ranking to completed paired evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analysis=json.loads(args.analysis.read_text());protocol=json.loads(args.protocol.read_text())
    if analysis['seed']!=protocol['seed']or protocol['status']!='frozen_before_heldout':
        raise ValueError('Use the predeclared held-out comparison')
    rows=[]
    for arm,identity in analysis['arms'].items():
        role=identity['role']
        if role=='parent':continue
        low_loss=[];scores=[];full=requested=0
        for name,condition in analysis['conditions'].items():
            parent,candidate=condition['parent'],condition[role]
            # The analysis counts individual low-case checks independently.
            # This list is recomputed against the original native result below.
            high=next((n for n in protocol['high_cases']if name.endswith('_'+n)),None)
            if high:
                full+=candidate['full_episodes'];requested+=candidate['requested_episodes']
                value=candidate['yaw_mae_rad_s']if high.startswith('rotate_')else candidate['vx_mae_m_s']
                scores.append(None if value is None else value/(.1 if high.startswith('rotate_')else .05))
        source=args.analysis.parent/'report.json'
        native=json.loads(source.read_text())['result']['cases']
        for name,p in native.items():
            if not name.startswith('g0_'):continue
            case=next(n for n in analysis['summaries']['parent']['cases']if name.endswith('_'+n))
            if case in protocol['high_cases']:continue
            c=native[f'g{arm}_'+name[3:]]
            low_loss.extend(dict(case=name[3:],check=k)for k,v in p['checks'].items()if v and not c['checks'].get(k,False))
        worst=max(scores)if scores and all(v is not None for v in scores)else None
        rows.append(dict(role=role,arm=arm,checkpoint=identity['path'],checkpoint_sha256=identity['sha256'],
            successful_updates=identity['successful_updates'],actor_updates=identity['actor_updates'],
            lost_low_parent_checks=low_loss,full_high_episodes=full,requested_high_episodes=requested,
            worst_normalized_high_mae=worst))
    best_loss=min(len(r['lost_low_parent_checks'])for r in rows)
    pool=[r for r in rows if len(r['lost_low_parent_checks'])==best_loss]
    best_full=max(r['full_high_episodes']for r in pool)
    pool=[r for r in pool if r['full_high_episodes']==best_full and r['worst_normalized_high_mae']is not None]
    if not pool:raise ValueError('No supported candidate at the best completion level')
    best_score=min(r['worst_normalized_high_mae']for r in pool)
    selected=min((r for r in pool if r['worst_normalized_high_mae']<=best_score*1.02),key=lambda r:r['actor_updates'])
    result=dict(schema='flat-speed-checkpoint-selection-v1',status='candidate_selected_pending_long_confirmation',
        analysis_sha256=digest(args.analysis),protocol_sha256=digest(args.protocol),selector_sha256=digest(Path(__file__)),
        candidates=rows,selected=selected,qualified=False,new_optimizer_updates=0)
    if args.output.exists():raise ValueError('Preserve the earlier selection')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

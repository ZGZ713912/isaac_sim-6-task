#!/usr/bin/env python3
"""Separate steady body tracking from translation and yaw stopping errors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous phase analysis')
    report_path=args.input/'report.json';report=json.loads(report_path.read_text())
    if report['status']!='evaluated':raise ValueError('Native evaluation must finish first')
    rows={}
    with np.load(args.input/'traces.npz')as data:
        groups=data['groups'];active=data['active'].astype(bool)
        done=data['done'].astype(bool);commands=data['commands']
        velocity=data['velocity'];omega=data['omega'];ticks=data['episode_ticks']
        for index,arm in report['checkpoint_comparison'].items():
            selected={}
            for name in ('forward_3','backward_3','rotate_4','rotate_4_reverse','start_stop_3',
                         'yaw_start_stop_positive','yaw_start_stop_negative'):
                for height in report['heights']:
                    for mu in report['ground_friction']:
                        case=f'g{index}_h{height:g}_mu{mu:g}_{name}'
                        ids=np.flatnonzero(groups==case)
                        if not len(ids):continue
                        yaw=name.startswith('rotate_')or name.startswith('yaw_')
                        actual=omega[:,ids,2]if yaw else velocity[:,ids,0]
                        goal=commands[:,ids,1 if yaw else 0]
                        policy_dt=1/report['policy_hz']
                        elapsed=ticks[:,ids]*policy_dt
                        valid=active[:,ids]&~done[:,ids]
                        # A normal horizon timeout has done=True on its last
                        # frame; it still counts as completing the replay.
                        full=active[-1,ids]&(ticks[-1,ids]>=round(report['horizon_seconds']/policy_dt))
                        def stats(mask):
                            mask=mask&valid;count=int(mask.sum())
                            return dict(frames=count,actual_mean=float(actual[mask].mean())if count else None,
                                target_mean=float(goal[mask].mean())if count else None,
                                mae=float(np.abs(actual[mask]-goal[mask]).mean())if count else None)
                        row=dict(unit='rad/s'if yaw else'm/s',full_episodes=int(full.sum()),
                            requested_episodes=len(ids),steady_after_four_seconds=stats(elapsed>=4.),
                            last_five_seconds_survivors=stats((elapsed>=report['horizon_seconds']-5.)&full[None,:]))
                        if 'start_stop' in name:
                            # Keep transition and settled-stop evidence separate.
                            phase=(elapsed//8.).astype(int)%4
                            settled=(elapsed%8.)>=1.
                            row['phases']={label:stats((phase==i)&settled)for i,label in enumerate(
                                ('first_direction','first_stop','opposite_direction','last_stop'))}
                        selected[f'h{height:g}_mu{mu:g}_{name}']=row
            rows[arm['role']]=selected
    result=dict(schema='flat-speed-body-phase-analysis-v1',qualified=False,new_optimizer_updates=0,
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='MAE is conditional on active nonterminal frames; full episode counts and missing data remain explicit',
        summaries=rows)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(args.input),filename_suffix='.phase_summary')as writer:
        for arm in report['checkpoint_comparison'].values():
            for case,row in rows[arm['role']].items():
                for phase,values in row.get('phases',{}).items():
                    for metric,value in values.items():
                        if value is not None:writer.add_scalar(f'FlatSpeedPhase/{arm["role"]}/{case}/{phase}/{metric}',value,arm['successful_updates'])
    print(json.dumps(dict(status='phase_analysis_complete',arms=list(rows))))


if __name__=='__main__':main()

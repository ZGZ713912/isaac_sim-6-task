#!/usr/bin/env python3
"""Summarize paired speed tests without treating missing frames as good MAE."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-role',default='trained')
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing comparison analysis')
    source=args.input/'report.json'
    report=json.loads(source.read_text())
    if report['status']!='evaluated' or report['probe_mode']!='checkpoint_compare':
        raise ValueError('Require completed parent/initial/trained comparison')
    cases=report['result']['cases']
    contract=json.loads((args.input/'contract.json').read_text())
    names=report.get('case_names')or list(dict.fromkeys(c['original_case']for c in contract['evaluation']['cases']))
    low=tuple(n for n in names if n.startswith('stand_')or n=='height_scan' or n in
        ('forward_05','backward_05','rotate_1','rotate_1_reverse','start_stop_05'))
    high=tuple(n for n in names if n not in low)
    rows={}
    indices=[int(i) for i in report['checkpoint_comparison']]
    for index in indices:
        role=report['checkpoint_comparison'][str(index)]['role']
        all_rows=[v for n,v in cases.items() if n.startswith(f'g{index}_')]
        summary=dict(full_episodes=sum(v['full_horizon_episodes']for v in all_rows),
            requested_episodes=sum(v['requested_episodes']for v in all_rows),
            cases_passing_all_checks=sum(v['passed']for v in all_rows),total_cases=len(all_rows),cases={})
        for name in low+high:
            group=[v for n,v in cases.items() if n.startswith(f'g{index}_')and n.endswith('_'+name)]
            supported=all(v['frames']>0 for v in group)
            summary['cases'][name]=dict(full_episodes=sum(v['full_horizon_episodes']for v in group),
                requested_episodes=sum(v['requested_episodes']for v in group),
                mean_case_vx_mae_m_s=sum(v['vx_mae_m_s']for v in group)/len(group) if supported else None,
                mean_case_yaw_mae_rad_s=sum(v['yaw_mae_rad_s']for v in group)/len(group) if supported else None,
                no_post_warmup_cases=sum(v['frames']==0 for v in group),
                worst_tilt_deg=max(v['tilt_max_deg']for v in group))
        rows[role]=summary
    conditions={};lost_low=[];lost_complete=[]
    for height in report['heights']:
        for mu in report['ground_friction']:
            for name in low+high:
                suffix=f'h{height:g}_mu{mu:g}_{name}'
                entries={report['checkpoint_comparison'][str(i)]['role']:cases[f'g{i}_{suffix}']for i in indices}
                conditions[suffix]={role:dict(full_episodes=v['full_horizon_episodes'],requested_episodes=v['requested_episodes'],
                    frames=v['frames'],vx_mae_m_s=v['vx_mae_m_s']if v['frames'] else None,
                    yaw_mae_rad_s=v['yaw_mae_rad_s']if v['frames'] else None,
                    height_mae_m=v['height_mae_m']if v['frames'] else None,
                    tilt_max_deg=v['tilt_max_deg'],closure_gap_max_m=v['closure_gap_max_m'],
                    termination_reasons=v['termination_reasons'])for role,v in entries.items()}
                parent,trained=entries['parent'],entries[args.target_role]
                if name in low:
                    lost_low += [dict(case=suffix,check=k)for k,v in parent['checks'].items()if v and not trained['checks'].get(k,False)]
                if parent['full_horizon_episodes']==parent['requested_episodes'] and trained['full_horizon_episodes']<trained['requested_episodes']:
                    lost_complete.append(suffix)
    result=dict(schema='flat-speed-checkpoint-comparison-analysis-v1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        qualified=False,full_training_admitted=False,new_optimizer_updates=0,
        scoring='True external command and measured body velocity; unsupported MAE is null; survival is separate',
        seed=report.get('seed'),target_role=args.target_role,arms=report['checkpoint_comparison'],summaries=rows,conditions=conditions,
        lost_low_parent_checks=lost_low,lost_parent_complete_cases=lost_complete)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(args.input),filename_suffix='.comparison_summary')as writer:
        for index,arm in report['checkpoint_comparison'].items():
            role=arm['role'];step=arm['successful_updates'];s=rows[role]
            writer.add_scalar('FlatSpeedComparison/'+role+'/full_episodes',s['full_episodes'],step)
            writer.add_scalar('FlatSpeedComparison/'+role+'/requested_episodes',s['requested_episodes'],step)
            for name,row in s['cases'].items():
                for metric,value in row.items():
                    if value is not None:
                        writer.add_scalar('FlatSpeedComparison/'+role+'/'+name+'/'+metric,value,step)
    print(json.dumps(dict(summaries=rows,lost_low_parent_checks=lost_low,lost_parent_complete_cases=lost_complete)))


if __name__=='__main__':
    main()

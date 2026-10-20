#!/usr/bin/env python3
"""Keep completed-episode denominators alongside prototype tracking errors."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args()
    source=args.input/'report.json';target=args.input/'analysis.json'
    if target.exists():
        raise ValueError('Preserve existing probe analysis')
    report=json.loads(source.read_text())
    if report['status']!='evaluated' or report['probe_mode']=='checkpoint_compare':
        raise ValueError('Require completed zero-PPO reference probe')
    contract=json.loads((args.input/'contract.json').read_text())
    catalogue={c['name']:c['original_case'] for c in contract['evaluation']['cases']}
    names=list(dict.fromkeys(catalogue.values()))
    arms={}
    for gain in report['gains']:
        prefix=f'g{gain:g}_'
        rows={n:v for n,v in report['result']['cases'].items() if n.startswith(prefix)}
        arm=dict(full_episodes=sum(v['full_horizon_episodes']for v in rows.values()),
            requested_episodes=sum(v['requested_episodes']for v in rows.values()),
            passing_case_checks=sum(v['passed']for v in rows.values()),cases={})
        for name in names:
            group=[v for n,v in rows.items() if catalogue[n]==name]
            supported=all(v['frames'] for v in group)
            arm['cases'][name]=dict(full_episodes=sum(v['full_horizon_episodes']for v in group),
                requested_episodes=sum(v['requested_episodes']for v in group),
                vx_mae_m_s=sum(v['vx_mae_m_s']for v in group)/len(group) if supported else None,
                yaw_mae_rad_s=sum(v['yaw_mae_rad_s']for v in group)/len(group) if supported else None,
                unsupported_conditions=sum(not v['frames'] for v in group))
        arms[f'{gain:g}']=arm
    result=dict(schema='flat-speed-prototype-analysis-v1',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        contract_sha256=hashlib.sha256((args.input/'contract.json').read_bytes()).hexdigest(),
        qualified=False,adopted=False,new_PPO_updates=0,mode=report['probe_mode'],
        seed=contract['evaluation']['seed'],arms=arms,
        scoring='Measured body velocity against external request; unsupported errors are null')
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(args.input),filename_suffix='.prototype_summary') as writer:
        for gain,arm in arms.items():
            prefix=f'FlatSpeedPrototype/{report["probe_mode"]}/arm_{gain}/'
            writer.add_scalar(prefix+'full_episodes',arm['full_episodes'],1338)
            writer.add_scalar(prefix+'requested_episodes',arm['requested_episodes'],1338)
            for name,row in arm['cases'].items():
                for metric,value in row.items():
                    if value is not None:
                        writer.add_scalar(prefix+name+'/'+metric,value,1338)
    print(json.dumps(result))


if __name__=='__main__':
    main()

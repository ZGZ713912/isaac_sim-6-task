#!/usr/bin/env python3
"""Authenticate unchanged V4 state for learning its three existing quiet output layers."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    import torch
    from export_robust_getup import digest,load_mean
    from train_robust_getup import write_json
    from wheeled_algo.quiet_output_checkpoint import MODULE,initialize_state,optimizer_actor_names,validate_state
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise ValueError('Never overwrite an initialization or receipt')
    mean,_=load_mean(args.source)
    source=torch.load(args.source,map_location='cpu',weights_only=False)
    names=optimizer_actor_names(mean)
    state=initialize_state(args.source,source,digest(ROOT/MODULE),names)
    validate_state(state,source,source,names)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(state,args.output)
    load_mean(args.output)
    write_json(args.output.with_suffix('.json'),dict(schema='quiet-output-optimizer-initialization-v1',
        checkpoint=str(args.output),sha256=digest(args.output),source_checkpoint=str(args.source),source_sha256=digest(args.source),
        learning_clock=state['infos']['completed_updates'],all_saved_payload_exact=True,
        optimizer_actor_parameter_names=names,trainable_parameters=1170,new_PPO=0,qualification=None,hardware_started=False))
    print(args.output)


if __name__=='__main__':
    main()

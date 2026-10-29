#!/usr/bin/env python3
"""Create one zero-update motion smoothing counterfactual; never train/export."""
from copy import deepcopy
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    import torch
    from export_robust_getup import digest,load_mean
    from train_robust_getup import write_json
    from wheeled_algo.motion_target_damping import VERSION,MODULE,validate_motion_damping_checkpoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--alpha',type=float,required=True)
    args = parser.parse_args()
    load_mean(args.source)
    source = torch.load(args.source,map_location='cpu',weights_only=False)
    if args.output.exists():
        raise ValueError('Never overwrite a checkpoint')
    state = deepcopy(source)
    info = state['infos']
    info['settings']['motion_leg_damping_version'] = VERSION
    info['settings']['motion_leg_damping_alpha'] = args.alpha
    info['source_sha256'][MODULE] = digest(ROOT/MODULE)
    info['motion_leg_damping_migration'] = dict(schema='motion-leg-damping-migration-v1',
        source_checkpoint=str(args.source.resolve()),source_sha256=digest(args.source),
        learning_clock=info['completed_updates'],new_PPO=0,
        scope='Same weights/optimizer/RNG/clock and original mean; only active upright motion leg smoothing, previous applied targets from existing35D')
    info['hardware_ready'] = False
    validate_motion_damping_checkpoint(state,source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(state,args.output)
    load_mean(args.output)
    write_json(args.output.with_suffix('.json'),dict(schema='motion-leg-damping-checkpoint-v1',
        checkpoint=str(args.output),sha256=digest(args.output),source_sha256=digest(args.source),
        alpha=args.alpha,all_weights_optimizer_rng_clock_exact=True,new_PPO=0,
        learning_clock=info['completed_updates'],qualification=None,hardware_started=False))
    print(args.output)


if __name__=='__main__':
    main()

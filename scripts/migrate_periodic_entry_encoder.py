#!/usr/bin/env python3
"""Create one explicit zero-PPO periodic entry function probe."""
from copy import deepcopy
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]


def main():
    import torch
    from export_robust_getup import digest, load_mean
    from train_robust_getup import write_json
    from wheeled_algo.periodic_entry_actor import VERSION, validate_periodic_entry_checkpoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    load_mean(args.source)
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    if (source['infos']['settings'].get('gap_entry_recovery_split_version') != 'frozen-v1'
            or source['infos']['settings'].get('gap_entry_encoder_version') is not None):
        raise ValueError('Require an authenticated original split-entry checkpoint')
    if args.output.exists():
        raise ValueError('Never overwrite a checkpoint')
    state = deepcopy(source)
    info = state['infos']
    info['settings']['gap_entry_encoder_version'] = VERSION
    info['source_sha256']['src/wheeled_algo/periodic_entry_actor.py'] = digest(ROOT/'src/wheeled_algo/periodic_entry_actor.py')
    info['entry_encoder_migration'] = dict(schema='periodic-entry-migration-v1',
        source_checkpoint=str(args.source.resolve()), source_sha256=digest(args.source),
        learning_clock=info['completed_updates'], new_PPO=0,
        scope='Same weights/optimizer/RNG/clock and existing gates; periodic evaluation of the learned entry head only')
    info['hardware_ready'] = False
    validate_periodic_entry_checkpoint(state, source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    load_mean(args.output)
    write_json(args.output.with_suffix('.json'), dict(schema='periodic-entry-checkpoint-receipt-v1',
        checkpoint=str(args.output), sha256=digest(args.output), source_sha256=digest(args.source),
        function_version=VERSION, all_weights_optimizer_rng_clock_exact=True, new_PPO=0,
        learning_clock=info['completed_updates'], qualification=None, hardware_started=False))
    print(args.output)


if __name__ == '__main__':
    main()

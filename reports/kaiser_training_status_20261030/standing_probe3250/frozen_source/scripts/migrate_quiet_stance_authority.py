#!/usr/bin/env python3
"""Create a zero PPO checkpoint for a declared quiet feedback function change."""
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
    from wheeled_algo.quiet_stance_authority import MODULE, SCHEMA, VERSION, validate_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--retained-authority', type=float, required=True)
    args = parser.parse_args()
    load_mean(args.source)
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise ValueError('Never overwrite a checkpoint or migration receipt')
    state = deepcopy(source)
    info = state['infos']
    info['schema'] = SCHEMA
    info['settings'].update(quiet_stance_authority_version=VERSION,
        quiet_stance_retained_authority=args.retained_authority, quiet_stance_recovery_threshold=.05)
    info['source_sha256'][MODULE] = digest(ROOT/MODULE)
    info['quiet_stance_authority_migration'] = dict(schema=VERSION,
        source_checkpoint=str(args.source.resolve()), source_sha256=digest(args.source),
        learning_clock=info['completed_updates'], new_PPO=0,
        scope='Pure sensor-to-action mean change outside recovery; observer/std and all saved training state exact')
    info['hardware_ready'] = False
    validate_checkpoint(state, source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    load_mean(args.output)
    write_json(args.output.with_suffix('.json'), dict(schema=VERSION,
        checkpoint=str(args.output), sha256=digest(args.output), source_sha256=digest(args.source),
        retained_authority=args.retained_authority, all_weights_optimizer_rng_clock_exact=True,
        new_PPO=0, learning_clock=info['completed_updates'], qualification=None, hardware_started=False))
    print(args.output)


if __name__ == '__main__':
    main()

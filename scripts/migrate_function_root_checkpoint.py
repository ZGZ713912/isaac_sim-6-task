"""Initialize bounded entry PPO without weakening neutral V2 migrations."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def migrate(source, output, experiment_consumed):
    import torch
    from export_robust_getup import load_mean
    from train_robust_getup import write_json
    from wheeled_algo.function_root_checkpoint import MODULE, digest, initialize_state, validate_state
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.parent.exists() or output.exists():
        raise ValueError('Never overwrite or reuse a function-root output directory')
    before, _ = load_mean(source)
    original = torch.load(source, map_location='cpu', weights_only=False)
    result = initialize_state(source, original, digest(ROOT / MODULE))
    validate_state(result, original, original)
    output.parent.mkdir(parents=True, exist_ok=False)
    torch.save(result, output)
    after, info = load_mean(output)
    assert all(torch.equal(value, after.state_dict()[key]) for key, value in before.state_dict().items())
    write_json(output.with_suffix('.json'), dict(schema='entry-function-root-initialization-receipt-v1',
        checkpoint=str(output), sha256=digest(output), root=str(source), root_sha256=digest(source),
        actor_and_critic_exact=True, RNG_exact=True, old_Adam_moments_exact=True,
        appended_empty_Adam_parameter_tensors=6, learning_clock=info['completed_updates'],
        new_PPO=0, actual_experiment_consumed=experiment_consumed,
        qualification=None, hardware_ready=False))
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--experiment-consumed', required=True, type=int)
    args = parser.parse_args()
    print(migrate(args.source, args.output, args.experiment_consumed))

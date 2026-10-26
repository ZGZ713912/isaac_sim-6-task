"""Add authenticated entry-learning metadata to a preserved native checkpoint."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def migrate(source, learning_source, output, experiment_consumed):
    import torch
    from export_robust_getup import load_mean
    from train_robust_getup import digest, write_json
    from wheeled_algo.entry_checkpoint import make_entry_learning_record
    load_mean(learning_source)
    old = torch.load(source, map_location="cpu", weights_only=False)
    parent = torch.load(learning_source, map_location="cpu", weights_only=False)
    if (not old["infos"].get("entry_recovery_only_tuning")
            or old["infos"]["settings"].get("gap_entry_optimizer_layout") != "append-v1"
            or old["infos"].get("entry_recovery_learning") is not None
            or old["infos"].get("migrated_from_sha256") != digest(learning_source)):
        raise ValueError("Require the original entry-only PPO checkpoint and exact restored source")
    if output.exists():
        raise ValueError("Never overwrite a checkpoint")
    state = deepcopy(old)
    info = state["infos"]
    updates = info["completed_updates"] - parent["infos"]["completed_updates"]
    info["entry_recovery_learning"] = make_entry_learning_record(
        learning_source, parent, state["actor_state_dict"], info["completed_updates"], updates)
    module = "src/wheeled_algo/entry_checkpoint.py"
    info["source_sha256"][module] = digest(ROOT / module)
    info["entry_checkpoint_contract_migration"] = dict(
        schema="entry-checkpoint-contract-migration-v1", source=str(source.resolve()),
        source_sha256=digest(source), learning_source_sha256=digest(learning_source),
        metadata_only=True, new_PPO=0)
    output.parent.mkdir(parents=True, exist_ok=False)
    torch.save(state, output)
    mean, _ = load_mean(output)
    write_json(output.with_suffix(".json"), dict(
        schema="entry-checkpoint-contract-receipt-v1", checkpoint=str(output), sha256=digest(output),
        source=str(source), source_sha256=digest(source), learning_source_sha256=digest(learning_source),
        actor_critic_Adam_rng_clock_unchanged=True, new_PPO=0,
        experiment_consumed=experiment_consumed, learning_clock=info["completed_updates"],
        qualification=None, hardware_ready=False))
    return mean


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--learning-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-consumed", type=int, required=True)
    args = parser.parse_args()
    migrate(args.source, args.learning_source, args.output, args.experiment_consumed)
    print(args.output)

"""Make an explicit neutral entry-learning branch with retained Adam history."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def migrate(source, output, experiment_consumed, support_weight):
    import math
    import torch
    from export_robust_getup import load_mean
    from train_robust_getup import digest, write_json
    from wheeled_algo.entry_optimizer import append_entry_optimizer_state
    mean, _ = load_mean(source)
    original = torch.load(source, map_location="cpu", weights_only=False)
    settings = original["infos"]["settings"]
    if settings.get("gap_entry_recovery_split_version") != "frozen-v1" or settings.get("gap_entry_optimizer_layout") is not None:
        raise ValueError("Migrate a neutral copied entry head exactly once")
    if not math.isfinite(support_weight) or not 0. <= support_weight <= 2.:
        raise ValueError("Support weight must be within0..2")
    if output.exists():
        raise ValueError("Never overwrite a checkpoint")
    entry = [p for p in mean.entry_recovery.parameters()]
    state = deepcopy(original)
    state["optimizer_state_dict"] = append_entry_optimizer_state(original["optimizer_state_dict"], len(entry))
    state["infos"]["settings"].update(gap_entry_optimizer_layout="append-v1", recovery_support_weight=support_weight)
    module = "src/wheeled_algo/entry_optimizer.py"
    state["infos"]["source_sha256"][module] = digest(ROOT / module)
    state["infos"]["entry_optimizer_migration"] = dict(
        schema="entry-optimizer-append-v1", source=str(source.resolve()), source_sha256=digest(source),
        new_parameter_tensors=len(entry), new_parameter_count=sum(p.numel() for p in entry),
        optimizer="Original actor/critic slots retain values and steps by group/order; new entry slots absent until first gradient update; canonical serialization IDs updated explicitly",
        actor_critic_rng_clock_unchanged=True, support_weight=support_weight, new_PPO=0)
    state["infos"]["hardware_ready"] = False
    output.parent.mkdir(parents=True, exist_ok=False)
    torch.save(state, output)
    migrated, _ = load_mean(output)
    assert all(torch.equal(v, migrated.state_dict()[k]) for k, v in mean.state_dict().items())
    write_json(output.with_suffix(".json"), dict(
        schema="entry-optimizer-migration-receipt-v1", checkpoint=str(output), sha256=digest(output),
        source_sha256=digest(source), actor_weights_and_buffers_exact=True,
        appended_parameter_tensors=len(entry), empty_new_Adam_slots=True,
        old_slots_preserved_by_group_order=True, support_weight=support_weight,
        new_PPO=0, experiment_consumed=experiment_consumed,
        learning_clock=state["infos"]["completed_updates"], qualification=None, hardware_ready=False))
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-consumed", type=int, required=True)
    parser.add_argument("--recovery-support-weight", type=float, required=True)
    args = parser.parse_args()
    print(migrate(args.source, args.output, args.experiment_consumed, args.recovery_support_weight))

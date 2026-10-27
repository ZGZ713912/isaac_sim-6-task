"""Authenticate explicitly trained entry heads while retaining frozen ancestry."""
import hashlib
import json
from pathlib import Path


PREFIX = "mlp.mean.entry_recovery."
VERSION = "entry-recovery-training-v1"


def entry_digest(actor):
    keys = sorted(k for k in actor if k.startswith(PREFIX))
    expected = sorted(PREFIX + layer + suffix
                      for layer in ("0", "2", "4") for suffix in (".weight", ".bias"))
    if keys != expected:
        raise ValueError("Require the exact six existing entry network tensors")
    result = hashlib.sha256()
    for key in keys:
        value = actor[key].detach().cpu().contiguous()
        result.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def _frozen_actor_exact(before, after):
    import torch
    return (before.keys() == after.keys()
            and all(torch.equal(v.detach().cpu(), after[k].detach().cpu())
                    for k, v in before.items() if not k.startswith(PREFIX)))


def make_entry_learning_record(source, source_state, actor, completed_updates, new_updates):
    """Checkpoint metadata only; never changes weights, RNG or optimizer state."""
    start = source_state["infos"]["completed_updates"]
    if (isinstance(new_updates, bool) or not isinstance(new_updates, int)
            or new_updates < 0 or completed_updates != start + new_updates):
        raise ValueError("Entry learning clocks must preserve the exact update increment")
    if not _frozen_actor_exact(source_state["actor_state_dict"], actor):
        raise ValueError("Entry-only checkpoint changed another actor tensor or std")
    return dict(schema=VERSION, source_checkpoint=str(Path(source).resolve()),
                source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                source_learning_clock=start, completed_learning_clock=completed_updates,
                new_optimizer_updates=new_updates, entry_state_sha256=entry_digest(actor),
                frozen_non_entry_actor_and_std=True,
                scope="Existing copied entry NN learned by native PPO; no new actor inputs or runtime controller")


def validate_entry_learning(state, source_state):
    """Validate a hash-authenticated source, trained head, clocks and frozen scope."""
    info = state["infos"]
    record = info["entry_recovery_learning"]
    settings = info["settings"]
    if (record.get("schema") != VERSION
            or settings.get("gap_entry_recovery_split_version") != "frozen-v1"
            or settings.get("gap_entry_optimizer_layout") != "append-v1"
            or record.get("frozen_non_entry_actor_and_std") is not True):
        raise ValueError("Trained entry requires the explicit optimizer and provenance contract")
    start = source_state["infos"]["completed_updates"]
    count = record.get("new_optimizer_updates")
    end = record.get("completed_learning_clock")
    if (isinstance(count, bool) or not isinstance(count, int) or count < 0
            or record.get("source_learning_clock") != start or end != start + count
            or info["completed_updates"] < end):
        raise ValueError("Trained entry provenance has inconsistent learning clocks")
    if entry_digest(state["actor_state_dict"]) != record.get("entry_state_sha256"):
        raise ValueError("Trained entry tensors differ from their recorded digest")
    if info.get("entry_recovery_only_tuning") and info["completed_updates"] == end:
        if not _frozen_actor_exact(source_state["actor_state_dict"], state["actor_state_dict"]):
            raise ValueError("Entry-only training changed a frozen actor tensor or std")

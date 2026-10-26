"""Explicit optimizer layout for learning the already-copied RL entry head."""
from copy import deepcopy

VERSION = "append-v1"


def append_entry_optimizer_state(optimizer, parameter_count):
    """Retain old moments by group/order and append empty entry Adam slots."""
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count < 1:
        raise ValueError("A positive entry parameter-tensor count is required")
    groups = optimizer["param_groups"]
    if len(groups) != 2 or [g.get("name") for g in groups] != ["actor", "critic"]:
        raise ValueError("Require the authenticated actor/critic optimizer layout")
    old_ids = [p for group in groups for p in group["params"]]
    if len(old_ids) != len(set(old_ids)) or not set(optimizer["state"]) <= set(old_ids):
        raise ValueError("Optimizer parameter IDs must be distinct and cover all state")
    actor_count, critic_count = (len(g["params"]) for g in groups)
    mapping = {key: index for index, key in enumerate(groups[0]["params"])}
    mapping.update({key: actor_count + parameter_count + index
                    for index, key in enumerate(groups[1]["params"])})
    result = deepcopy(optimizer)
    result["state"] = {mapping[key]: deepcopy(value) for key, value in optimizer["state"].items()}
    result["param_groups"][0]["params"] = list(range(actor_count + parameter_count))
    result["param_groups"][1]["params"] = list(range(actor_count + parameter_count,
                                                     actor_count + parameter_count + critic_count))
    assert not set(range(actor_count, actor_count + parameter_count)) & result["state"].keys()
    return result


def prepare_entry_optimizer_parameters(mean, settings):
    """Called before native optimizer creation/loading; runtime mean is unchanged."""
    version = settings.get("gap_entry_optimizer_layout")
    if version is None:
        return
    if version != VERSION or settings.get("gap_entry_recovery_split_version") != "frozen-v1":
        raise ValueError("Entry optimizer layout requires the authenticated copied entry head")
    mean.entry_recovery.requires_grad_(True)


def freeze_except_entry(distribution):
    """Keep old fallen, balance, motion, geometry and exploration unchanged."""
    if not hasattr(distribution.mean, "entry_recovery"):
        raise ValueError("Entry-only learning requires a separate existing RL entry head")
    distribution.requires_grad_(False)
    distribution.mean.entry_recovery.requires_grad_(True)

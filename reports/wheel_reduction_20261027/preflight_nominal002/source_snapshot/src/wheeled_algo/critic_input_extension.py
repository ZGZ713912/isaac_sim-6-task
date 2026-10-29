"""Identity extension of a flat critic and its existing Adam moments."""
from copy import deepcopy

import torch


def extend_flat_critic(checkpoint):
    """Append two zero-weight privileged inputs; preserve all learned columns.

    The deployed 35D actor, its std, and all its Adam state are untouched.
    The new critic function initially ignores the new inputs. Adam's existing
    step counter and moments are retained, with zero moments for new columns.
    """
    key = "mlp.0.weight"
    weight = checkpoint["critic_state_dict"][key]
    if weight.shape != (256, 81):
        raise ValueError("Only the known 81D flat critic can be extended to 83D")
    groups = checkpoint["optimizer_state_dict"]["param_groups"]
    if len(groups) != 2 or groups[1].get("name") != "critic":
        raise ValueError("Critic extension requires the original named actor/critic Adam groups")
    index = groups[1]["params"][0]
    slots = checkpoint["optimizer_state_dict"]["state"][index]
    if any(slots[name].shape != weight.shape for name in ("exp_avg", "exp_avg_sq")):
        raise ValueError("Adam first-layer slots do not match the frozen critic layout")
    migrated = deepcopy(checkpoint)
    migrated["critic_state_dict"][key] = torch.cat((weight.clone(), weight.new_zeros(256, 2)), -1)
    for name, value in slots.items():
        if torch.is_tensor(value) and value.shape == weight.shape:
            migrated["optimizer_state_dict"]["state"][index][name] = torch.cat((value.clone(), value.new_zeros(256, 2)), -1)
    return migrated

"""Authenticate actual native state, matching Adam slots by parameter name."""
from pathlib import Path
import hashlib
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration037_migration_audit import equal
from wheeled_algo.robust_getup import SensorConditionedMean

PREFIX = "mlp.mean.quiet_velocity."
RNG_KEYS = ("environment_rng", "domain_rng", "torch_rng", "cuda_rng")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def named_slots(state, mean):
    groups = {g["name"]: g for g in state["optimizer_state_dict"]["param_groups"]}
    names = ["mlp." + n for n, p in SensorConditionedMean(mean).named_parameters() if p.requires_grad]
    critic_names = list(state["critic_state_dict"])
    assert len(names) == len(groups["actor"]["params"])
    assert len(critic_names) == len(groups["critic"]["params"]) == 8
    result = {}
    for scope, keys in (("actor", names), ("critic", critic_names)):
        result[scope] = {name: state["optimizer_state_dict"]["state"].get(index, {})
                         for name, index in zip(keys, groups[scope]["params"], strict=True)}
    return result


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(x) for x in value)
    return True


def main():
    torch.set_num_threads(1)
    parent_path = RUNS / "quiet_coupled_state045/model_migrated.pt"
    initial_path = RUNS / "pilot026_coupled_state045/model_initial.pt"
    counterpart_path = RUNS / "pilot025_wheel_memory040/model_initial.pt"
    parent, initial, counterpart = [load(p) for p in (parent_path, initial_path, counterpart_path)]
    mean, _ = load_mean(initial_path)
    other_mean, _ = load_mean(counterpart_path, migration_source_root=counterpart_path.parent / "source")
    assert equal(parent["actor_state_dict"], initial["actor_state_dict"])
    assert equal(parent["critic_state_dict"], initial["critic_state_dict"])
    # Serialization may renumber IDs when the number of actor parameters changes.
    # The parent and native initial state have identical ordered group membership.
    for old, new in zip(parent["optimizer_state_dict"]["param_groups"], initial["optimizer_state_dict"]["param_groups"], strict=True):
        assert len(old["params"]) == len(new["params"]) and old["name"] == new["name"]
        assert equal({k: v for k, v in old.items() if k not in ("params", "lr")},
                     {k: v for k, v in new.items() if k not in ("params", "lr")})
        assert new["lr"] == .001
        for before, after in zip(old["params"], new["params"], strict=True):
            assert equal(parent["optimizer_state_dict"]["state"].get(before, {}),
                         initial["optimizer_state_dict"]["state"].get(after, {}))
    assert all(equal(parent["infos"][key], initial["infos"][key]) for key in RNG_KEYS)
    assert all(equal(initial["infos"][key], counterpart["infos"][key]) for key in RNG_KEYS)
    assert equal(initial["critic_state_dict"], counterpart["critic_state_dict"])
    shared = [k for k in initial["actor_state_dict"] if not k.startswith(PREFIX)]
    assert all(equal(initial["actor_state_dict"][k], counterpart["actor_state_dict"][k]) for k in shared)
    added = [k for k in initial["actor_state_dict"] if k.startswith(PREFIX)]
    assert len(added) == 6 and all(torch.count_nonzero(initial["actor_state_dict"][k]) == 0 for k in added)
    slots = named_slots(initial, mean)
    other_slots = named_slots(counterpart, other_mean)
    old_actor_names = [k for k in slots["actor"] if not k.startswith(PREFIX)]
    assert all(equal(slots["actor"][k], other_slots["actor"][k]) for k in old_actor_names)
    assert equal(slots["critic"], other_slots["critic"])
    assert all(not slots["actor"][k] for k in added)
    assert initial["infos"]["quiet_velocity_only_tuning"]
    assert initial["infos"]["completed_updates"] == 441
    settings = initial["infos"]["settings"]
    assert settings["quiet_velocity_feedback_mode"] == "coupled_state"
    assert settings["quiet_exploration_fraction"] == .06 and settings["quiet_wheel_exploration_fraction"] == .35
    progress = json.loads((initial_path.parent / "progress.json").read_text())
    assert progress["trainable_actor_parameters"] == 204
    result = dict(schema="quiet045-native-state-audit-v1",
        initial_checkpoint_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        parent_actor_critic_all_adam_and_rng_exact=True,
        equal_budget040_old_actor_critic_named_adam_and_rng_exact=True,
        unchanged_old_actor_parameter_slots=len(old_actor_names), critic_slots=8,
        added_zero_parameter_slots=6, added_trainable_values=204, initial_clock=441,
        updates_observed_at_initial_audit=progress["new_optimizer_updates"],
        optimizer_matching="authenticated parameter names; serialized integer IDs are not cross-architecture identities",
        hardware_ready=False)
    (REPORTS / "iteration045_initial.json").write_text(json.dumps(result, indent=2) + "\n")
    final_path = initial_path.parent / "model_final.pt"
    if final_path.exists():
        final = load(final_path)
        assert progress["status"] == "completed" and progress["new_optimizer_updates"] == 32
        final_mean, info = load_mean(final_path)
        assert all(finite(final[k]) for k in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"))
        changed = [k for k, v in final["actor_state_dict"].items() if not equal(v, initial["actor_state_dict"][k])]
        assert set(changed) == set(added)
        final_slots = named_slots(final, final_mean)
        assert all(equal(slots["actor"][k], final_slots["actor"][k]) for k in old_actor_names)
        steps = {}
        for key in added:
            assert not slots["actor"][key] and final_slots["actor"][key]
            steps[key] = int(final_slots["actor"][key]["step"].item())
            assert steps[key] == 384
        critic_changed = [k for k, v in final["critic_state_dict"].items() if not equal(v, initial["critic_state_dict"][k])]
        assert len(critic_changed) == 8 and info["completed_updates"] == 473
        final_result = dict(schema="quiet045-final-state-audit-v1",
            checkpoint_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),
            all_model_and_adam_finite=True, frozen_actor_and_std_and_named_adam_exact=True,
            frozen_actor_slots=len(old_actor_names), only_added_actor_changed=changed,
            critic_changed=critic_changed, added_adam_minibatch_steps=steps,
            new_macro_ppo_updates=32, base_clock=473, robust_consumed=961, hardware_ready=False)
        (REPORTS / "iteration045_final_state.json").write_text(json.dumps(final_result, indent=2) + "\n")
        print(json.dumps(final_result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

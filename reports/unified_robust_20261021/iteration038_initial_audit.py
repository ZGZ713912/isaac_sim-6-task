"""Compare native initial state by parameter order, including Adam ID renumbering."""
from pathlib import Path
import hashlib
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(Path(__file__).parent)]
from iteration038_migration_audit import equal


def main():
    torch.set_num_threads(1)
    parent_path = ROOT / "runs/unified_robust_20261022/quiet_wheel_velocity038/model_migrated.pt"
    initial_path = ROOT / "runs/unified_robust_20261022/pilot023_wheel_velocity038/model_initial.pt"
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    assert equal(parent["actor_state_dict"], initial["actor_state_dict"])
    assert equal(parent["critic_state_dict"], initial["critic_state_dict"])
    old_groups, new_groups = parent["optimizer_state_dict"]["param_groups"], initial["optimizer_state_dict"]["param_groups"]
    count = 0
    for old, new in zip(old_groups, new_groups, strict=True):
        assert len(old["params"]) == len(new["params"]) and old["name"] == new["name"]
        assert {k: v for k, v in old.items() if k not in ("params", "lr")} == {k: v for k, v in new.items() if k not in ("params", "lr")}
        assert new["lr"] == (.001)
        for before, after in zip(old["params"], new["params"], strict=True):
            assert equal(parent["optimizer_state_dict"]["state"].get(before, {}),
                         initial["optimizer_state_dict"]["state"].get(after, {}))
            count += 1
    rng_keys = ("environment_rng", "domain_rng", "torch_rng", "cuda_rng")
    assert all(equal(parent["infos"][key], initial["infos"][key]) for key in rng_keys)
    assert initial["infos"]["quiet_velocity_only_tuning"]
    assert initial["infos"]["completed_updates"] == 441
    task = initial["infos"]["settings"]
    assert task["quiet_velocity_feedback"] and task["quiet_exploration_fraction"] == .1
    assert task["quiet_velocity_feedback_mode"] == "wheels"
    assert task["standing_drift_speed_scale_m_s"] == .02
    progress = json.loads((initial_path.parent / "progress.json").read_text())
    assert progress["trainable_actor_parameters"] == 12
    result = dict(schema="quiet038-native-initial-v1", initial_checkpoint_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        actor_critic_and_all_old_adam_exact=True, optimizer_parameter_slots_checked=count,
        optimizer_scope="old/new named-group order authenticated; serialized IDs may be renumbered",
        all_four_rng_exact=True, declared_lr_revision={"actor": [old_groups[0]["lr"], .001], "critic": .001},
        trainable_actor_parameters=12, initial_clock=441,
        updates_observed_at_audit=progress["new_optimizer_updates"])
    (ROOT / "reports/unified_robust_20261021/iteration038_initial.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

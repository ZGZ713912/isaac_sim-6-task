"""Authenticate zero-function extension and the complete old optimizer/RNG state."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from export_robust_getup import load_mean
from wheeled_algo.robust_getup import SensorConditionedMean


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def main():
    torch.set_num_threads(1)
    parent = ROOT / "runs/unified_robust_20261022/quiet_velocity036/model_migrated.pt"
    checkpoint = ROOT / "runs/unified_robust_20261022/quiet_wheel_memory040/model_migrated.pt"
    old = torch.load(parent, map_location="cpu", weights_only=False)
    new = torch.load(checkpoint, map_location="cpu", weights_only=False)
    old_mean, _ = load_mean(parent, migration_source_root=parent.parent / "source")
    new_mean, _ = load_mean(checkpoint)
    assert all(equal(v, new["actor_state_dict"][k]) for k, v in old["actor_state_dict"].items())
    added = set(new["actor_state_dict"]) - set(old["actor_state_dict"])
    assert added == {f"mlp.mean.quiet_velocity.{name}.weight" for name in ("mid", "low", "high")}
    assert all(torch.count_nonzero(new["actor_state_dict"][key]) == 0 and new["actor_state_dict"][key].shape == (2, 4) for key in added)
    assert new["infos"]["settings"]["quiet_velocity_feedback_mode"] == "wheel_memory"
    assert equal(old["critic_state_dict"], new["critic_state_dict"])
    old_optimizer, new_optimizer = old["optimizer_state_dict"], new["optimizer_state_dict"]
    assert equal(old_optimizer["state"], new_optimizer["state"])
    names = [name for name, p in SensorConditionedMean(new_mean).named_parameters() if p.requires_grad]
    for before, after in zip(old_optimizer["param_groups"], new_optimizer["param_groups"], strict=True):
        assert equal({k: v for k, v in before.items() if k != "params"},
                     {k: v for k, v in after.items() if k != "params"})
        if before["name"] == "actor":
            assert after["params"][:-3] == before["params"] and len(names) == len(after["params"])
            assert names[-3:] == [f"mean.quiet_velocity.{name}.weight" for name in ("mid", "low", "high")]
            assert all(key not in new_optimizer["state"] for key in after["params"][-3:])
        else:
            assert equal(before, after)
    rng_keys = ("environment_rng", "domain_rng", "torch_rng", "cuda_rng")
    assert all(equal(old["infos"][key], new["infos"][key]) for key in rng_keys)
    allowed = {"version", "quiet_velocity_feedback", "quiet_exploration_fraction", "quiet_velocity_feedback_mode", "quiet_wheel_exploration_fraction"}
    differences = {k: [old["infos"]["settings"].get(k), v] for k, v in new["infos"]["settings"].items()
                   if old["infos"]["settings"].get(k) != v}
    assert set(differences) <= allowed
    assert old["infos"]["completed_updates"] == new["infos"]["completed_updates"] == 441
    data = np.load(ROOT / "runs/unified_robust_20261022/quiet_velocity_baseline036/robust/trajectory.npz")
    static = torch.from_numpy(data["observations"][::16].reshape(-1, 35).copy())
    other = np.load(ROOT / "runs/unified_robust_20261022/side_confirmation032/robust/trajectory.npz")
    motion_falls = torch.from_numpy(other["observations"][::40].reshape(-1, 35).copy())
    results = {}
    with torch.inference_mode():
        for name, x in (("expanded_static", static), ("motion_falls", motion_falls)):
            before, after = old_mean(x), new_mean(x)
            assert torch.equal(before, after)
            results[name] = dict(observations=len(x), maximum_raw_action_error=0., exact=True)
    result = dict(schema="quiet040-migration-audit-v1", checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), clock=441,
        new_optimizer_updates=0, robust_consumed=897, added_actor_parameters=24,
        all_existing_actor_critic_optimizer_groups_moments_rng_exact=True,
        settings_revision=differences, numeric=results, hardware_ready=False)
    (ROOT / "reports/unified_robust_20261021/iteration040_migration.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

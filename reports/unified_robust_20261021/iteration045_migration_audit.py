"""Authenticate the204-value coupled-state extension and unchanged old state."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from wheeled_algo.robust_getup import SensorConditionedMean
from iteration044_selective_memory import equal

torch.set_num_threads(1)
parent = ROOT / "runs/unified_robust_20261022/quiet_velocity036/model_migrated.pt"
checkpoint = ROOT / "runs/unified_robust_20261022/quiet_coupled_state045/model_migrated.pt"
old, new = [torch.load(p, map_location="cpu", weights_only=False) for p in (parent, checkpoint)]
old_mean, _ = load_mean(parent, migration_source_root=parent.parent / "source")
new_mean, _ = load_mean(checkpoint)
assert all(equal(v, new["actor_state_dict"][k]) for k, v in old["actor_state_dict"].items())
added = set(new["actor_state_dict"]) - set(old["actor_state_dict"])
expected = {f"mlp.mean.quiet_velocity.{name}.{field}" for name in ("mid", "low", "high") for field in ("weight", "bias")}
assert added == expected
assert all(torch.count_nonzero(new["actor_state_dict"][k]) == 0 for k in added)
assert all(new["actor_state_dict"][k].shape == ((4, 16) if k.endswith("weight") else (4,)) for k in added)
assert equal(old["critic_state_dict"], new["critic_state_dict"])
before_opt, after_opt = old["optimizer_state_dict"], new["optimizer_state_dict"]
assert equal(before_opt["state"], after_opt["state"])
names = [n for n, p in SensorConditionedMean(new_mean).named_parameters() if p.requires_grad]
for before, after in zip(before_opt["param_groups"], after_opt["param_groups"], strict=True):
    assert equal({k: v for k, v in before.items() if k != "params"}, {k: v for k, v in after.items() if k != "params"})
    if before["name"] == "actor":
        assert after["params"][:-6] == before["params"] and len(names) == len(after["params"])
        assert names[-6:] == [f"mean.quiet_velocity.{name}.{field}" for name in ("mid", "low", "high") for field in ("weight", "bias")]
        assert all(k not in after_opt["state"] for k in after["params"][-6:])
    else:
        assert equal(before, after)
assert all(equal(old["infos"][k], new["infos"][k]) for k in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"))
differences = {k: [old["infos"]["settings"].get(k), v] for k, v in new["infos"]["settings"].items() if old["infos"]["settings"].get(k) != v}
assert set(differences) <= {"version", "quiet_velocity_feedback", "quiet_exploration_fraction", "quiet_velocity_feedback_mode", "quiet_wheel_exploration_fraction"}
results = {}
with torch.inference_mode():
    for name, run, stride in (("static", "quiet_velocity_baseline036", 16), ("motion_falls", "side_confirmation032", 40)):
        data = np.load(ROOT / "runs/unified_robust_20261022" / run / "robust/trajectory.npz")
        x = torch.from_numpy(data["observations"][::stride].reshape(-1, 35).copy())
        assert torch.equal(old_mean(x), new_mean(x))
        results[name] = dict(observations=len(x), maximum_raw_action_error=0., exact=True)
result = dict(schema="quiet045-migration-audit-v1", checkpoint=str(checkpoint.relative_to(ROOT)),
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), base_clock=new["infos"]["completed_updates"],
    optimizer_updates=0, robust_consumed=929, added_actor_parameters=204,
    all_existing_actor_critic_adam_groups_moments_rng_exact=True, settings_revision=differences,
    numeric=results, hardware_ready=False)
(REPORTS / "iteration045_migration.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)

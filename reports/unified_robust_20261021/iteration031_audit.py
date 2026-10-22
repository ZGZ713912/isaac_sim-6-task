"""Reproduce the zero-PPO height-expert state and real-observation audit."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from export_robust_getup import load_mean
from wheeled_algo.robust_getup import SensorConditionedMean

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/unified_robust_20261022"
BASE = RUNS / "pilot014_standing_drift026/model_final.pt"
LOW = RUNS / "pilot015_standing_continue029_resume001/model_final.pt"
NEW = RUNS / "quiet_height_experts031/model_migrated.pt"


def exact(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def load(path):
    mean, _ = load_mean(path, migration_source_root=path.parent / "source")
    state = torch.load(path, map_location="cpu", weights_only=False)
    group = next(g for g in state["optimizer_state_dict"]["param_groups"] if g["name"] == "actor")
    names = [n for n, p in SensorConditionedMean(mean).named_parameters() if p.requires_grad]
    return mean, state, dict(zip(names, group["params"], strict=True))


base, old, old_ids = load(BASE)
low, low_state, low_ids = load(LOW)
new, state, new_ids = load(NEW)
checks = dict(original_actor_tensors_exact=all(torch.equal(v, state["actor_state_dict"][k])
    for k, v in old["actor_state_dict"].items()),
    critic_exact=exact(old["critic_state_dict"], state["critic_state_dict"]),
    original_optimizer_moments_exact=all(exact(v, state["optimizer_state_dict"]["state"][k])
        for k, v in old["optimizer_state_dict"]["state"].items()))
for target, source, ids in (("quiet_low", low_state, low_ids), ("quiet_high", old, old_ids)):
    names = [n for n in old_ids if n.startswith("mean.quiet.")]
    checks[target + "_weights_exact"] = all(torch.equal(source["actor_state_dict"]["mlp." + n],
        state["actor_state_dict"]["mlp." + n.replace("mean.quiet.", "mean." + target + ".", 1)]) for n in names)
    checks[target + "_moments_exact"] = all(exact(source["optimizer_state_dict"]["state"][ids[n]],
        state["optimizer_state_dict"]["state"][new_ids[n.replace("mean.quiet.", "mean." + target + ".", 1)]]) for n in names)
z = np.load(RUNS / "standing_confirmation026/robust/trajectory.npz")
obs = torch.from_numpy(z["observations"][::100].copy().reshape(-1, 35))
errors = {}
with torch.inference_mode():
    for height, reference in ((.23, low), (.305, base), (.42, base)):
        x = obs.clone()
        x[:, 3] = height * 5.
        errors[str(height)] = float((new(x) - reference(x)).abs().max())
    for label in ("motion", "inverted"):
        x = obs.clone()
        if label == "motion":
            x[:, 0] = 1.
        else:
            x[:, 7:10] = torch.tensor([0., 0., 1.])
        errors[label] = float((new(x) - base(x)).abs().max())
checks["real_observation_endpoint_outputs_exact"] = all(v == 0. for v in errors.values())
checks["environment_rng_exact"] = exact(old["infos"]["environment_rng"], state["infos"]["environment_rng"])
checks["domain_rng_exact"] = exact(old["infos"]["domain_rng"], state["infos"]["domain_rng"])
assert all(checks.values()), checks
receipt = dict(schema="robust-quiet-height-experts031-audit-v1", checks=checks,
    max_action_errors=errors, real_observations_per_case=len(obs), optimizer_updates=0,
    source_checkpoint_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
    low_checkpoint_sha256=hashlib.sha256(LOW.read_bytes()).hexdigest(),
    migrated_sha256=hashlib.sha256(NEW.read_bytes()).hexdigest(),
    robust_consumed=513, selected_base_lineage=345, low_head_source_lineage=408,
    combined_lineage_is_not_sum=True, long_training_admitted=False, hardware_motion=False)
(Path(__file__).parent / "iteration031_migration.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))

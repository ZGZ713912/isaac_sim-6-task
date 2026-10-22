"""Single-factor zero-PPO ablation: remove only045's three new affine biases."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration037_migration_audit import equal
from wheeled_algo.robust_getup import SensorConditionedMean, quiet_gate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    parent = RUNS / "pilot026_coupled_state045/model_final.pt"
    directory = RUNS / "quiet_bias_ablation047"
    assert not directory.exists(), "Refuse to overwrite ablation"
    original = torch.load(parent, map_location="cpu", weights_only=False)
    state = deepcopy(original)
    mean, info = load_mean(parent)
    names = ["mlp." + n for n, p in SensorConditionedMean(mean).named_parameters() if p.requires_grad]
    actor_group = next(g for g in state["optimizer_state_dict"]["param_groups"] if g["name"] == "actor")
    assert len(names) == len(actor_group["params"]) == 31
    mapping = dict(zip(names, actor_group["params"], strict=True))
    zeroed = [f"mlp.mean.quiet_velocity.{name}.bias" for name in ("mid", "low", "high")]
    removed = {}
    for key in zeroed:
        assert state["actor_state_dict"][key].shape == (4,)
        state["actor_state_dict"][key].zero_()
        index = mapping[key]
        assert index in state["optimizer_state_dict"]["state"]
        removed[key] = index
        del state["optimizer_state_dict"]["state"][index]
    assert [k for k, v in state["actor_state_dict"].items() if not equal(v, original["actor_state_dict"][k])] == zeroed
    assert equal(state["critic_state_dict"], original["critic_state_dict"])
    assert equal(state["optimizer_state_dict"]["param_groups"], original["optimizer_state_dict"]["param_groups"])
    assert all(equal(v, original["optimizer_state_dict"]["state"][k]) for k, v in state["optimizer_state_dict"]["state"].items())
    assert set(original["optimizer_state_dict"]["state"]) - set(state["optimizer_state_dict"]["state"]) == set(removed.values())
    assert all(equal(state["infos"][k], original["infos"][k]) for k in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"))
    state["infos"]["post_training_intervention"] = dict(schema="quiet047-affine-bias-ablation-v1",
        source_checkpoint_sha256=sha(parent), zeroed_actor_bias=zeroed, reset_only_bias_adam=removed,
        purpose="Causal diagnostic only; distinguish constant affine terms from learned state-dependent feedback",
        new_optimizer_updates=0, robust_consumed=961, hardware_ready=False)
    directory.mkdir()
    shutil.copytree(parent.parent / "source", directory / "source")
    shutil.copy2(parent.parent / "task_contract.json", directory / "task_contract.json")
    checkpoint = directory / "model_ablated.pt"
    torch.save(state, checkpoint)
    ablated, _ = load_mean(checkpoint)
    data = np.load(RUNS / "side_confirmation032/robust/trajectory.npz")
    x = torch.from_numpy(data["observations"][::40].reshape(-1, 35).copy())
    with torch.inference_mode():
        a, b = mean(x), ablated(x)
        inactive = quiet_gate(x).flatten() == 0.
        assert inactive.any() and torch.equal(a[inactive], b[inactive])
        numeric = dict(observations=len(x), quiet_inactive_observations=int(inactive.sum()),
                       quiet_inactive_raw_action_error=0., maximum_raw_action_change=float((a - b).abs().max()))
    result = dict(schema="quiet047-affine-bias-ablation-plan-v1", checkpoint=str(checkpoint.relative_to(ROOT)),
        source_checkpoint_sha256=sha(parent), checkpoint_sha256=sha(checkpoint),
        base_clock=info["completed_updates"], optimizer_updates=0, robust_consumed=961,
        only_changed_actor_parameters=zeroed, reset_only_bias_adam=removed,
        all_other_actor_critic_adam_groups_rng_exact=True, numeric=numeric,
        planned_native=dict(episodes=48, per_height=16, seconds=12., seed=211322,
                           domain_scale=.35, ground_friction=[.3, .5], isolated_origins=True),
        full_suite_planned=False, long_training_admitted=False, hardware_ready=False)
    (REPORTS / "iteration047_plan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

"""Real checkpoint/Adam/gradient checks before any entry-only PPO."""
import hashlib
import json
from pathlib import Path
import sys

sys.path[:0] = ["src", "scripts"]
import numpy as np
import torch
from torch import nn
from export_robust_getup import load_mean
from wheeled_algo.entry_optimizer import prepare_entry_optimizer_parameters, freeze_except_entry
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.gap_recovery_exploration import GapRecoveryNoiseMean

torch.set_num_threads(1)
base = Path("runs/robust_policy_repair_20261024")
source = base / "final_pair_entry_neutral001/model_migrated.pt"
original = torch.load(source, map_location="cpu", weights_only=False)
old, _ = load_mean(source)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


rows = []
for weight in (0, 1):
    cp = base / f"entry_optimizer_support{weight}_source001/model_migrated.pt"
    state = torch.load(cp, map_location="cpu", weights_only=False)
    mean, info = load_mean(cp)
    checks = dict(actor_exact=equal(original["actor_state_dict"], state["actor_state_dict"]),
        critic_exact=equal(original["critic_state_dict"], state["critic_state_dict"]),
        clock_exact=info["completed_updates"] == original["infos"]["completed_updates"])
    for key in ("domain_rng", "environment_rng", "torch_rng", "cuda_rng"):
        checks[key + "_exact"] = equal(original["infos"][key], info[key])
    before, after = original["optimizer_state_dict"], state["optimizer_state_dict"]
    for group0, group1 in zip(before["param_groups"], after["param_groups"], strict=True):
        for id0, id1 in zip(group0["params"], group1["params"], strict=False):
            assert equal(before["state"].get(id0), after["state"].get(id1)), (group0["name"], id0, id1)
    checks["old_Adam_slots_and_steps_exact_by_group_order"] = True
    settings = info["settings"]
    distribution = GapRecoveryNoiseMean(mean, settings["quiet_exploration_fraction"],
        settings["quiet_wheel_exploration_fraction"], settings["gap_recovery_leg_exploration_fraction"])
    distribution.load_state_dict({k.removeprefix("mlp."): v for k, v in state["actor_state_dict"].items()}, strict=True)
    prepare_entry_optimizer_parameters(mean, settings)
    parameters = [p for p in distribution.parameters() if p.requires_grad]
    assert len(parameters) == len(after["param_groups"][0]["params"])
    critic_parameters = [nn.Parameter(torch.zeros_like(after["state"][i]["exp_avg"]))
                         for i in after["param_groups"][1]["params"]]
    optimizer = torch.optim.Adam([dict(params=parameters, name="actor"), dict(params=critic_parameters, name="critic")])
    optimizer.load_state_dict(after)
    checks["native_Adam_load_save_layout_exact"] = equal(after, optimizer.state_dict())
    freeze_except_entry(distribution)
    checks["trainable_entry_only"] = all(p.requires_grad == name.startswith("mean.entry_recovery.")
                                         for name, p in distribution.named_parameters())
    checks["entry_parameter_count"] = sum(p.numel() for p in distribution.parameters() if p.requires_grad) == 21894
    frames = []
    replay_rows = []
    with torch.inference_mode():
        for run in ("diagnosis012_final_circular_fixed/evaluation", "diagnosis012_final_arithmetic_nominal/robust",
                    "diagnosis012_final_arithmetic_rmcs/evaluation"):
            path = base / run / "trajectory.npz"
            observations = np.load(path)["observations"].reshape(-1, 35)
            exact = all(torch.equal(old(obs), mean(obs)) for obs in torch.tensor(observations).split(2048))
            assert exact
            replay_rows.append(dict(run=run, frames=len(observations), means_bit_exact=exact, sha256=sha(path)))
            frames.append(observations[::100])
    obs = torch.tensor(np.concatenate(frames))
    gate = handoff_entry_gate(obs).ravel()
    gradient = {}
    for label, selected in (("entry_full", gate == 1), ("entry_zero", gate == 0)):
        distribution.zero_grad(set_to_none=True)
        out = mean(obs[selected][:256])
        (out[:, 0] + .3 * out[:, 1] + .07 * out[:, 4]).mean().backward()
        norm = sum(float(p.grad.square().sum()) for p in mean.entry_recovery.parameters() if p.grad is not None) ** .5
        gradient[label + "_L2"] = norm
    checks["nonzero_entry_gradient"] = gradient["entry_full_L2"] > 0.
    checks["entry_zero_gradient_exactly_zero"] = gradient["entry_zero_L2"] == 0.
    assert all(checks.values()), checks
    rows.append(dict(checkpoint=str(cp), sha256=sha(cp), support_weight=weight,
                     checks=checks, replays=replay_rows, gradient=gradient))
out = dict(schema="entry-only-real-checkpoint-preflight-v1", source_sha256=sha(source), rows=rows,
    optimizer_updates=0, new_PPO=0, actual_experiment_consumed=2587,
    native_equality_still_required=True, qualification=None, hardware_ready=False)
Path("reports/robust_policy_repair_20261024/entry_learning_preflight001.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))

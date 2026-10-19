"""Read-only verification of real 448/512 branch composition before any PPO."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.guarded_feedback_residual import restore_command_branch

RUN = ROOT / "runs/flat_base_20261019"
source = ROOT / "models/wheel_leg_flat_base_20261018/source.onnx"
current = torch.load(RUN / "pilot011/model_000512.pt", map_location="cpu", weights_only=True)
previous = torch.load(RUN / "pilot010/model_000448.pt", map_location="cpu", weights_only=True)
mean = mean_from_checkpoint(current, source)
names = ["distribution.std_param"] + ["mlp." + n for n, _ in mean.named_parameters()]
composed, proof = restore_command_branch(current, previous, names)
actual = mean_from_checkpoint(composed, source)
expected = deepcopy(mean)
expected.command.load_state_dict(mean_from_checkpoint(previous, source).state_dict(), strict=True)


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


checks = {}
for key in ("critic_state_dict", "torch_rng_state", "cuda_rng_state", "env_generator_state"):
    checks[key] = exact(current[key], composed[key])
checks["optimizer_groups_exact_current"] = exact(current["optimizer_state_dict"]["param_groups"], composed["optimizer_state_dict"]["param_groups"])
checks["non_command_actor_exact_current"] = all(torch.equal(v, composed["actor_state_dict"][n])
    for n, v in current["actor_state_dict"].items() if not n.startswith("mlp.command."))
checks["command_weights_exact448"] = all(torch.equal(v, previous["actor_state_dict"][n.replace("mlp.command.", "mlp.", 1)])
    for n, v in composed["actor_state_dict"].items() if n.startswith("mlp.command."))
ids = dict(zip(names, current["optimizer_state_dict"]["param_groups"][0]["params"], strict=True))
command_ids = {i for n, i in ids.items() if n.startswith("mlp.command.residual.")}
checks["all_other_Adam_slots_exact_current"] = all(exact(v, composed["optimizer_state_dict"]["state"][i])
    for i, v in current["optimizer_state_dict"]["state"].items() if i not in command_ids)
checks["four_command_Adam_slots_exact448"] = all(exact(composed["optimizer_state_dict"]["state"][i], previous["optimizer_state_dict"]["state"][i]) for i in command_ids)
checks["lifetime_counters_and_consumption_exact_current"] = all(composed["infos"][k] == current["infos"][k]
    for k in ("successful_updates_total", "actor_updates_total", "attempted_updates_total", "consumed_optimizer_updates_total", "feedback_branch_actor_updates"))
checks["selected_command_learning_count44"] = composed["infos"]["active_branch_actor_updates"] == 44
inputs = []
for cohort in ("rmcs512_basic001", "rmcs512_envelope001", "ablation512_parent_command001"):
    d = np.load(RUN / cohort / "traces.npz")
    values = d["observations"][d["active"]]
    inputs.append(values[np.linspace(0, len(values) - 1, 1024).astype(int)])
x = torch.from_numpy(np.concatenate(inputs))
with torch.no_grad():
    checks["mean_exact_native_counterfactual_composition"] = torch.equal(actual(x), expected(x))
receipt = dict(schema="flat-576-parent-composition-CPU-v1", checks=checks, passed=all(checks.values()),
    actual_observation_samples=len(x), new_optimizer_updates=0, qualification_eligible=False, composition=proof,
    parent_checkpoint_sha256=hashlib.sha256((RUN / "pilot011/model_000512.pt").read_bytes()).hexdigest(),
    command_checkpoint_sha256=hashlib.sha256((RUN / "pilot010/model_000448.pt").read_bytes()).hexdigest())
path = Path(__file__).with_name("parent576_composition_CPU.json")
path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
if not receipt["passed"]:
    raise SystemExit(1)
